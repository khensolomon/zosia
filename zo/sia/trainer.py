# zo/sia/trainer.py
#
# What it does:
# This module's Trainer class has been updated to capture and save detailed
# training metadata (performance, duration, etc.) into the model checkpoint,
# making them fully self-documented.

import torch
import torch.nn as nn
import random
import os
import time
from datetime import datetime
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

from zo.sia.data_utils import normalize_string, tensor_from_sentence

class Trainer:
    def __init__(self, config, encoder, decoder, encoder_optimizer, decoder_optimizer, device):
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.device = device
        self.criterion = nn.NLLLoss()

    def _train_batch(self, input_tensors, target_tensors, max_length, sos_token_idx, eos_token_idx):
        batch_size = input_tensors.size(1)
        encoder_hidden = self.encoder.initHidden(self.device, batch_size=batch_size)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        target_length = target_tensors.size(0)
        loss = 0

        encoder_outputs, encoder_hidden = self.encoder(input_tensors, encoder_hidden)

        decoder_input = torch.tensor([[sos_token_idx] * batch_size], device=self.device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < 0.5 else False

        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensors[di])
                decoder_input = target_tensors[di].unsqueeze(0)
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach().view(1, -1)
                loss += self.criterion(decoder_output, target_tensors[di])
                if (decoder_input == eos_token_idx).all():
                    break
        
        loss.backward()

        clip = self.config.training.initial_training.clip
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss.item() / target_length

    def train(self, training_pairs, input_lang, output_lang, max_length):
        """The main training loop. Returns training duration and final loss."""
        print("\n--- Starting Model Training ---")
        start_time = time.time()
        
        pad_token_idx = input_lang.word2index[self.config.tokenizer.special_tokens.pad]
        sos_token_idx = input_lang.word2index[self.config.tokenizer.special_tokens.sos]
        eos_token_idx = input_lang.word2index[self.config.tokenizer.special_tokens.eos]
        
        self.criterion.ignore_index = pad_token_idx
        n_iters = self.config.training.initial_training.n_iterations
        batch_size = self.config.training.initial_training.batch_size
        
        training_iterator = tqdm(range(1, n_iters + 1), desc="Training Progress")
        last_loss = 0.0
        
        for iter_num in training_iterator:
            training_batch = [random.choice(training_pairs) for _ in range(batch_size)]
            input_tensors = torch.cat([tensor_from_sentence(input_lang, p[0], max_length, self.device) for p in training_batch], dim=1)
            target_tensors = torch.cat([tensor_from_sentence(output_lang, p[1], max_length, self.device) for p in training_batch], dim=1)
            
            loss = self._train_batch(input_tensors, target_tensors, max_length, sos_token_idx, eos_token_idx)
            last_loss = loss
            
            if iter_num % 100 == 0:
                training_iterator.set_postfix({"Batch Loss": f"{loss:.4f}"})
        
        duration = time.time() - start_time
        print(f"--- Training Complete in {duration:.2f} seconds ---")
        return duration, last_loss

    def evaluate(self, test_pairs, input_lang, output_lang, max_length):
        print("\n--- Calculating BLEU score on test set ---")
        sos_token_idx = input_lang.word2index[self.config.tokenizer.special_tokens.sos]
        eos_token_idx = input_lang.word2index[self.config.tokenizer.special_tokens.eos]
        self.encoder.eval()
        self.decoder.eval()
        candidates, references = [], []
        for src, tgt in tqdm(test_pairs, desc="Evaluating on Test Set"):
            candidate = self._translate_sentence(src, input_lang, output_lang, max_length, sos_token_idx, eos_token_idx)
            candidates.append(candidate.split())
            references.append([tgt.split()])
        bleu = corpus_bleu(references, candidates)
        self.encoder.train()
        self.decoder.train()
        print(f"Corpus BLEU score: {bleu * 100:.2f}")
        return bleu

    def _translate_sentence(self, sentence, input_lang, output_lang, max_length, sos_token_idx, eos_token_idx):
        with torch.no_grad():
            normalized_sentence = normalize_string(sentence)
            input_tensor = tensor_from_sentence(input_lang, normalized_sentence, max_length, self.device)
            encoder_hidden = self.encoder.initHidden(self.device)
            encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
            decoder_input = torch.tensor([[sos_token_idx]], device=self.device)
            decoder_hidden = encoder_hidden
            decoded_words = []
            for _ in range(max_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == eos_token_idx:
                    break
                decoded_words.append(output_lang.index2word[topi.item()])
                decoder_input = topi.detach()
            return ' '.join(decoded_words)

    def save_checkpoint(self, filename, input_lang, output_lang, max_length, training_metadata):
        """Saves the model and all training metadata to a file."""
        print("\n--- Saving Model Checkpoint ---")
        checkpoint_path = os.path.join(self.config.paths.experiments, filename)
        os.makedirs(self.config.paths.experiments, exist_ok=True)

        # Combine all data into a single dictionary
        save_data = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'input_lang': input_lang,
            'output_lang': output_lang,
            'config': self.config,
            'max_length': max_length,
            'creation_timestamp': time.time(),
            # Add the new metadata dictionary
            'training_metadata': training_metadata
        }

        torch.save(save_data, checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")

# Script to load a trained model and perform translation.

import torch
import torch.nn as nn
import os
import argparse
import sentencepiece as spm
import yaml
from typing import List

from src.models.transformer_model import Transformer, Encoder, Decoder
from src.utils.general_utils import load_config, get_device, load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Translator:
    """
    Handles loading a trained NMT model and performing translations.
    """
    def __init__(self,
                 model_path: str,
                 device: torch.device):
        """
        Args:
            model_path (str): Path to the saved model checkpoint (.pt file).
            device (torch.device): Device to run the model on.
        """
        self.device = device
        self.model, self.sp_model, self.data_config, self.model_config = self._load_model_and_artifacts(model_path)
        self.model.eval() # Set model to evaluation mode

        self.sos_token_id = self.sp_model.bos_id()
        self.eos_token_id = self.sp_model.eos_id()
        self.pad_token_id = self.sp_model.pad_id()
        self.max_seq_len = self.model_config["max_seq_len"]


    def _load_model_and_artifacts(self, model_path: str):
        """Loads the model, tokenizer, and configs from a checkpoint."""
        logger.info(f"Loading model from checkpoint: {model_path}")
        checkpoint = load_checkpoint(model_path, self.device)

        data_config = checkpoint.get('data_config')
        model_config = checkpoint.get('model_config')
        
        if not data_config or not model_config:
            raise ValueError("Checkpoint does not contain necessary data_config or model_config.")

        # Load SentencePiece model
        sp_model_path = checkpoint.get('sp_model_path')
        if not sp_model_path or not os.path.exists(sp_model_path):
            # Fallback if sp_model_path not directly in checkpoint (e.g., older checkpoints)
            vocab_dir = data_config.get("vocab_dir")
            tokenizer_prefix = data_config.get("tokenizer_prefix")
            sp_model_path = os.path.join(vocab_dir, f"{tokenizer_prefix}.model")
            logger.warning(f"sp_model_path not found in checkpoint. Attempting to load from {sp_model_path}")

        if not os.path.exists(sp_model_path):
             raise FileNotFoundError(f"SentencePiece model not found at {sp_model_path}. "
                                     "Ensure data preparation was run and tokenizer is saved.")

        sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        logger.info(f"Loaded SentencePiece model from {sp_model_path}. Vocab size: {sp_model.get_piece_size()}")

        # Initialize model architecture
        src_pad_idx = sp_model.pad_id()
        trg_pad_idx = sp_model.pad_id()
        
        # Ensure vocab size is correct
        input_dim = sp_model.get_piece_size()
        output_dim = sp_model.get_piece_size()

        enc = Encoder(input_dim=input_dim,
                      hid_dim=model_config["hidden_dim"],
                      n_layers=model_config["enc_layers"],
                      n_heads=model_config["enc_heads"],
                      pf_dim=model_config["enc_pf_dim"],
                      dropout=model_config["enc_dropout"],
                      max_seq_len=model_config["max_seq_len"],
                      device=self.device)

        dec = Decoder(output_dim=output_dim,
                      hid_dim=model_config["hidden_dim"],
                      n_layers=model_config["dec_layers"],
                      n_heads=model_config["dec_heads"],
                      pf_dim=model_config["dec_pf_dim"],
                      dropout=model_config["dec_dropout"],
                      max_seq_len=model_config["max_seq_len"],
                      device=self.device)

        model = Transformer(enc, dec, src_pad_idx, trg_pad_idx, self.device).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state loaded successfully.")

        return model, sp_model, data_config, model_config

    def translate_sentence(self,
                           sentence: str,
                           beam_size: int = 5,
                           max_decode_len_ratio: float = 1.5) -> str:
        """
        Translates a single sentence using the loaded model.
        Uses beam search for decoding.

        Args:
            sentence (str): The input sentence in the source language (Zo).
            beam_size (int): The size of the beam for beam search decoding.
            max_decode_len_ratio (float): Max generated length = source_len * ratio.

        Returns:
            str: The translated sentence in the target language (English).
        """
        self.model.eval() # Ensure model is in eval mode

        if not isinstance(sentence, str):
            raise TypeError(f"Input sentence must be a string, got {type(sentence)}")

        # Tokenize source sentence
        src_tokens = self.sp_model.encode_as_ids(sentence.strip())
        
        # Add SOS and EOS to source, truncate if too long
        src_tensor = torch.tensor([self.sp_model.bos_id()] + src_tokens + [self.sp_model.eos_id()], dtype=torch.long, device=self.device)
        src_tensor = src_tensor[:self.max_seq_len].unsqueeze(0) # Add batch dimension

        src_mask = self.model.make_src_mask(src_tensor)

        # Encode source sentence
        with torch.no_grad():
            enc_src = self.model.encoder(src_tensor, src_mask)

        # --- Beam Search Decoding ---
        # Initialize hypotheses: (sequence, score)
        # Sequence is a list of token IDs, score is log probability
        hypotheses = [(torch.tensor([self.sos_token_id], dtype=torch.long, device=self.device), 0.0)]

        max_output_length = min(self.max_seq_len, int(src_tensor.shape[1] * max_decode_len_ratio))
        if max_output_length < 5: # Ensure a minimum length
            max_output_length = 5

        for _ in range(max_output_length - 1): # -1 because SOS is already there
            new_hypotheses = []
            for seq_tensor, score in hypotheses:
                if seq_tensor[-1].item() == self.eos_token_id: # If sequence ended, keep it
                    new_hypotheses.append((seq_tensor, score))
                    continue

                # Prepare decoder input and mask
                trg_mask = self.model.make_trg_mask(seq_tensor.unsqueeze(0)) # Add batch dimension
                
                with torch.no_grad():
                    output = self.model.decoder(seq_tensor.unsqueeze(0), enc_src, trg_mask, src_mask)
                
                # Get probabilities for the next token
                next_token_logits = output[0, -1, :] # (vocab_size)
                log_probs = torch.log_softmax(next_token_logits, dim=-1) # Log probabilities

                # Get top `beam_size` candidates
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for log_prob, index in zip(top_log_probs, top_indices):
                    next_seq_tensor = torch.cat([seq_tensor, index.unsqueeze(0)])
                    new_score = score + log_prob.item()
                    new_hypotheses.append((next_seq_tensor, new_score))

            # Select top `beam_size` hypotheses from the combined list
            hypotheses = sorted(new_hypotheses, key=lambda x: x[1] / (len(x[0])-1), reverse=True)[:beam_size] # Normalize by length

            if all(hyp[-1].item() == self.eos_token_id for hyp, _ in hypotheses):
                break # All beams ended

        # Choose the best hypothesis (highest normalized score)
        best_hypothesis, _ = hypotheses[0]

        # Decode token IDs back to text
        translated_ids = best_hypothesis.tolist()
        # Remove SOS, EOS, PAD tokens for final output
        final_translation = []
        for token_id in translated_ids:
            if token_id == self.sos_token_id:
                continue
            if token_id == self.eos_token_id:
                break # Stop decoding after EOS
            if token_id == self.pad_token_id:
                continue
            final_translation.append(token_id)
        
        translated_text = self.sp_model.decode(final_translation)
        
        return translated_text

def main():
    parser = argparse.ArgumentParser(description="Translate text using a trained NMT model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint (.pt file).")
    parser.add_argument("--text", type=str,
                        help="Text to translate (single sentence).")
    parser.add_argument("--input_file", type=str,
                        help="Path to an input file containing sentences to translate (one sentence per line).")
    parser.add_argument("--output_file", type=str,
                        help="Path to save the translated sentences (if using --input_file).")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for decoding.")
    parser.add_argument("--max_decode_len_ratio", type=float, default=1.5,
                        help="Max generated length = source_len * ratio.")
    args = parser.parse_args()

    device = get_device()
    translator = Translator(args.model_path, device)

    if args.text:
        translated_text = translator.translate_sentence(args.text, args.beam_size, args.max_decode_len_ratio)
        logger.info(f"Input: {args.text}")
        logger.info(f"Translation: {translated_text}")
    elif args.input_file:
        if not os.path.exists(args.input_file):
            logger.error(f"Input file not found: {args.input_file}")
            return
        
        output_lines = []
        with open(args.input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
            for line in tqdm(lines, desc="Translating file"):
                translated_text = translator.translate_sentence(line.strip(), args.beam_size, args.max_decode_len_ratio)
                output_lines.append(translated_text)
                
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as outfile:
                for t_line in output_lines:
                    outfile.write(t_line + '\n')
            logger.info(f"Translations saved to {args.output_file}")
        else:
            for i, t_line in enumerate(output_lines):
                logger.info(f"Original[{i+1}]: {lines[i].strip()}")
                logger.info(f"Translated[{i+1}]: {t_line}")
    else:
        logger.warning("Please provide either --text or --input_file for translation.")

if __name__ == "__main__":
    main()
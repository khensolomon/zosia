# zo/sia/suggest.py
#
# What it does:
# Provides monolingual autocomplete. It auto-detects the language of the
# prefix if the --lang argument is omitted.
#
# How to use it:
# For automatic detection:
#   python -m zo.sia.suggest --text "how are"
#
# To manually specify language:
#   python -m zo.sia.suggest --lang en --text "how are"

import torch
import argparse
import os
import heapq
import math

# --- Import Our Custom Modules ---
from zo.sia.config import load_config
from zo.sia.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from zo.sia.data_utils import normalize_string, tensor_from_sentence, Lang
from zo.sia.detector import LanguageDetector

class Suggester:
    def __init__(self, checkpoint_path, device, lang_code):
        self.device = device
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
            
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        self.config = self.checkpoint['config']
        self.max_length = self.checkpoint['max_length']
        self.lang = self.checkpoint['output_lang']
        
        if self.config.model.get('attention', False):
            self.decoder = AttnDecoderRNN(self.config.model.decoder.hidden_size, self.lang.n_words, max_length=self.max_length).to(device)
        else:
            self.decoder = DecoderRNN(self.config.model.decoder.hidden_size, self.lang.n_words).to(device)

        self.decoder.load_state_dict(self.checkpoint['decoder_state_dict'])
        self.decoder.eval()
        print(f"Model for '{lang_code}' suggestions loaded successfully.")

    def suggest(self, prefix_sentence, beam_width=3):
        with torch.no_grad():
            eos_token_idx = self.lang.word2index[self.lang.special_tokens.eos]
            unk_token_idx = self.lang.word2index[self.lang.special_tokens.unk]
            normalized_prefix = normalize_string(prefix_sentence)
            prefix_words = normalized_prefix.split(' ') if normalized_prefix else []
            decoder_hidden = self.decoder.initHidden(self.device)
            if prefix_words:
                for word in prefix_words:
                    word_idx = self.lang.word2index.get(word, unk_token_idx)
                    decoder_input = torch.tensor([[word_idx]], device=self.device)
                    _, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, None)
                last_word_idx = self.lang.word2index.get(prefix_words[-1], unk_token_idx)
                start_node = (0.0, [last_word_idx], decoder_hidden)
            else:
                sos_token_idx = self.lang.word2index[self.lang.special_tokens.sos]
                start_node = (0.0, [sos_token_idx], decoder_hidden)
            beam = [start_node]
            completed_hypotheses = []
            for _ in range(self.max_length):
                next_beam = []
                for score, seq, hidden in beam:
                    if seq[-1] == eos_token_idx or len(seq) >= self.max_length:
                        completed_hypotheses.append((score, seq))
                        continue
                    decoder_input = torch.tensor([[seq[-1]]], device=self.device)
                    decoder_output, new_hidden, _ = self.decoder(decoder_input, hidden, None)
                    topv, topi = decoder_output.topk(beam_width)
                    for i in range(beam_width):
                        next_word_idx = topi[0][i].item()
                        new_score = score + topv[0][i].item()
                        new_seq = seq + [next_word_idx]
                        heapq.heappush(next_beam, (-new_score, new_seq, new_hidden))
                if not next_beam: break
                beam = [heapq.heappop(next_beam) for _ in range(min(len(next_beam), beam_width))]
                beam = [(-s, q, h) for s, q, h in beam]
            for score, seq, _ in beam:
                completed_hypotheses.append((score, seq))
            completed_hypotheses.sort(key=lambda x: x[0] / len(x[1]) if len(x[1]) > 1 else x[0], reverse=True)
            suggestions = []
            for score, seq in completed_hypotheses[:beam_width]:
                generated_words = [self.lang.index2word[idx] for idx in seq[1:] if idx != eos_token_idx]
                sentence = ' '.join(generated_words)
                num_tokens = len(seq[1:])
                avg_log_prob = score / num_tokens if num_tokens > 0 else 0
                certainty = math.exp(avg_log_prob)
                if sentence:
                    suggestions.append((sentence, certainty))
            return suggestions

def main():
    parser = argparse.ArgumentParser(description="Generate monolingual sentence suggestions.")
    parser.add_argument('--lang', type=str, choices=['zo', 'en'], help="The language for the suggestions (optional, will be auto-detected).")
    parser.add_argument('--text', type=str, required=True, help="The prefix text to generate suggestions for.")
    parser.add_argument('--beam_width', type=int, default=3, help="Number of suggestions to generate.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    lang = args.lang
    if not lang:
        print("Language not specified, attempting to auto-detect...")
        detector = LanguageDetector()
        lang = detector.detect(args.text)
        print(f"Detected language: {lang}")

    if lang == 'unknown':
        print("Error: Could not determine the language of the prefix.")
        return

    src_lang = 'en' if lang == 'zo' else 'zo'
    tgt_lang = lang

    config = load_config()
    filename = config.training.checkpoint.filename_template.format(src=src_lang, tgt=tgt_lang)
    checkpoint_path = os.path.join(config.paths.experiments, filename)

    try:
        device = torch.device(args.device)
        suggester = Suggester(checkpoint_path, device, lang)
        
        print(f"\nGenerating {args.beam_width} suggestions for: '{args.text}'")
        suggestions = suggester.suggest(args.text, beam_width=args.beam_width)
        
        for i, (suggestion, certainty) in enumerate(suggestions):
            # FIX: Format certainty to 2 decimal places
            print(f"{i+1}: {suggestion} (Certainty: {certainty:.2f})")

    except FileNotFoundError:
        print(f"Error: A model for generating '{lang}' suggestions was not found.")
        print(f"Looked for: {checkpoint_path}")
        print(f"Please train this model first using: python -m zo.sia.main --source {src_lang} --target {tgt_lang}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

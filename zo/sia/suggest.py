# zo/sia/suggest.py
#
# What it does:
# Provides monolingual autocomplete. The certainty calculation has been
# significantly improved to be based only on the generated part of the
# suggestion, making it more accurate and reliable for auto-detection.
#
# How to use it:
# The command is the same, but the auto-detection and certainty scores
# for short prefixes will now work correctly.
#
#   python -m zo.sia.suggest --text "hong"

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
            
        is_main_run = __name__ == '__main__'
        if is_main_run: print(f"Loading checkpoint from {checkpoint_path}...")
        
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
        if is_main_run: print(f"Model for '{lang_code}' suggestions loaded successfully.")

    def suggest(self, prefix_sentence, beam_width=3):
        with torch.no_grad():
            unk_token_idx = self.lang.word2index[self.lang.special_tokens.unk]
            sos_token_idx = self.lang.word2index[self.lang.special_tokens.sos]
            eos_token_idx = self.lang.word2index[self.lang.special_tokens.eos]

            normalized_prefix = normalize_string(prefix_sentence)
            prefix_indices = [self.lang.word2index.get(word, unk_token_idx) for word in normalized_prefix.split(' ') if word]

            decoder_hidden = self.decoder.initHidden(self.device)
            
            # --- Priming Step: Calculate the score of the prefix ---
            prefix_score = 0.0
            current_seq = [sos_token_idx]
            if prefix_indices:
                for prefix_idx in prefix_indices:
                    decoder_input = torch.tensor([[current_seq[-1]]], device=self.device)
                    decoder_output, new_hidden, _ = self.decoder(decoder_input, decoder_hidden, None)
                    log_prob = decoder_output[0, prefix_idx].item()
                    prefix_score += log_prob
                    current_seq.append(prefix_idx)
                    decoder_hidden = new_hidden
            
            # The beam starts with the fully scored prefix
            beam = [(prefix_score, current_seq, decoder_hidden)]

            # --- Beam Search for Completion ---
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
            for total_score, seq in completed_hypotheses[:beam_width]:
                # Isolate the generated part of the sequence
                start_index = 1 + len(prefix_indices)
                generated_indices = seq[start_index:]
                
                # FIX: Correctly calculate the score and certainty of the suggestion only
                suggestion_score = total_score - prefix_score
                num_suggestion_tokens = len(generated_indices)
                
                if num_suggestion_tokens > 0:
                    avg_log_prob = suggestion_score / num_suggestion_tokens
                    certainty = math.exp(avg_log_prob)
                else:
                    certainty = 0.0

                generated_words = [self.lang.index2word[idx] for idx in generated_indices if idx != eos_token_idx]
                sentence = ' '.join(generated_words)
                
                if sentence:
                    suggestions.append((sentence, certainty))
            
            return suggestions

def get_suggestions_for_lang(lang, text, beam_width, config, device):
    """Helper function to load a model and get suggestions for a specific language."""
    supported_langs = config.supported_languages
    src_lang = [l for l in supported_langs if l != lang][0]
    tgt_lang = lang
    filename = config.training.checkpoint.filename_template.format(src=src_lang, tgt=tgt_lang)
    checkpoint_path = os.path.join(config.paths.experiments, filename)
    try:
        suggester = Suggester(checkpoint_path, device, lang)
        suggestions = suggester.suggest(text, beam_width=beam_width)
        return suggestions
    except FileNotFoundError:
        return []

def main():
    config = load_config()
    supported_langs = config.supported_languages

    parser = argparse.ArgumentParser(description="Generate monolingual sentence suggestions.")
    parser.add_argument('--lang', type=str, choices=supported_langs, help="The language for the suggestions (optional, will be auto-detected).")
    parser.add_argument('--text', type=str, required=True, help="The prefix text to generate suggestions for.")
    parser.add_argument('--beam_width', type=int, default=3, help="Number of suggestions to generate.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.lang:
        suggestions = get_suggestions_for_lang(args.lang, args.text, args.beam_width, config, device)
        detected_lang = args.lang
    else:
        print("Language not specified, attempting to auto-detect by model confidence...")
        all_suggestions = {}
        for lang_option in supported_langs:
            all_suggestions[lang_option] = get_suggestions_for_lang(lang_option, args.text, args.beam_width, config, device)

        best_lang, highest_certainty = None, -1.0
        for lang, suggestions_list in all_suggestions.items():
            if suggestions_list:
                top_suggestion_certainty = suggestions_list[0][1]
                if top_suggestion_certainty > highest_certainty:
                    highest_certainty = top_suggestion_certainty
                    best_lang = lang
        
        if best_lang:
            print(f"Detected language based on model confidence: {best_lang}")
            suggestions = all_suggestions[best_lang]
            detected_lang = best_lang
        else:
            print("Error: Could not generate suggestions. Ensure models are trained for all directions.")
            return

    if detected_lang == 'unknown':
        print("Error: Could not determine the language of the prefix.")
        return

    print(f"\nGenerating suggestions for: '{args.text}'")
    if suggestions:
        for i, (suggestion, certainty) in enumerate(suggestions):
            print(f"{i+1}: {suggestion} (Certainty: {certainty:.2f})")
    else:
        print("No suggestions found.")

if __name__ == '__main__':
    main()

# zo/sia/translate.py
#
# What it does:
# Translates sentences by automatically detecting the input language. The user
# can also manually specify the source and target languages to override the
# detector.
# This script has been updated to read the list of supported languages
# from the config file, making it fully extensible.
#
# How to use it:
# For automatic detection:
#   python -m zo.sia.translate --text "na dam hiam"
#
# To manually specify direction:
#   python -m zo.sia.translate --source en --target zo --text "hello"

import torch
import argparse
import os
import math

from zo.sia.config import load_config
from zo.sia.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from zo.sia.data_utils import normalize_string, tensor_from_sentence, Lang
from zo.sia.detector import LanguageDetector

# ... (Translator class is unchanged) ...
class Translator:
    def __init__(self, checkpoint_path, device):
        self.device = device
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config, self.input_lang, self.output_lang, self.max_length = [self.checkpoint.get(k) for k in ['config', 'input_lang', 'output_lang', 'max_length']]
        self.encoder = EncoderRNN(self.input_lang.n_words, self.config.model.encoder.hidden_size).to(device)
        self.decoder = AttnDecoderRNN(self.config.model.decoder.hidden_size, self.output_lang.n_words, max_length=self.max_length).to(device) if self.config.model.get('attention', False) else DecoderRNN(self.config.model.decoder.hidden_size, self.output_lang.n_words).to(device)
        self.encoder.load_state_dict(self.checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(self.checkpoint['decoder_state_dict'])
        self.encoder.eval()
        self.decoder.eval()
        print("Model loaded successfully.")
    def translate(self, sentence):
        with torch.no_grad():
            sos_token_idx, eos_token_idx = [self.input_lang.word2index[self.input_lang.special_tokens[k]] for k in ['sos', 'eos']]
            input_tensor = tensor_from_sentence(self.input_lang, normalize_string(sentence), self.max_length, self.device)
            encoder_hidden = self.encoder.initHidden(self.device)
            encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
            decoder_input = torch.tensor([[sos_token_idx]], device=self.device)
            decoder_hidden, decoded_words, total_score = encoder_hidden, [], 0.0
            for _ in range(self.max_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                total_score += topv.item()
                if topi.item() == eos_token_idx: break
                decoded_words.append(self.output_lang.index2word[topi.item()])
                decoder_input = topi.detach()
            certainty = math.exp(total_score / (len(decoded_words) + 1)) if decoded_words else 0.0
            return ' '.join(decoded_words), certainty

def main():
    config = load_config()
    supported_langs = config.get('supported_languages', ['en', 'zo'])

    parser = argparse.ArgumentParser(description="Translate text using a trained ZoSia model.")
    parser.add_argument('--source', type=str, choices=supported_langs, help="Source language (optional).")
    parser.add_argument('--target', type=str, choices=supported_langs, help="Target language (optional).")
    parser.add_argument('--text', type=str, help="Text to translate. If omitted, starts interactive mode.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use ('cpu' or 'cuda').")
    args = parser.parse_args()

    detector = LanguageDetector(supported_languages=supported_langs)
    device = torch.device(args.device)

    # ... (rest of main function is largely unchanged, but now uses supported_langs) ...
    if args.text:
        source_lang = args.source or detector.detect(args.text)
        if not args.source: print(f"Detected language: {source_lang}")
        
        target_lang = args.target
        if not target_lang:
            other_langs = [lang for lang in supported_langs if lang != source_lang]
            if other_langs: target_lang = other_langs[0]

        if source_lang == 'unknown' or not target_lang or source_lang == target_lang:
            print("Error: Could not determine valid translation direction.")
            return

        filename = config.training.checkpoint.filename_template.format(src=source_lang, tgt=target_lang)
        checkpoint_path = os.path.join(config.paths.experiments, filename)
        
        try:
            translator = Translator(checkpoint_path, device)
            translation, certainty = translator.translate(args.text)
            print(f"> {translation} (Certainty: {certainty:.2f})")
        except FileNotFoundError:
            print(f"Error: Model for '{source_lang} -> {target_lang}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        # Interactive mode logic
        print("\n--- Starting Interactive Translation Session ---")
        translators = {}
        while True:
            try:
                input_text = input("> ")
                if input_text.lower() in ['quit', 'exit']: break
                if not input_text: continue
                
                source_lang = detector.detect(input_text)
                print(f"Detected: {source_lang}")
                if source_lang == 'unknown':
                    print("= Could not detect language.")
                    continue
                
                other_langs = [lang for lang in supported_langs if lang != source_lang]
                if not other_langs: continue
                target_lang = other_langs[0]

                direction = f"{source_lang}-{target_lang}"
                if direction not in translators:
                    filename = config.training.checkpoint.filename_template.format(src=source_lang, tgt=target_lang)
                    checkpoint_path = os.path.join(config.paths.experiments, filename)
                    try:
                        translators[direction] = Translator(checkpoint_path, device)
                    except FileNotFoundError:
                        print(f"= No model found for {direction} translation.")
                        continue
                
                translation, certainty = translators[direction].translate(input_text)
                print(f"= {translation} (Certainty: {certainty:.2f})")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

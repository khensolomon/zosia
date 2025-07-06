# zo/sia/translate.py
#
# What it does:
# Translates sentences by automatically detecting the input language. The user
# can also manually specify the source and target languages to override the
# detector.
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

# --- Import Our Custom Modules ---
from zo.sia.config import load_config
from zo.sia.model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from zo.sia.data_utils import normalize_string, tensor_from_sentence, Lang
from zo.sia.detector import LanguageDetector

class Translator:
    def __init__(self, checkpoint_path, device):
        self.device = device
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
            
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        self.config = self.checkpoint['config']
        self.input_lang = self.checkpoint['input_lang']
        self.output_lang = self.checkpoint['output_lang']
        self.max_length = self.checkpoint['max_length']

        self.encoder = EncoderRNN(self.input_lang.n_words, self.config.model.encoder.hidden_size).to(device)

        if self.config.model.get('attention', False):
            self.decoder = AttnDecoderRNN(self.config.model.decoder.hidden_size, self.output_lang.n_words, max_length=self.max_length).to(device)
        else:
            self.decoder = DecoderRNN(self.config.model.decoder.hidden_size, self.output_lang.n_words).to(device)

        self.encoder.load_state_dict(self.checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(self.checkpoint['decoder_state_dict'])

        self.encoder.eval()
        self.decoder.eval()
        print("Model loaded successfully.")

    def translate(self, sentence):
        with torch.no_grad():
            sos_token_idx = self.input_lang.word2index[self.input_lang.special_tokens.sos]
            eos_token_idx = self.input_lang.word2index[self.input_lang.special_tokens.eos]

            normalized_sentence = normalize_string(sentence)
            input_tensor = tensor_from_sentence(self.input_lang, normalized_sentence, self.max_length, self.device)
            
            encoder_hidden = self.encoder.initHidden(self.device)
            encoder_outputs, encoder_hidden = self.encoder(input_tensor, encoder_hidden)

            decoder_input = torch.tensor([[sos_token_idx]], device=self.device)
            decoder_hidden = encoder_hidden
            decoded_words = []
            total_score = 0.0

            for _ in range(self.max_length):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                total_score += topv.item()
                if topi.item() == eos_token_idx:
                    break
                decoded_words.append(self.output_lang.index2word[topi.item()])
                decoder_input = topi.detach()
            
            num_tokens = len(decoded_words) + 1
            avg_log_prob = total_score / num_tokens if num_tokens > 0 else 0
            certainty = math.exp(avg_log_prob)

            return ' '.join(decoded_words), certainty

def main():
    parser = argparse.ArgumentParser(description="Translate text using a trained ZoSia model.")
    parser.add_argument('--source', type=str, choices=['zo', 'en'], help="Source language (optional, will be auto-detected if omitted).")
    parser.add_argument('--target', type=str, choices=['zo', 'en'], help="Target language (optional, will be inferred if source is detected).")
    parser.add_argument('--text', type=str, help="A single text string to translate. If not provided, starts an interactive session.")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for translation ('cpu' or 'cuda').")
    args = parser.parse_args()

    config = load_config()
    detector = LanguageDetector()
    device = torch.device(args.device)

    if args.text:
        source_lang = args.source
        target_lang = args.target

        if not source_lang:
            print("Source language not specified, attempting to auto-detect...")
            source_lang = detector.detect(args.text)
            print(f"Detected language: {source_lang}")

        if not target_lang:
            target_lang = 'zo' if source_lang == 'en' else 'en'
        
        if source_lang == 'unknown' or not target_lang:
            print("Error: Could not determine translation direction. Please specify --source and --target.")
            return

        filename = config.training.checkpoint.filename_template.format(src=source_lang, tgt=target_lang)
        checkpoint_path = os.path.join(config.paths.experiments, filename)
        
        try:
            translator = Translator(checkpoint_path, device)
            translation, certainty = translator.translate(args.text)
            # FIX: Format certainty to 2 decimal places
            print(f"> {translation} (Certainty: {certainty:.2f})")
        except FileNotFoundError:
            print(f"Error: A model for the direction '{source_lang} -> {target_lang}' was not found.")
            print(f"Please train it first using: python -m zo.sia.main --source {source_lang} --target {target_lang}")
        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        print("\n--- Starting Interactive Translation Session ---")
        print("Type a sentence and press Enter. The script will auto-detect the language.")
        print("To override, type your sentence like this: en:hello world")
        print("Type 'quit' or 'exit' to end.")
        
        translators = {}

        while True:
            try:
                input_text = input("> ")
                if input_text.lower() in ['quit', 'exit']:
                    break
                if not input_text:
                    continue

                if ":" in input_text and input_text.split(':', 1)[0] in ['en', 'zo']:
                    source_lang, sentence = input_text.split(':', 1)
                    sentence = sentence.strip()
                else:
                    sentence = input_text
                    source_lang = detector.detect(sentence)
                
                print(f"Detected: {source_lang}")
                
                if source_lang == 'unknown':
                    print("= Could not confidently detect the language.")
                    continue

                target_lang = 'zo' if source_lang == 'en' else 'en'
                direction = f"{source_lang}-{target_lang}"
                
                if direction not in translators:
                    filename = config.training.checkpoint.filename_template.format(src=source_lang, tgt=target_lang)
                    checkpoint_path = os.path.join(config.paths.experiments, filename)
                    try:
                        translators[direction] = Translator(checkpoint_path, device)
                    except FileNotFoundError:
                        print(f"= No model found for {source_lang} -> {target_lang} translation.")
                        continue
                
                translator = translators[direction]
                translation, certainty = translator.translate(sentence)
                # FIX: Format certainty to 2 decimal places
                print(f"= {translation} (Certainty: {certainty:.2f})")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

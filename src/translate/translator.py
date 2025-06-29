# -----------------------------------------------------------------------------
# File: src/translate/translator.py
#
# Description:
#   This script translates a sentence using a trained model. It now accepts a
#   --config-dir argument to make it fully testable.
# -----------------------------------------------------------------------------

import yaml
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import re

# --- Local Imports ---
from src.models.transformer import build_transformer
from tokenizers import Tokenizer

def load_config(config_dir: Path):
    """
    Loads all .yaml files from a given config directory, merges them,
    and robustly resolves all nested ${group.key} placeholders.
    """
    config = {}
    for config_file in sorted(config_dir.glob('*.yaml')):
        with open(config_file, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content:
                for key, value in content.items():
                    if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value

    config_str = yaml.dump(config)
    for _ in range(5):
        placeholders = set(re.findall(r'\$\{(.*?)\}', config_str))
        if not placeholders: break
        for p_str in placeholders:
            if p_str == 'paths.root':
                root_path_val = str(Path(config['paths']['root']).resolve())
                config_str = config_str.replace(f'${{{p_str}}}', root_path_val)
                continue
            try:
                lookup_config = yaml.safe_load(config_str)
                group, key = p_str.split('.')
                value = lookup_config.get(group, {}).get(key)
                if isinstance(value, str) and not re.search(r'\$\{(.*?)\}', value):
                    config_str = config_str.replace(f'${{{p_str}}}', value)
            except (ValueError, KeyError): continue
        config = yaml.safe_load(config_str)
    
    return config

def _generate_subsequent_mask(size, device):
    """Generates a square boolean mask for the target sequence."""
    mask = torch.tril(torch.ones(size, size, device=device)).bool()
    return mask

def beam_search_decode(model, src_tensor, src_mask, tgt_tokenizer, device, max_len, beam_size, alpha):
    """Performs beam search decoding with length penalty."""
    sos_id = tgt_tokenizer.token_to_id("[SOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")

    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)
        
        hypotheses = [(torch.tensor([[sos_id]], device=device), 0.0)]
        completed_hypotheses = []

        for _ in range(max_len):
            if not hypotheses or len(completed_hypotheses) >= beam_size:
                break

            all_candidates = []
            
            for seq, score in hypotheses:
                tgt_mask = _generate_subsequent_mask(seq.size(1), device).unsqueeze(0).unsqueeze(0)
                out = model.decode(encoder_output, src_mask, seq, tgt_mask)
                log_probs = F.log_softmax(model.project(out[:, -1]), dim=-1)
                
                top_scores, top_ids = torch.topk(log_probs, beam_size, dim=1)

                for i in range(beam_size):
                    next_id = top_ids[0, i].item()
                    new_score = score + top_scores[0, i].item()
                    new_seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
                    all_candidates.append((new_seq, new_score))
            
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            hypotheses = []

            for seq, score in ordered:
                if seq[0, -1].item() == eos_id:
                    lp = ((5 + seq.size(1)) ** alpha) / ((5 + 1) ** alpha)
                    completed_hypotheses.append((seq, score / lp))
                else:
                    hypotheses.append((seq, score))
                
                if len(hypotheses) == beam_size:
                    break
        
        if not completed_hypotheses:
            completed_hypotheses.extend(hypotheses)
            
    if not completed_hypotheses:
        return torch.tensor([[]], device=device, dtype=torch.long)
        
    best_hypothesis = sorted(completed_hypotheses, key=lambda x: x[1], reverse=True)[0]
    return best_hypothesis[0]

def main():
    """Main function to handle argument parsing and translation."""
    parser = argparse.ArgumentParser(description="ZoSia Translator")
    parser.add_argument('--config-dir', type=str, default='./config', help="Path to the configuration directory.")
    parser.add_argument('--model_file', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--text', type=str, required=True, help="The text sentence to translate.")
    parser.add_argument('--src_lang', type=str, required=True, help="Source language code")
    parser.add_argument('--tgt_lang', type=str, required=True, help="Target language code")
    parser.add_argument('--beam_size', type=int, default=1, help="Beam size for decoding. 1 for greedy.")
    parser.add_argument('--alpha', type=float, default=0.6, help="Length penalty alpha. 0 for no penalty.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(Path(args.config_dir))
    model_cfg = cfg['model']

    try:
        tokenizer_path = Path(cfg['data_paths']['tokenizers'])
        src_tokenizer = Tokenizer.from_file(str(tokenizer_path / cfg['tokenizer']['tokenizer_file'].format(lang=args.src_lang)))
        tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / cfg['tokenizer']['tokenizer_file'].format(lang=args.tgt_lang)))
    except Exception as e:
        print(f"[ERROR] Error loading tokenizers: {e}")
        return

    src_pad_id = src_tokenizer.token_to_id("[PAD]")
    tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")

    model = build_transformer(
        src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(),
        src_pad_id, tgt_pad_id,
        model_cfg['d_model'], model_cfg['num_encoder_layers'], model_cfg['num_decoder_layers'],
        model_cfg['num_heads'], model_cfg['d_ff'], model_cfg['dropout'], model_cfg['max_seq_len']
    ).to(device)

    checkpoint = torch.load(args.model_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("[OK] Model checkpoint loaded successfully.")

    src_encoded = src_tokenizer.encode(args.text)
    src_tensor = torch.tensor(src_encoded.ids, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = (src_tensor != src_pad_id).unsqueeze(1).unsqueeze(2).to(device)

    output_seq = beam_search_decode(
        model, src_tensor, src_mask,
        tgt_tokenizer, device,
        model_cfg['max_seq_len'], args.beam_size, args.alpha
    )

    translated_text = tgt_tokenizer.decode(output_seq.squeeze(0).tolist(), skip_special_tokens=True).strip()
    
    print("---------------------------------------------")
    if translated_text:
        print(f"Translation: {translated_text}")
    else:
        print("Translation: (Empty)")

if __name__ == '__main__':
    main()

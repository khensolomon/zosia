import torch
import sentencepiece as spm
import os
import argparse
from src.models.transformer_model import Transformer, Encoder, Decoder # Assuming these are defined in transformer_model.py
from src.utils.general_utils import load_config, get_device, load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

def translate_sentence(sentence: str,
                       model: Transformer,
                       sp_model: spm.SentencePieceProcessor,
                       device: torch.device,
                       max_output_len: int = 50):
    """
    Translates a single source sentence using the trained Transformer model
    with greedy decoding.
    """
    model.eval() # Set model to evaluation mode

    # 1. Tokenize the input sentence
    # sp_model.encode_as_ids() returns a list of integer IDs
    src_tokens_ids = sp_model.encode_as_ids(sentence)

    # 2. Add Start-of-Sentence (SOS) and End-of-Sentence (EOS) tokens
    # Ensure sp_model.bos_id() and sp_model.eos_id() are correctly mapped in your data pipeline
    src_tokens_ids = [sp_model.bos_id()] + src_tokens_ids + [sp_model.eos_id()]

    # 3. Convert to PyTorch tensor and add batch dimension
    src_tensor = torch.LongTensor(src_tokens_ids).unsqueeze(0).to(device)

    # 4. Create source mask (required by the Transformer)
    src_mask = model.make_src_mask(src_tensor)

    # 5. Encode the source sentence
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 6. Initialize the target sequence with SOS token
    trg_indexes = [sp_model.bos_id()]
    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device) # Shape: [1, 1]

    # 7. Decode token by token using greedy search
    for i in range(max_output_len):
        trg_mask = model.make_trg_mask(trg_tensor) # Mask for current target sequence

        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # Get the next predicted token (argmax over vocabulary dimension)
        pred_token_id = output.argmax(2)[:, -1].item()

        # Append the predicted token to the target sequence
        trg_indexes.append(pred_token_id)
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # Check if EOS token is predicted
        if pred_token_id == sp_model.eos_id():
            break

    # 8. Decode the sequence of token IDs back to a human-readable string
    # Remove SOS, EOS, and PAD tokens for clean output
    translated_tokens_ids = [idx for idx in trg_indexes if idx not in [sp_model.bos_id(), sp_model.eos_id(), sp_model.pad_id()]]
    translated_sentence = sp_model.decode(translated_tokens_ids)

    return translated_sentence

def main():
    parser = argparse.ArgumentParser(description="Translate a sentence using a trained Transformer NMT model.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint (.pt file) to load.")
    parser.add_argument("--sentence", type=str, required=True,
                        help="The sentence to translate.")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = load_checkpoint(args.checkpoint_path, device)

    # Reconstruct model and tokenizer based on saved configs
    model_config = checkpoint['model_config']
    data_config = checkpoint['data_config']
    sp_model_path = checkpoint['sp_model_path']

    # Load SentencePiece tokenizer
    sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
    src_pad_idx = sp_model.pad_id()
    trg_pad_idx = sp_model.pad_id()
    trg_output_dim = sp_model.get_piece_size()

    # Initialize model with correct parameters
    enc = Encoder(input_dim=sp_model.get_piece_size(),
                  hid_dim=model_config["hidden_dim"],
                  n_layers=model_config["enc_layers"],
                  n_heads=model_config["enc_heads"],
                  pf_dim=model_config["enc_pf_dim"],
                  dropout=model_config["enc_dropout"],
                  max_seq_len=model_config["max_seq_len"],
                  device=device)

    dec = Decoder(output_dim=trg_output_dim,
                  hid_dim=model_config["hidden_dim"],
                  n_layers=model_config["dec_layers"],
                  n_heads=model_config["dec_heads"],
                  pf_dim=model_config["dec_pf_dim"],
                  dropout=model_config["dec_dropout"],
                  max_seq_len=model_config["max_seq_len"],
                  device=device)

    model = Transformer(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)

    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model state loaded.")

    # Translate the sentence
    input_sentence = args.sentence
    logger.info(f"Input Sentence: \"{input_sentence}\"")

    translated_sentence = translate_sentence(input_sentence, model, sp_model, device)

    logger.info(f"Translated Sentence: \"{translated_sentence}\"")
    logger.info("Translation complete.")

if __name__ == "__main__":
    main()
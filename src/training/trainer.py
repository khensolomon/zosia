# The main training loop.

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import math
import argparse
import yaml
import sentencepiece as spm
from tqdm import tqdm
# CHANGED: Updated import for autocast and GradScaler to the newer `torch.amp` path
from torch.amp import autocast, GradScaler 

from src.models.transformer_model import Transformer, Encoder, Decoder
from src.data.dataset_utils import get_dataloaders
from src.training.evaluator import calculate_bleu
from src.utils.general_utils import load_config, setup_logging, get_device, save_checkpoint, load_checkpoint
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Learning Rate Schedulers ---
class NoamOpt:
    "Optim wrapper that implements learning rate scheduling with warmup."
    def __init__(self, model_size, warmup_steps, optimizer):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self._step = 0
        self._rate = 0

    def step(self):
        "Update parameters and lr"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.model_size**(-0.5) * \
               min(step**(-0.5), step * self.warmup_steps**(-1.5))
    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains optimizer state and its own state (_step, warmup_steps, model_size).
        """
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            '_step': self._step,
            'warmup_steps': self.warmup_steps,
            'model_size': self.model_size,
            '_rate': self._rate # Good to save current rate too
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler state.
        """
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self._step = state_dict['_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.model_size = state_dict['model_size']
        self._rate = state_dict['_rate']

def train_epoch(model: Transformer,
                dataloader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.CrossEntropyLoss,
                clip: float,
                device: torch.device,
                scaler: GradScaler = None, # For mixed precision
                accumulate_grad_batches: int = 1):
    """
    Performs one epoch of training.
    """
    model.train()
    epoch_loss = 0
    pbar = tqdm(dataloader, desc="Training")

    # optimizer.zero_grad() # Zero gradients once per accumulation step
    optimizer.optimizer.zero_grad() # Access the underlying optimizer

    for batch_idx, (src, trg) in enumerate(pbar):
        src, trg = src.to(device), trg.to(device)

        # CHANGED: Added device_type='cuda' as recommended by PyTorch
        with autocast(device_type='cuda', enabled=scaler is not None): 
            output = model(src, trg[:, :-1]) # Trg input is shifted right (remove last token)

            # Output is (batch_size, trg_len-1, output_dim)
            # Target is (batch_size, trg_len)
            output_dim = output.shape[-1]
            # Flatten output and target for criterion
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1) # Target shifted left (remove first token)

            # Ignore padding tokens in loss calculation
            loss = criterion(output, trg)
            loss = loss / accumulate_grad_batches # Scale loss by accumulation factor

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulate_grad_batches == 0:
            if scaler is not None:
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            # optimizer.zero_grad() # Zero gradients after optimization step
            optimizer.optimizer.zero_grad() # Access the underlying optimizer

        epoch_loss += loss.item() * accumulate_grad_batches # Scale loss back for tracking
        pbar.set_postfix(loss=epoch_loss / (batch_idx + 1))

    return epoch_loss / len(dataloader)

def evaluate_epoch(model: Transformer,
                   dataloader: torch.utils.data.DataLoader,
                   criterion: nn.CrossEntropyLoss,
                   device: torch.device,
                   sp_model: spm.SentencePieceProcessor,
                   max_seq_len: int,
                   scaler: GradScaler = None):
    """
    Evaluates the model on the validation/test set.
    """
    model.eval()
    epoch_loss = 0
    all_predicted_tokens = []
    all_target_tokens = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for src, trg in pbar:
            src, trg = src.to(device), trg.to(device)

            # CHANGED: Added device_type='cuda' as recommended by PyTorch
            with autocast(device_type='cuda', enabled=scaler is not None):
                # Teacher forcing for loss calculation
                output = model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                loss = criterion(output.contiguous().view(-1, output_dim), trg[:, 1:].contiguous().view(-1))
            
            epoch_loss += loss.item()

            # For BLEU score, use greedy decoding (or beam search)
            # This is a simplified greedy decoding for evaluation, for robust BLEU use beam search in translator.py
            max_trg_len = trg.shape[1] # Use target length as max decoding length
            batch_size = src.shape[0]

            # Start decoding with SOS token
            translated_tokens = torch.full((batch_size, 1), sp_model.bos_id(), dtype=torch.long, device=device)

            for t in range(max_trg_len -1): # Generate up to max_trg_len - 1 tokens
                src_mask = model.make_src_mask(src)
                trg_mask = model.make_trg_mask(translated_tokens)

                # CHANGED: Added device_type='cuda' as recommended by PyTorch
                with autocast(device_type='cuda', enabled=scaler is not None):
                    output = model.decoder(translated_tokens, model.encoder(src, src_mask), trg_mask, src_mask)

                pred_token = output[:, -1].argmax(1).unsqueeze(1) # Take last token's prediction
                translated_tokens = torch.cat((translated_tokens, pred_token), dim=1)

                # Break if all sequences have generated EOS or reached max_trg_len
                if (pred_token == sp_model.eos_id()).all():
                    break
            
            # Decode token IDs to text for BLEU calculation
            for i in range(batch_size):
                # Remove SOS, EOS, PAD tokens for BLEU calculation
                pred_ids = translated_tokens[i].tolist()
                pred_text = sp_model.decode([id for id in pred_ids if id not in [sp_model.bos_id(), sp_model.eos_id(), sp_model.pad_id()]])
                all_predicted_tokens.append(pred_text)

                trg_ids = trg[i].tolist()
                trg_text = sp_model.decode([id for id in trg_ids if id not in [sp_model.bos_id(), sp_model.eos_id(), sp_model.pad_id()]])
                all_target_tokens.append([trg_text]) # sacrebleu expects list of references

    avg_loss = epoch_loss / len(dataloader)
    bleu_score = calculate_bleu(all_predicted_tokens, all_target_tokens)
    return avg_loss, bleu_score

def main():
    parser = argparse.ArgumentParser(description="Train a Zo-English Transformer NMT model.")
    # Default config paths are already correctly set here
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                        help="Path to the training configuration YAML file. Default: config/training_config.yaml")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml",
                        help="Path to the model configuration YAML file. Default: config/model_config.yaml")
    parser.add_argument("--data_config", type=str, default="config/data_config.yaml",
                        help="Path to the data configuration YAML file. Default: config/data_config.yaml")
    args = parser.parse_args()

    # Load configurations with robust error handling
    training_config = {}
    model_config = {}
    data_config = {}

    try:
        training_config = load_config(args.config)
        logger.info(f"Loaded main training configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Error: Main training configuration file not found at '{args.config}'. Please check the path.")
        return # Exit if a critical config is missing
    except yaml.YAMLError as e:
        logger.error(f"Error parsing main training configuration file '{args.config}': {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading main training config '{args.config}': {e}")
        return

    try:
        model_config = load_config(args.model_config)
        logger.info(f"Loaded model configuration from {args.model_config}")
    except FileNotFoundError:
        logger.error(f"Error: Model configuration file not found at '{args.model_config}'. Please check the path.")
        return # Exit if a critical config is missing
    except yaml.YAMLError as e:
        logger.error(f"Error parsing model configuration file '{args.model_config}': {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading model config '{args.model_config}': {e}")
        return

    try:
        data_config = load_config(args.data_config)
        logger.info(f"Loaded data configuration from {args.data_config}")
    except FileNotFoundError:
        logger.error(f"Error: Data configuration file not found at '{args.data_config}'. Please check the path.")
        return # Exit if a critical config is missing
    except yaml.YAMLError as e:
        logger.error(f"Error parsing data configuration file '{args.data_config}': {e}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading data config '{args.data_config}': {e}")
        return

    # Setup logging and experiment directory
    exp_name_prefix = training_config.get("experiment_name_prefix", "nmt_run")
    # It's safer to get base_output_dir from training_config or a dedicated project config
    # rather than loading another default_config.yaml in the main run.
    # Assuming base_output_dir is in training_config or you have a project-wide config loaded once.
    # For simplicity, let's assume it's in training_config for now.
    base_output_dir = training_config.get("base_output_dir", "experiments") # Get from training_config
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_output_dir, f"{exp_name_prefix}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

    # setup_logging function needs the base experiment directory to set up file handlers
    log_dir_path = os.path.join(experiment_dir, "logs")
    run_log_name = "training"
    setup_logging(log_dir_path, run_log_name) # Ensure this sets up file logging correctly

    logger.info(f"Starting new experiment: {experiment_dir}")
    logger.info(f"Training Config:\n{yaml.dump(training_config, indent=2)}")
    logger.info(f"Model Config:\n{yaml.dump(model_config, indent=2)}")
    logger.info(f"Data Config:\n{yaml.dump(data_config, indent=2)}")

    # Initialize WandB if enabled
    use_wandb = training_config.get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=training_config.get("wandb_project_name", "ZoSia_NMT"),
                       entity=training_config.get("wandb_entity", None),
                       config={**training_config, **model_config, **data_config},
                       name=f"{exp_name_prefix}_{timestamp}",
                       dir=experiment_dir) # Logs will be stored in this directory
            logger.info("Weights & Biases initialized.")
        except ImportError:
            logger.warning("Weights & Biases (wandb) not installed. Please run `pip install wandb` to enable it.")
            use_wandb = False # Disable wandb if not installed
        except Exception as e:
            logger.error(f"Error initializing Weights & Biases: {e}")
            use_wandb = False # Disable wandb on error


    device = get_device()
    logger.info(f"Using device: {device}")

    # Load SentencePiece tokenizer
    vocab_dir = data_config.get("vocab_dir")
    tokenizer_prefix = data_config.get("tokenizer_prefix")
    
    if not vocab_dir or not tokenizer_prefix:
        logger.error("Error: 'vocab_dir' or 'tokenizer_prefix' missing in data_config.yaml.")
        return # Exit if essential paths are missing

    sp_model_path = os.path.join(vocab_dir, f"{tokenizer_prefix}.model")
    if not os.path.exists(sp_model_path):
        logger.error(f"SentencePiece model not found at {sp_model_path}. Please run make_dataset.py first to create it.")
        return

    try:
        sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        logger.info(f"Tokenizer loaded successfully from {sp_model_path}")
        logger.info(f"Tokenizer vocabulary size: {sp_model.get_piece_size()}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer model from {sp_model_path}: {e}")
        return # Exit if tokenizer loading fails

    src_pad_idx = sp_model.pad_id()
    trg_pad_idx = sp_model.pad_id()
    trg_output_dim = sp_model.get_piece_size() # Vocabulary size for target output

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(data_config, training_config, sp_model)
    logger.info("DataLoaders prepared.")

    # Initialize model
    # Ensure model_config has necessary keys, using .get() with defaults for robustness
    input_dim = sp_model.get_piece_size() # Assuming same vocab for src/tgt
    hidden_dim = model_config.get("hidden_dim")
    enc_layers = model_config.get("enc_layers")
    enc_heads = model_config.get("enc_heads")
    enc_pf_dim = model_config.get("enc_pf_dim")
    enc_dropout = model_config.get("enc_dropout")
    dec_layers = model_config.get("dec_layers")
    dec_heads = model_config.get("dec_heads")
    dec_pf_dim = model_config.get("dec_pf_dim")
    dec_dropout = model_config.get("dec_dropout")
    max_seq_len_model = model_config.get("max_seq_len") # Use a different name to avoid confusion with data_config's max_sequence_length

    # Basic check for essential model configs
    if any(x is None for x in [hidden_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout,
                               dec_layers, dec_heads, dec_pf_dim, dec_dropout, max_seq_len_model]):
        logger.error("Error: One or more essential model configuration parameters are missing.")
        return

    enc = Encoder(input_dim=input_dim,
                  hid_dim=hidden_dim,
                  n_layers=enc_layers,
                  n_heads=enc_heads,
                  pf_dim=enc_pf_dim,
                  dropout=enc_dropout,
                  max_seq_len=max_seq_len_model,
                  device=device)

    dec = Decoder(output_dim=trg_output_dim,
                  hid_dim=hidden_dim,
                  n_layers=dec_layers,
                  n_heads=dec_heads,
                  pf_dim=dec_pf_dim,
                  dropout=dec_dropout,
                  max_seq_len=max_seq_len_model,
                  device=device)

    model = Transformer(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)

    # Initialize weights (optional, but can help training stability)
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)
    model.apply(initialize_weights)

    # Commented out for cleaner logs
    logger.info(f"Model initialized: \n{model}")
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    learning_rate = training_config.get("learning_rate")
    warmup_steps = training_config.get("warmup_steps")
    gradient_clip = training_config.get("gradient_clip")
    epochs = training_config.get("epochs")
    accumulate_grad_batches = training_config.get("accumulate_grad_batches", 1) # Default to 1

    if any(x is None for x in [learning_rate, warmup_steps, gradient_clip, epochs]):
        logger.error("Error: One or more essential training configuration parameters (learning_rate, warmup_steps, gradient_clip, epochs) are missing.")
        return

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # Noam scheduler
    scheduler = NoamOpt(hidden_dim, warmup_steps, optimizer) # Use hidden_dim from model_config

    # Loss function: ignore padding index for target
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    # Mixed precision scaler
    scaler = GradScaler() if training_config.get("mixed_precision", False) and device.type == 'cuda' else None
    if scaler:
        logger.info("Using Automatic Mixed Precision (AMP).")

    best_val_bleu = -1.0 # Track best BLEU for saving model
    best_val_loss = float('inf') # Track best loss for tie-breaking or alternative saving

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, scheduler, criterion,
                                 gradient_clip, device,
                                 scaler, accumulate_grad_batches)
        val_loss, val_bleu = evaluate_epoch(model, val_loader, criterion, device, sp_model,
                                            max_seq_len_model, scaler) # Use model_config's max_seq_len for evaluation

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        logger.info(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        logger.info(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}")
        logger.info(f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):.3f} | Val. BLEU: {val_bleu:.2f}")

        if use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "train_ppl": math.exp(train_loss),
                "val_loss": val_loss,
                "val_ppl": math.exp(val_loss),
                "val_bleu": val_bleu,
                "epoch": epoch,
                "learning_rate": scheduler.rate() # Log current LR from scheduler
            })

        # Save checkpoint
        checkpoint_filename = f"epoch_{epoch+1:02d}.pt"
        checkpoint_path = os.path.join(experiment_dir, "checkpoints", checkpoint_filename)
        
        # Determine whether to save as best model (based on BLEU)
        is_best_bleu = val_bleu > best_val_bleu
        if is_best_bleu:
            best_val_bleu = val_bleu
            best_val_loss = val_loss # Update best_val_loss too

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
            'val_loss': val_loss,
            'val_bleu': val_bleu,
            'sp_model_path': sp_model_path,
            'data_config': data_config,
            'model_config': model_config,
            'training_config': training_config,
            'amp_scaler_state_dict': scaler.state_dict() if scaler else None # Save scaler state
        }, checkpoint_path, is_best=is_best_bleu, best_model_filename="best_model.pt") # Pass is_best to save_checkpoint
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        if is_best_bleu:
            logger.info(f"New best model (BLEU: {best_val_bleu:.2f}) saved to {os.path.join(experiment_dir, 'checkpoints', 'best_model.pt')}")


    if use_wandb:
        wandb.finish()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
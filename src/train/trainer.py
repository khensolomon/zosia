# -----------------------------------------------------------------------------
# File: src/train/trainer.py
#
# Description:
#   Main script for training the Transformer model. It has been upgraded to
#   save the best model to both the run-specific experiment folder and the
#   main experiments folder for easy access.
# -----------------------------------------------------------------------------

import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import re
import sys

# --- Local Imports ---
from src.models.transformer import build_transformer
from src.dataset.utils import BilingualDataset, PadCollate
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

# --- Main Training Class ---
class Trainer:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.src_lang = args.src_lang
        self.tgt_lang = args.tgt_lang
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.writer = None
        self.start_epoch = 0
        self.global_step = 0
        self.src_pad_id = 0
        self.tgt_pad_id = 0
        self.src_vocab_size = 0
        self.tgt_vocab_size = 0
        
        self.best_val_loss = float('inf')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        now = datetime.now().strftime("%Y%m%d%H%M%S")
        direction_short = f"{self.src_lang}{self.tgt_lang}"
        self.exp_dir = Path(cfg['training']['experiment_dir']) / f"{now}_{direction_short}"
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.exp_dir)
        print(f"Experiment data will be saved to: {self.exp_dir}")

    def _setup(self):
        self._prepare_dataloaders()
        self._build_model()
        self._build_optimizer()
        self._build_loss_fn()

    def _prepare_dataloaders(self):
        print("Preparing dataloaders...")
        data_cfg = self.cfg['data_paths']
        processed_path = Path(data_cfg['processed'])

        tokenizer_path = Path(data_cfg['tokenizers'])
        src_tokenizer = Tokenizer.from_file(str(tokenizer_path / self.cfg['tokenizer']['tokenizer_file'].format(lang=self.src_lang)))
        tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path / self.cfg['tokenizer']['tokenizer_file'].format(lang=self.tgt_lang)))
        
        self.src_pad_id = src_tokenizer.token_to_id("[PAD]")
        self.tgt_pad_id = tgt_tokenizer.token_to_id("[PAD]")
        self.src_vocab_size = src_tokenizer.get_vocab_size()
        self.tgt_vocab_size = tgt_tokenizer.get_vocab_size()

        train_dataset = BilingualDataset(str(processed_path / f"train.{self.src_lang}"), str(processed_path / f"train.{self.tgt_lang}"), src_tokenizer, tgt_tokenizer)
        val_dataset = BilingualDataset(str(processed_path / f"val.{self.src_lang}"), str(processed_path / f"val.{self.tgt_lang}"), src_tokenizer, tgt_tokenizer)
        
        collate_obj = PadCollate(src_pad_id=self.src_pad_id, tgt_pad_id=self.tgt_pad_id)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.cfg['training']['batch_size'], shuffle=True, collate_fn=collate_obj, num_workers=self.cfg['training']['num_workers'])
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.cfg['training']['batch_size'], shuffle=False, collate_fn=collate_obj, num_workers=self.cfg['training']['num_workers'])
        
        print("[OK] Dataloaders ready.")
        
    def _build_model(self):
        print("Building model...")
        self.model = build_transformer(
            self.src_vocab_size, self.tgt_vocab_size,
            self.src_pad_id, self.tgt_pad_id,
            self.cfg['model']['d_model'], self.cfg['model']['num_encoder_layers'],
            self.cfg['model']['num_decoder_layers'], self.cfg['model']['num_heads'],
            self.cfg['model']['d_ff'], self.cfg['model']['dropout'], self.cfg['model']['max_seq_len']
        ).to(self.device)
        print("[OK] Model built and moved to device.")

    def _build_optimizer(self):
        opt_cfg = self.cfg['training']['optimizer']
        if opt_cfg['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['training']['learning_rate'], betas=tuple(opt_cfg['betas']), eps=opt_cfg['eps'])
        else:
            raise NotImplementedError(f"Optimizer {opt_cfg['name']} not supported.")
        print(f"[OK] Optimizer '{opt_cfg['name']}' configured.")

    def _build_loss_fn(self):
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.cfg['training']['label_smoothing'], ignore_index=self.tgt_pad_id)
        print("[OK] Loss function (CrossEntropyLoss with Label Smoothing) configured.")

    def _run_one_epoch(self, epoch):
        self.model.train()
        batch_iterator = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.cfg['training']['num_epochs']}")
        for batch in batch_iterator:
            self.optimizer.zero_grad(set_to_none=True)
            
            encoder_input = batch['encoder_input'].to(self.device)
            decoder_input = batch['decoder_input'].to(self.device)
            label = batch['label'].to(self.device)

            src_mask = (encoder_input != self.src_pad_id).unsqueeze(1).unsqueeze(2)
            tgt_mask = (decoder_input != self.tgt_pad_id).unsqueeze(1).unsqueeze(2) & self._generate_subsequent_mask(decoder_input.size(1))
            
            encoder_output = self.model.encode(encoder_input, src_mask)
            decoder_output = self.model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
            output = self.model.project(decoder_output)

            loss = self.loss_fn(output.view(-1, self.tgt_vocab_size), label.view(-1))
            loss.backward()
            self.optimizer.step()
            
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
            self.global_step += 1
            
    def _validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        if not self.val_dataloader or len(self.val_dataloader.dataset) == 0:
            print(f"Validation not possible for Epoch {epoch+1}: No validation data.")
            return

        batch_iterator = tqdm(self.val_dataloader, desc=f"Validating Epoch {epoch+1}")
        with torch.no_grad():
            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(self.device)
                decoder_input = batch['decoder_input'].to(self.device)
                label = batch['label'].to(self.device)

                src_mask = (encoder_input != self.src_pad_id).unsqueeze(1).unsqueeze(2)
                tgt_mask = (decoder_input != self.tgt_pad_id).unsqueeze(1).unsqueeze(2) & self._generate_subsequent_mask(decoder_input.size(1))

                encoder_output = self.model.encode(encoder_input, src_mask)
                decoder_output = self.model.decode(encoder_output, src_mask, decoder_input, tgt_mask)
                output = self.model.project(decoder_output)
                
                loss = self.loss_fn(output.view(-1, self.tgt_vocab_size), label.view(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_dataloader)
        print(f"\nValidation Loss for Epoch {epoch+1}: {avg_loss:.4f}")
        self.writer.add_scalar('Loss/validation', avg_loss, epoch)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            print(f"[BEST MODEL] New best model found! Saving checkpoint... (Epoch {epoch+1})")
            self._save_best_checkpoint(epoch)

    def _save_best_checkpoint(self, epoch):
        """
        Saves the best model checkpoint to two locations:
        1. Inside the specific experiment run's directory (for archival).
        2. Directly in the main experiments directory (for easy access).
        """
        filename_template = self.cfg['training']['best_model_filename']
        model_filename = filename_template.format(
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang
        )
        
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'validation_loss': self.best_val_loss
        }

        # Path 1: Inside the timestamped experiment directory
        exp_save_path = self.exp_dir / model_filename
        torch.save(checkpoint_data, exp_save_path)
        print(f"Best model for this run saved to: {exp_save_path}")

        # Path 2: Directly in the main experiments directory
        main_experiments_dir = Path(self.cfg['training']['experiment_dir'])
        main_save_path = main_experiments_dir / model_filename
        torch.save(checkpoint_data, main_save_path)
        print(f"Overall best model for {self.src_lang}-{self.tgt_lang} updated at: {main_save_path}\n")

    def _generate_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size, device=self.device)).bool()
        return mask

    def run(self):
        self._setup()
        for epoch in range(self.start_epoch, self.cfg['training']['num_epochs']):
            self._run_one_epoch(epoch)
            self._validate(epoch)
        
        print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZoSia Model Trainer")
    parser.add_argument('--config-dir', type=str, default='./config', help="Path to the configuration directory.")
    parser.add_argument('--src_lang', type=str, required=True, help="Source language code")
    parser.add_argument('--tgt_lang', type=str, required=True, help="Target language code")
    args = parser.parse_args()

    config = load_config(Path(args.config_dir))
    trainer = Trainer(args, config)
    trainer.run()

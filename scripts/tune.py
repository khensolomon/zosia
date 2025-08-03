"""
Hyperparameter Tuning Script for Zolai-NMT
version: 2025.08.03.044500

This script uses the Optuna library to automatically find the best
hyperparameters for the NMT model on a given dataset.

--- Features ---

- Resumable Studies: The script saves its progress to a database file
  (./experiments/tuning_studies.db). If a study is interrupted, you can
  run the same command again to resume from where it left off.

- Modern CLI: Uses intuitive sub-commands like `run` and `show-best`.

- Effort Presets: A simple `--effort` flag allows you to control the
  duration and thoroughness of the tuning process.

--- CLI Usage ---

1. Run a new tuning study:
   This command will run a new study with a medium level of effort (50 trials)
   and save the best parameters to ./config/hyperparams-{src}-{tgt}.yaml.

   python ./scripts/tune.py run --source zo --target en --effort medium

2. Show the best results from a previous study:
   This command will load the results from the last study for a language
   pair and display the best parameters found.

   python ./scripts/tune.py show-best --source zo --target en
"""
import os
import sys
import argparse
import yaml
import logging
import optuna
from tqdm import tqdm

# Add the script's directory to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import the refactored training logic and other necessary components from nmt.py
from nmt import (
    run_training_for_tuning,
    build_data_index,
    StreamingTranslationDataset,
    Tokenizer,
    Config
)

# Configure Optuna logging to show trial results
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

def objective(trial, args, params):
    """
    The main objective function that Optuna will optimize.
    Each call to this function is a single "trial" with one set of hyperparameters.
    """
    # --- 1. Define the Search Space ---
    trial_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'embedding_size': trial.suggest_categorical('embedding_size', [128, 256]),
        'num_heads': trial.suggest_categorical('num_heads', [4, 8]),
        'ff_hidden_size': trial.suggest_categorical('ff_hidden_size', [256, 512]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        # We keep these constant for the tuning process
        'epochs': params['epochs'],
        'batch_size': params['batch_size'],
        'validation_split': params['validation_split'],
        'patience': params['patience'],
        'vocab_size': params['vocab_size']
    }

    # --- 2. Run the Training and Get the Score ---
    best_validation_loss = run_training_for_tuning(trial_params, args)

    # --- 3. Return the Score for Optuna to Optimize ---
    return best_validation_loss

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Zolai-NMT")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- `run` command ---
    run_parser = subparsers.add_parser('run', help="Run a new tuning study.")
    run_parser.add_argument('--source', type=str, required=True, choices=Config.SUPPORTED_LANGUAGES)
    run_parser.add_argument('--target', type=str, required=True, choices=Config.SUPPORTED_LANGUAGES)
    run_parser.add_argument('--effort', type=str, default='medium', choices=['quick', 'medium', 'thorough'],
                              help="The level of effort for the study.")
    run_parser.add_argument('--trials', type=int, default=None,
                              help="Manually specify the number of trials, overriding --effort.")

    # --- `show-best` command ---
    show_parser = subparsers.add_parser('show-best', help="Show the best results from a previous study.")
    show_parser.add_argument('--source', type=str, required=True, choices=Config.SUPPORTED_LANGUAGES)
    show_parser.add_argument('--target', type=str, required=True, choices=Config.SUPPORTED_LANGUAGES)

    args = parser.parse_args()

    # --- Command Logic ---
    if args.command == 'run':
        effort_map = {'quick': 10, 'medium': 50, 'thorough': 200}
        n_trials = args.trials if args.trials is not None else effort_map[args.effort]

        args.use_tsv_index = True
        args.use_templates = True
        args.include_template_tags = None
        args.exclude_template_tags = None
        args.files = None
        
        with open(Config.DEFAULT_HYPERPARAMS_FILE, 'r') as f:
            base_params = yaml.safe_load(f)

        # --- Persistent Storage Setup ---
        os.makedirs(Config.EXPERIMENTS_DIR, exist_ok=True)
        storage_path = f"sqlite:///{os.path.join(Config.EXPERIMENTS_DIR, 'tuning_studies.db')}"
        
        # Sort language codes to ensure consistent study name
        lang_pair = sorted([args.source, args.target])
        study_name = f"{lang_pair[0]}-{lang_pair[1]}-tuning"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True, # This is the key to resuming
            direction='minimize'
        )
        
        # Adjust n_trials if we are resuming a study
        completed_trials = len(study.trials)
        if completed_trials >= n_trials:
            print(f"Study '{study_name}' already has {completed_trials} trials. Target is {n_trials}.")
            print("To run more trials, increase the --effort or --trials value.")
            return
            
        remaining_trials = n_trials - completed_trials
        print(f"Resuming study '{study_name}'. {completed_trials} trials complete, {remaining_trials} remaining.")

        with tqdm(total=remaining_trials, desc="Tuning Progress") as pbar:
            def callback(study, trial):
                pbar.update(1)

            study.optimize(lambda trial: objective(trial, args, base_params), n_trials=remaining_trials, callbacks=[callback])

        print("\n--- Tuning Complete ---")
        print(f"Best validation loss: {study.best_value}")
        print("Best hyperparameters found:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        output_path = Config.HYPERPARAMS_OUTPUT_PATTERN.format(src=args.source, tgt=args.target)
        best_tuned_params = {k: v for k, v in study.best_params.items() if k in [
            'learning_rate', 'num_layers', 'embedding_size', 'num_heads', 'ff_hidden_size', 'dropout'
        ]}
        
        os.makedirs(Config.CONFIG_DIR, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(best_tuned_params, f, default_flow_style=False)
        
        print(f"\nBest parameters saved to: {output_path}")

    elif args.command == 'show-best':
        path = Config.HYPERPARAMS_OUTPUT_PATTERN.format(src=args.source, tgt=args.target)
        if not os.path.exists(path):
            print(f"Error: No tuned hyperparameter file found at '{path}'.")
            print("Please run a study first using the 'run' command.")
            sys.exit(1)
        
        with open(path, 'r') as f:
            best_params = yaml.safe_load(f)
        
        print(f"--- Best Found Hyperparameters for {args.source}-{args.target} ---")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

if __name__ == '__main__':
    main()

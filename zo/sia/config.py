"""
Zolai-NMT Configuration Module
version: 2025.08.08.1220

Handles loading and merging of configuration from YAML files and .env overrides.
No other script should access configuration files directly. All access must go
through an instance of the `Config` class.
"""
import os
import yaml

class ConfigObject:
    """
    A flexible, dictionary-like object that allows accessing keys as attributes.
    This provides a cleaner syntax (e.g., `cfg.data.paths`) compared to dict
    access (`cfg['data']['paths']`). It recursively converts nested dicts.
    """
    def __init__(self, data):
        # The __dict__ is used to store the attributes of the instance.
        # We directly populate it from the input dictionary.
        for key, value in data.items():
            # Ensure keys are valid Python identifiers before setting as attributes
            if not key.isidentifier():
                print(f"Warning: Config key '{key}' is not a valid identifier and will be skipped for dot notation.")
                continue
            
            if isinstance(value, dict):
                # Recursively convert nested dictionaries into ConfigObjects
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        # Provides a dictionary-like representation for clarity when printing
        return repr(self.__dict__)

class Config:
    """
    A centralized configuration manager that loads settings from multiple YAML
    files and allows for local overrides via a .env file.
    
    Configuration is accessed using dot notation, e.g., `config.data.paths.corpus_dir`.
    """
    def __init__(self, config_dir='./config', env_file='.env'):
        # Load base configurations from YAML files
        app_cfg = self._load_yaml(os.path.join(config_dir, 'app.yaml'))
        data_cfg = self._load_yaml(os.path.join(config_dir, 'data.yaml'))
        model_cfg = self._load_yaml(os.path.join(config_dir, 'model.yaml'))

        # Create a nested configuration structure to preserve namespaces
        self._config = {
            'app': app_cfg,
            'data': data_cfg,
            'model': model_cfg
        }

        # Apply overrides from .env file if it exists
        if os.path.exists(env_file):
            print(f"Applying overrides from '{env_file}'...")
            self._apply_env_overrides(env_file)

        # Convert the nested dictionary to a custom object for dot notation access
        self._structured_config = ConfigObject(self._config)

    def _load_yaml(self, path):
        """Loads a single YAML file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _apply_env_overrides(self, env_file):
        """Parses a .env file and updates the configuration dictionary."""
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    print(f"Warning: Skipping malformed line in .env: {line}")
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'') # Remove quotes

                # Convert value to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass # Keep as string

                self._set_nested_key(self._config, key, value)

    def _set_nested_key(self, d, key_str, value):
        """Sets a value in a nested dictionary using a dot-separated key."""
        keys = key_str.split('.')
        current_level = d
        for key in keys[:-1]:
            current_level = current_level.setdefault(key, {})
        current_level[keys[-1]] = value

    def __getattr__(self, name):
        """Allows direct access to the top-level configuration groups."""
        if hasattr(self._structured_config, name):
            return getattr(self._structured_config, name)
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def get_lang_pair_config(self, source, target):
        """
        In the future, this can be extended to load language-pair specific
        hyperparameter overrides from a dedicated YAML file if needed.
        For now, it returns the base model config.
        """
        # Placeholder for future functionality
        return self.model

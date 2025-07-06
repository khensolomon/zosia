# zo/sia/config.py
#
# What it does:
# This module provides a single function, `load_config`, to handle all project
# configurations. It loads `default.yaml` as the base, then loads every other
# .yaml file from the `/config` directory into its own top-level key.
# Finally, it applies any overrides from the root `.env` file.
#
# Why it's used:
# It centralizes all configuration logic, creating a clean separation between
# the application code and its settings. This allows for easy management of
# different environments and prevents hard-coding values.
#
# How to use it:
# From any other script, import and call the function:
#
#   from zo.sia.config import load_config
#   config = load_config()
#
# The function returns an `addict.Dict` object, which allows for easy,
# attribute-style access to nested configuration values (e.g., `config.data.sources`).
#
# Dependencies: PyYAML, python-dotenv, addict

import os
import yaml
from dotenv import dotenv_values
from addict import Dict
import re

def _deep_merge(source, destination):
    """
    Recursively merges source dict into destination dict.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination

def _resolve_variables(config):
    """
    Resolves variables in the format ${path.to.key} within the config.
    """
    config_str = yaml.dump(config)
    variable_pattern = re.compile(r'\$\{(.*?)\}')
    
    def get_value_from_path(d, path):
        keys = path.split('.')
        for key in keys:
            d = d[key]
        return d

    for match in variable_pattern.finditer(config_str):
        path = match.group(1)
        try:
            resolved_value = get_value_from_path(config, path)
            if not isinstance(resolved_value, str):
                raise TypeError(f"Resolved value for {path} must be a string.")
            config_str = config_str.replace(match.group(0), resolved_value)
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not resolve variable {match.group(0)}. Error: {e}")

    return yaml.safe_load(config_str)

def _apply_env_overrides(config, env_file_path):
    """
    Applies overrides from a .env file to the configuration dictionary.
    """
    if not os.path.exists(env_file_path):
        return config

    overrides = dotenv_values(env_file_path)
    for key, value in overrides.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        
        try:
            original_val = config
            for k_orig in keys:
                original_val = original_val[k_orig]
            if isinstance(original_val, (int, float, bool)):
                 value = type(original_val)(value)
        except (KeyError, ValueError):
            pass

        d[keys[-1]] = value
    return config

def load_config():
    """
    Loads all configurations.
    1. Loads default.yaml as the base.
    2. Loads other .yaml files into keys named after the file.
    3. Applies overrides from the .env file.
    4. Resolves path variables (e.g., ${paths.root}).
    5. Returns the final configuration as an addict.Dict.
    """
    script_dir = os.path.dirname(__file__)
    config_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'config'))
    root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Configuration directory not found at: {config_dir}")

    # 1. Load default.yaml as the base
    merged_config = {}
    default_path = os.path.join(config_dir, 'default.yaml')
    if os.path.exists(default_path):
        with open(default_path, 'r') as f:
            merged_config = yaml.safe_load(f) or {}

    # 2. Load other yaml files into keys named after the file
    for filename in os.listdir(config_dir):
        if filename.endswith((".yaml", ".yml")) and filename != 'default.yaml':
            file_path = os.path.join(config_dir, filename)
            config_key = os.path.splitext(filename)[0]
            with open(file_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                if yaml_content:
                    merged_config[config_key] = yaml_content

    # 3. Apply .env overrides before resolving variables
    env_file_path = os.path.join(root_dir, '.env')
    config_with_overrides = _apply_env_overrides(merged_config, env_file_path)

    # 4. Resolve variables
    final_config = _resolve_variables(config_with_overrides)

    # 5. Convert to addict.Dict
    return Dict(final_config)

if __name__ == '__main__':
    print("--- Demonstrating zo/sia/config.py ---")
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"Project Name: {config.project_name}")
        print(f"Data Sources: {config.data.sources}")
        print(f"Using Attention: {config.model.attention}")
    except Exception as e:
        print(f"Error loading configuration: {e}")

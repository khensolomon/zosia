# zo/sia/config.py
#
# What it does:
# This module provides a single function, `load_config`, to handle all project
# configurations. It has been updated to be more robust by loading files
# relative to the current working directory.
#
# Why it's used:
# This makes the configuration system behave like a standard command-line tool,
# which is essential for our testing pipeline and for predictable behavior.

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
    Loads all configurations relative to the current working directory.
    """
    # FIX: Use the current working directory as the project root.
    # This ensures that the script correctly finds the config and .env files
    # both in regular execution and during testing.
    root_dir = os.getcwd()
    config_dir = os.path.join(root_dir, 'config')
    
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Configuration directory not found at: {config_dir}")

    # Load default.yaml as the base
    merged_config = {}
    default_path = os.path.join(config_dir, 'default.yaml')
    if os.path.exists(default_path):
        with open(default_path, 'r') as f:
            merged_config = yaml.safe_load(f) or {}

    # Load other yaml files into keys named after the file
    for filename in os.listdir(config_dir):
        if filename.endswith((".yaml", ".yml")) and filename != 'default.yaml':
            file_path = os.path.join(config_dir, filename)
            config_key = os.path.splitext(filename)[0]
            with open(file_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
                if yaml_content:
                    # If a key already exists, merge deeply, otherwise set it.
                    if config_key in merged_config and isinstance(merged_config[config_key], dict):
                         _deep_merge(yaml_content, merged_config[config_key])
                    else:
                        merged_config[config_key] = yaml_content

    # Apply .env overrides from the root directory
    env_file_path = os.path.join(root_dir, '.env')
    config_with_overrides = _apply_env_overrides(merged_config, env_file_path)

    # Resolve variables
    final_config = _resolve_variables(config_with_overrides)

    return Dict(final_config)

if __name__ == '__main__':
    print("--- Demonstrating zo/sia/config.py ---")
    try:
        # To run this directly, you must be in the project's root directory
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"Project Name: {config.project_name}")
        print(f"Data Sources: {config.data.sources}")
        print(f"Using Attention: {config.model.attention}")
    except Exception as e:
        print(f"Error loading configuration: {e}")

# -----------------------------------------------------------------------------
# File: tests/integration/test_full_pipeline.py
#
# Description:
#   This file contains an integration test for the entire project pipeline.
#   It now calls the Python modules directly and uses a dedicated config
#   directory for a clean, isolated, and cross-platform test run.
#
# Usage:
#   Run tests from the root directory of the project using pytest:
#   `pytest tests/integration`
# -----------------------------------------------------------------------------

import subprocess
import pytest
from pathlib import Path
import yaml
import shutil
import sys

@pytest.fixture(scope="module")
def setup_test_environment():
    """
    A pytest fixture to set up a temporary, isolated environment for testing.
    This runs once per test module.
    """
    test_root = Path("./tests/temp_test_env").resolve()
    
    if test_root.exists():
        shutil.rmtree(test_root)
        
    paths = {
        "root": test_root,
        "config": test_root / "config",
        "data": test_root / "data",
        "parallel": test_root / "data" / "parallel_base",
        "monolingual": test_root / "data" / "monolingual" / "zo",
        "experiments": test_root / "experiments",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    # --- Create dummy data and index files ---
    # FIX: Added a third sentence to ensure the train split is not empty.
    (paths["parallel"] / "test.en").write_text(
        "this is a test.\nanother test sentence.\na third line for splitting."
    )
    (paths["parallel"] / "test.zo").write_text(
        "hih pen test khat ahi.\nkammal test dang khat.\na thumna pan ding."
    )
    (paths["parallel"] / "index.yaml").write_text(yaml.dump({"corpora": ["test"]}))
    (paths["monolingual"] / "mono.zo").write_text("leitung a nuam.")
    (paths["monolingual"] / "index.yaml").write_text(yaml.dump({"files": ["mono.zo"]}))

    # --- Create temporary config files that point to our temp directories ---
    project_config_dir = Path("./config").resolve()
    for config_file in project_config_dir.glob("*.yaml"):
        with open(config_file, 'r') as f:
            # Use as_posix() for cross-platform path compatibility in YAML
            content = f.read().replace('root: "."', f'root: "{paths["root"].as_posix()}"')
            (paths["config"] / config_file.name).write_text(content)

    yield paths

    print("\nCleaning up test environment...")
    shutil.rmtree(test_root)

def run_command(command: list[str]):
    """Helper function to run a command and check its output."""
    print(f"\nRunning command: {' '.join(command)}")
    # Use the same python executable that runs pytest for consistency
    result = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8')
    
    # Always print stdout/stderr for easier debugging
    if result.stdout: print(f"--- STDOUT ---\n{result.stdout.strip()}")
    if result.stderr: print(f"--- STDERR ---\n{result.stderr.strip()}")
        
    assert result.returncode == 0, f"Command failed with exit code {result.returncode}"
    return result

def test_full_pipeline(setup_test_environment):
    """
    Tests the full data -> train -> translate pipeline in an isolated environment.
    """
    paths = setup_test_environment
    python_executable = sys.executable
    config_dir_arg = ["--config-dir", str(paths["config"])]

    # Step 1: Run preprocessing by calling the python modules directly
    run_command([python_executable, "-m", "src.tokenizers.builder"] + config_dir_arg)
    run_command([python_executable, "-m", "src.dataset.builder", "--src_lang", "en", "--tgt_lang", "zo"] + config_dir_arg)
    
    # Step 2: Run training for one epoch
    with open(paths["config"] / "training.yaml", "r") as f:
        training_config = yaml.safe_load(f)
    training_config["training"]["num_epochs"] = 1
    training_config["training"]["batch_size"] = 1 # Use batch size of 1 for tiny test data
    with open(paths["config"] / "training.yaml", "w") as f:
        yaml.dump(training_config, f)
        
    run_command([python_executable, "-m", "src.train.trainer", "--src_lang", "en", "--tgt_lang", "zo"] + config_dir_arg)
    
    # Step 3: Test translation
    exp_dir = next(paths["experiments"].iterdir())
    model_file = next(exp_dir.glob("*.pt"))
    
    run_command([
        python_executable, "-m", "src.translate.translator",
        "--model_file", str(model_file),
        "--text", "test",
        "--src_lang", "en",
        "--tgt_lang", "zo",
    ] + config_dir_arg)

    print("\n[SUCCESS] Integration test pipeline completed successfully.")

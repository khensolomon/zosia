# tests/test_full_pipeline.py
#
# What it does:
# This file contains an integration test for the entire project pipeline.
# It has been updated to include the `supported_languages` key in the
# temporary configuration, ensuring the test environment is complete.
#
# How to run it:
# Run pytest from the project's root directory:
#   pytest tests/test_full_pipeline.py

import pytest
import os
import subprocess
import sys
import yaml
import json

# A helper function to run our scripts as command-line processes
def run_script(command, cwd):
    """
    Runs a command in a specified directory and returns its output,
    raising an error on failure.
    """
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    
    full_command = f"{sys.executable} -m {command}"
    
    result = subprocess.run(
        full_command,
        shell=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd
    )
    
    if result.returncode != 0:
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
    assert result.returncode == 0, f"Command failed: {full_command}"
    return result.stdout

@pytest.fixture(scope="module")
def temp_project(tmp_path_factory):
    """
    A pytest fixture that sets up a complete, temporary project structure
    with dummy data and configuration files.
    """
    root = tmp_path_factory.mktemp("zosia_project")
    
    dirs = [
        "config", "data/parallel_base", "data/monolingual/zo", 
        "data/locale", "experiments", "zo/sia", "scripts"
    ]
    for d in dirs:
        os.makedirs(root / d, exist_ok=True)

    (root / "data/parallel_base/word.zo").write_text("halo\nnasep")
    (root / "data/parallel_base/word.en").write_text("hello\nwork")
    (root / "data/parallel_base/index.yaml").write_text(
        "type: parallel\ntrain_pairs:\n  - word"
    )
    (root / "data/monolingual/zo/mono.txt").write_text("a inn ah a om")
    (root / "data/monolingual/zo/index.yaml").write_text("files:\n  - mono.txt")

    (root / "data/locale/en.profile.json").write_text(json.dumps({"name": "en", "ngrams": [" a", "b ", "c"]}))
    (root / "data/locale/zo.profile.json").write_text(json.dumps({"name": "zo", "ngrams": [" x", "y ", "z"]}))

    # FIX: Add the `supported_languages` key to the dummy config.
    (root / "config/default.yaml").write_text(
        'project_name: "TestSia"\n'
        'supported_languages:\n'
        '  - en\n'
        '  - zo'
    )
    (root / "config/training.yaml").write_text(
        "initial_training:\n  n_iterations: 2\n  batch_size: 1\n  optimizer: sgd\n  learning_rate: 0.01\n  clip: 1.0\n"
        "checkpoint:\n  filename_template: '{src}-{tgt}_test.pth'"
    )
    (root / "config/model.yaml").write_text(
        "attention: false\nencoder:\n  hidden_size: 8\ndecoder:\n  hidden_size: 8"
    )
    (root / "config/tokenizer.yaml").write_text(
        "max_length: 10\nspecial_tokens:\n  sos: '<s>'\n  eos: '</s>'\n  pad: '<p>'\n  unk: '<unk>'"
    )
    (root / "config/data.yaml").write_text(
        "sources:\n  - name: parallel_base\n    path: '${paths.data}/parallel_base'"
    )

    (root / ".env").write_text(
        f"paths.root={str(root).replace('\\', '/')}\n"
        f"paths.data={str(root / 'data').replace('\\', '/')}\n"
        f"paths.monolingual={str(root / 'data/monolingual').replace('\\', '/')}\n"
        f"paths.experiments={str(root / 'experiments').replace('\\', '/')}\n"
    )

    return root

def test_pipeline(temp_project):
    """
    Runs the full project pipeline in order within the temporary directory.
    """
    # 1. Main training for zo -> en
    print("Testing: main.py (zo -> en)")
    run_script("zo.sia.main --source zo --target en", cwd=temp_project)
    assert os.path.exists(temp_project / "experiments/zo-en_test.pth")

    # 2. Back-translation
    print("Testing: back_translate.py")
    run_script(
        "scripts.back_translate --source zo --target en",
        cwd=temp_project
    )
    assert os.path.exists(temp_project / "data/synthetic/zo-en/index.yaml")

    # 3. Translate script
    print("Testing: translate.py")
    output = run_script(
        "zo.sia.translate --source zo --target en --text 'halo'",
        cwd=temp_project
    )
    assert "Certainty" in output

    # 4. Suggest script
    print("Testing: suggest.py")
    output = run_script(
        "zo.sia.suggest --lang en --text 'hello'",
        cwd=temp_project
    )
    assert "Certainty" in output

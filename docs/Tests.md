# Test

## tests/unit/test_models.py

unit tests for individual components of your Transformer model. We'll use dummy inputs to check shapes and basic functionality.

## tests/integration/test_full_pipeline.py

This test will focus on ensuring that major components of your NMT pipeline work together, particularly the inference part. It might involve loading a small dummy model or the actual best_model.pt (if available and small enough for testing environments) and performing a basic translation flow.

For a full pipeline, you'd ideally mock external dependencies like actual file system operations on large datasets or web requests if your pipeline involves them. For now, we'll focus on the inference flow.

To run all tests: `pytest`

To run only unit tests: `pytest tests/unit`

To run only integration tests: `pytest tests/integration`

To see more detailed output: `pytest -v`

```bash
python -m pytest
pytest

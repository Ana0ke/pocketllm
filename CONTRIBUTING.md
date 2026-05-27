# Contributing to PocketLLM

First off, thank you for considering contributing to PocketLLM! 🧊

## How to Contribute

### Add a New Device Profile

The easiest way to contribute is adding support for a new device:

1. Create a YAML file in `src/pocketllm/devices/profiles/`
2. Use this template:

```yaml
id: your-device-id
name: Human-Readable Device Name
ram_gb: 8
cpu: CPU Model
cpu_cores: 4
arch: aarch64
gpu: GPU model or "none"
recommended_quant: int4
max_model_size_gb: 6.5
backends:
  - llama.cpp
notes: "Any notes about running models on this device"
```

3. Submit a pull request!

### Add a New Quantization Backend

1. Add your quantization method to the `GGUF_QUANT_MAP` in `quantizer.py` or create a new method
2. Add the backend dependency to `pyproject.toml` as an optional dependency
3. Add tests
4. Submit a pull request

### Report Benchmark Results

If you have access to edge hardware, we'd love your benchmark data:

1. Run `pocketllm benchmark <package_path>` on your device
2. Share the results in a GitHub Issue with the label `benchmark`

### Development Setup

```bash
# Clone the repo
git clone https://github.com/your-username/PocketLLM.git
cd PocketLLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/
```

### Code Style

- Python 3.10+
- Use type hints
- Follow PEP 8 (enforced by ruff)
- Write docstrings for public functions

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Run `pytest` and `ruff check`
6. Commit with a descriptive message
7. Push and create a Pull Request

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

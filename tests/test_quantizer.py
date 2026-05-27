"""Tests for quantizer module."""

import pytest
from pathlib import Path

from pocketllm.pipeline.quantizer import Quantizer, GGUF_QUANT_MAP


class TestQuantizer:
    """Tests for Quantizer."""

    def test_gguf_quant_map(self):
        assert "int4" in GGUF_QUANT_MAP
        assert GGUF_QUANT_MAP["int4"] == "Q4_K_M"
        assert "int8" in GGUF_QUANT_MAP
        assert GGUF_QUANT_MAP["int8"] == "Q8_0"

    def test_unsupported_method(self):
        quantizer = Quantizer()
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            quantizer.quantize(Path("/tmp/fake_model"), "invalid_method")

    def test_gptq_import_or_not_implemented(self):
        """GPTQ should raise ImportError (module missing) or NotImplementedError."""
        quantizer = Quantizer()
        with pytest.raises((ImportError, NotImplementedError)):
            quantizer.quantize(Path("/tmp/fake_model"), "gptq")

    def test_awq_import_or_not_implemented(self):
        """AWQ should raise ImportError (module missing) or NotImplementedError."""
        quantizer = Quantizer()
        with pytest.raises((ImportError, NotImplementedError)):
            quantizer.quantize(Path("/tmp/fake_model"), "awq")

    def test_mnn_import_or_not_implemented(self):
        """MNN should raise ImportError (module missing) or NotImplementedError."""
        quantizer = Quantizer()
        with pytest.raises((ImportError, NotImplementedError)):
            quantizer.quantize(Path("/tmp/fake_model"), "mnn-int4")

"""Quantizer module - Handles model quantization."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping of quantization method to GGUF type string
GGUF_QUANT_MAP = {
    "int4": "Q4_K_M",
    "int8": "Q8_0",
    "f16": "F16",
    "q4_0": "Q4_0",
    "q4_k_s": "Q4_K_S",
    "q4_k_m": "Q4_K_M",
    "q5_0": "Q5_0",
    "q5_k_m": "Q5_K_M",
    "q5_k_s": "Q5_K_S",
    "q6_k": "Q6_K",
    "q8_0": "Q8_0",
}


class Quantizer:
    """Handles LLM quantization using multiple backends."""

    def __init__(self, llama_cpp_path: Optional[str] = None):
        self.llama_cpp_path = llama_cpp_path or self._find_llama_cpp()

    def _find_llama_cpp(self) -> Optional[str]:
        """Try to find llama.cpp installation."""
        try:
            result = subprocess.run(
                ["which", "llama-quantize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return str(Path(result.stdout.strip()).parent.parent)
        except Exception:
            pass
        return None

    def quantize(
        self,
        model_path: Path,
        method: str,
        device=None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Quantize a model using the specified method.

        Args:
            model_path: Path to the downloaded model files.
            method: Quantization method (int4, int8, gptq, awq, mnn-int4, etc.)
            device: Target device profile (optional, for device-specific optimizations).
            output_dir: Output directory for quantized model.

        Returns:
            Path to the quantized model file.
        """
        if output_dir is None:
            output_dir = model_path.parent / "quantized"
        output_dir.mkdir(parents=True, exist_ok=True)

        if method in GGUF_QUANT_MAP or method.startswith("q"):
            return self._quantize_gguf(model_path, method, output_dir)
        elif method == "gptq":
            return self._quantize_gptq(model_path, output_dir)
        elif method == "awq":
            return self._quantize_awq(model_path, output_dir)
        elif method.startswith("mnn"):
            return self._quantize_mnn(model_path, method, output_dir)
        else:
            raise ValueError(
                f"Unsupported quantization method: {method}. "
                f"Supported: {list(GGUF_QUANT_MAP.keys())}, gptq, awq, mnn-int4, mnn-int8"
            )

    def _quantize_gguf(
        self, model_path: Path, method: str, output_dir: Path
    ) -> Path:
        """Quantize using llama.cpp GGUF format."""
        gguf_type = GGUF_QUANT_MAP.get(method, method.upper())
        output_file = output_dir / f"{model_path.name}-{gguf_type.lower()}.gguf"

        logger.info(f"Converting model to GGUF format: {model_path}")

        # Step 1: Convert HF to GGUF (F16 first)
        f16_gguf = output_dir / f"{model_path.name}-f16.gguf"
        self._run_llama_cpp_convert(model_path, f16_gguf)

        # Step 2: Quantize GGUF
        if gguf_type != "F16":
            logger.info(f"Quantizing to {gguf_type}...")
            self._run_llama_cpp_quantize(f16_gguf, output_file, gguf_type)
            # Clean up intermediate F16 file
            f16_gguf.unlink(missing_ok=True)
        else:
            output_file = f16_gguf

        return output_file

    def _run_llama_cpp_convert(self, model_path: Path, output: Path) -> None:
        """Run llama.cpp convert script."""
        # Try python convert script first
        convert_script = self._find_convert_script()
        if convert_script:
            cmd = [
                "python", str(convert_script),
                str(model_path), "--outfile", str(output),
                "--outtype", "f16",
            ]
        else:
            # Fallback to llama-cli convert
            cmd = [
                "llama-convert",
                str(model_path), str(output),
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(
                f"GGUF conversion failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def _run_llama_cpp_quantize(
        self, input_gguf: Path, output_gguf: Path, quant_type: str
    ) -> None:
        """Run llama-quantize."""
        cmd = ["llama-quantize", str(input_gguf), str(output_gguf), quant_type]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            raise RuntimeError(
                f"Quantization failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def _find_convert_script(self) -> Optional[Path]:
        """Find llama.cpp convert script."""
        candidates = [
            Path("llama.cpp/convert_hf_to_gguf.py"),
            Path("/usr/local/bin/convert_hf_to_gguf.py"),
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _quantize_gptq(self, model_path: Path, output_dir: Path) -> Path:
        """Quantize using auto-gptq."""
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError(
                "auto-gptq is required for GPTQ quantization. "
                "Install it with: pip install pocketllm[gptq]"
            )

        logger.info(f"Quantizing with GPTQ: {model_path}")
        # TODO: Implement GPTQ quantization pipeline
        raise NotImplementedError("GPTQ quantization will be available in v0.2")

    def _quantize_awq(self, model_path: Path, output_dir: Path) -> Path:
        """Quantize using autoawq."""
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError(
                "autoawq is required for AWQ quantization. "
                "Install it with: pip install autoawq"
            )

        logger.info(f"Quantizing with AWQ: {model_path}")
        # TODO: Implement AWQ quantization pipeline
        raise NotImplementedError("AWQ quantization will be available in v0.2")

    def _quantize_mnn(
        self, model_path: Path, method: str, output_dir: Path
    ) -> Path:
        """Quantize using MNN."""
        try:
            import MNN
        except ImportError:
            raise ImportError(
                "MNN is required for MNN quantization. "
                "Install it with: pip install MNN"
            )

        logger.info(f"Quantizing with MNN ({method}): {model_path}")
        # TODO: Implement MNN quantization pipeline
        raise NotImplementedError("MNN quantization will be available in v0.2")

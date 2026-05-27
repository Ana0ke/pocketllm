"""Device dataclass and selector logic."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent / "profiles"


@dataclass
class DeviceProfile:
    """Represents a target device's hardware profile."""

    id: str
    name: str
    ram_gb: int
    cpu: str
    cpu_cores: int
    arch: str
    gpu: str = "none"
    recommended_quant: str = "int4"
    max_model_size_gb: float = 0.0
    backends: List[str] = field(default_factory=lambda: ["llama.cpp"])
    notes: str = ""


class DeviceSelector:
    """Manages device profiles and selects optimal configurations."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or PROFILES_DIR
        self._profiles: Optional[dict[str, DeviceProfile]] = None

    @property
    def profiles(self) -> dict[str, DeviceProfile]:
        """Lazy-load device profiles from YAML files."""
        if self._profiles is None:
            self._profiles = self._load_profiles()
        return self._profiles

    def _load_profiles(self) -> dict[str, DeviceProfile]:
        """Load all device profiles from YAML files."""
        profiles = {}

        if not self.profiles_dir.exists():
            logger.warning(f"Profiles directory not found: {self.profiles_dir}")
            return profiles

        for yaml_file in self.profiles_dir.glob("*.yaml"):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                profile = DeviceProfile(
                    id=data["id"],
                    name=data["name"],
                    ram_gb=data["ram_gb"],
                    cpu=data["cpu"],
                    cpu_cores=data["cpu_cores"],
                    arch=data["arch"],
                    gpu=data.get("gpu", "none"),
                    recommended_quant=data.get("recommended_quant", "int4"),
                    max_model_size_gb=data.get("max_model_size_gb", 0.0),
                    backends=data.get("backends", ["llama.cpp"]),
                    notes=data.get("notes", ""),
                )
                profiles[profile.id] = profile
            except Exception as e:
                logger.error(f"Failed to load profile {yaml_file}: {e}")

        return profiles

    def get_device(self, device_id: str) -> Optional[DeviceProfile]:
        """Get a device profile by ID."""
        return self.profiles.get(device_id)

    def list_devices(self) -> List[DeviceProfile]:
        """List all available device profiles."""
        return list(self.profiles.values())

    def recommend_quant(
        self, device: DeviceProfile, model_size_mb: float
    ) -> str:
        """Recommend the best quantization method based on device and model size.

        Args:
            device: Target device profile.
            model_size_mb: Size of the original model in MB.

        Returns:
            Recommended quantization method string.
        """
        # Simple heuristic: if model is too large for device RAM, use more aggressive quant
        model_size_gb = model_size_mb / 1024
        usable_ram = device.ram_gb * 0.75  # Leave 25% for system

        if model_size_gb * 0.25 <= usable_ram:
            return "int4"  # 4-bit fits comfortably
        elif model_size_gb * 0.5 <= usable_ram:
            return "int4"  # 4-bit still needed
        else:
            return "int4"  # Even int4 might not fit, but it's the best we can do

    def find_compatible_devices(
        self, model_size_mb: float, quant: str = "int4"
    ) -> List[DeviceProfile]:
        """Find devices that can run a model at the given quantization level.

        Args:
            model_size_mb: Size of the original model in MB.
            quant: Target quantization level.

        Returns:
            List of compatible device profiles.
        """
        # Rough compression ratios
        compression = {"int4": 0.25, "int8": 0.5, "f16": 1.0}
        ratio = compression.get(quant, 0.25)
        quantized_size_gb = (model_size_mb / 1024) * ratio

        compatible = []
        for device in self.profiles.values():
            if device.max_model_size_gb > 0:
                if quantized_size_gb <= device.max_model_size_gb:
                    compatible.append(device)
            else:
                # Estimate: 75% of RAM available for model
                if quantized_size_gb <= device.ram_gb * 0.75:
                    compatible.append(device)

        return compatible

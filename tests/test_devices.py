"""Tests for device profile loading and selection."""

import pytest
from pathlib import Path

from pocketllm.devices.selector import DeviceSelector, DeviceProfile


class TestDeviceProfile:
    """Tests for DeviceProfile dataclass."""

    def test_create_profile(self):
        profile = DeviceProfile(
            id="test-device",
            name="Test Device",
            ram_gb=4,
            cpu="ARM Cortex-A76",
            cpu_cores=4,
            arch="aarch64",
        )
        assert profile.id == "test-device"
        assert profile.ram_gb == 4
        assert profile.recommended_quant == "int4"  # default
        assert profile.backends == ["llama.cpp"]  # default


class TestDeviceSelector:
    """Tests for DeviceSelector."""

    def test_load_profiles(self):
        selector = DeviceSelector()
        profiles = selector.profiles
        assert len(profiles) > 0

    def test_get_device(self):
        selector = DeviceSelector()
        device = selector.get_device("rpi5")
        assert device is not None
        assert device.name == "Raspberry Pi 5"
        assert device.ram_gb == 8

    def test_get_unknown_device(self):
        selector = DeviceSelector()
        device = selector.get_device("nonexistent")
        assert device is None

    def test_list_devices(self):
        selector = DeviceSelector()
        devices = selector.list_devices()
        assert len(devices) > 0
        assert all(isinstance(d, DeviceProfile) for d in devices)

    def test_recommend_quant(self):
        selector = DeviceSelector()
        device = selector.get_device("rpi5")
        # Small model should recommend int4
        quant = selector.recommend_quant(device, model_size_mb=1500)
        assert quant == "int4"

    def test_find_compatible_devices(self):
        selector = DeviceSelector()
        # A small model should be compatible with many devices
        compatible = selector.find_compatible_devices(model_size_mb=500, quant="int4")
        assert len(compatible) > 0

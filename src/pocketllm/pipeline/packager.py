"""Packager module - Creates device-specific deployment packages."""

import logging
import shutil
import tarfile
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


class Packager:
    """Creates deployment packages for target devices."""

    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent.parent.parent.parent / "templates"
        self.templates_dir = templates_dir

    def package(
        self,
        quantized_path: Path,
        device,
        output_dir: Path,
    ) -> Path:
        """Create a deployment package for the target device.

        Args:
            quantized_path: Path to the quantized model file.
            device: Target device profile.
            output_dir: Output directory for the package.

        Returns:
            Path to the created package.
        """
        device_type = self._get_device_type(device)
        package_dir = output_dir / "package"
        package_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        model_dest = package_dir / "model.gguf"
        if quantized_path.is_file():
            shutil.copy2(quantized_path, model_dest)
        else:
            shutil.copytree(quantized_path, package_dir / "model", dirs_exist_ok=True)

        # Generate device-specific files from templates
        self._generate_from_templates(device_type, device, package_dir)

        # Create launch script
        self._create_launch_script(device, package_dir)

        # Create tar.gz archive
        archive_path = self._create_archive(package_dir, output_dir, device)

        return archive_path

    def _get_device_type(self, device) -> str:
        """Determine device type from profile."""
        device_id = device.id if hasattr(device, "id") else str(device)
        if "rpi" in device_id or "raspberry" in device_id:
            return "raspberry_pi"
        elif "jetson" in device_id:
            return "jetson"
        elif "android" in device_id:
            return "android"
        else:
            return "generic_linux"

    def _generate_from_templates(
        self, device_type: str, device, package_dir: Path
    ) -> None:
        """Generate configuration files from Jinja2 templates."""
        template_dir = self.templates_dir / device_type
        if not template_dir.exists():
            logger.warning(f"No templates found for {device_type}, using generic")
            template_dir = self.templates_dir / "generic_linux"

        if not template_dir.exists():
            return

        env = Environment(loader=FileSystemLoader(str(template_dir)))

        for template_file in template_dir.glob("*.j2"):
            template = env.get_template(template_file.name)
            rendered = template.render(
                device=device,
                model_file="model.gguf",
                threads=device.cpu_cores if hasattr(device, "cpu_cores") else 4,
            )
            output_file = package_dir / template_file.stem  # Remove .j2 extension
            output_file.write_text(rendered)
            logger.info(f"Generated: {output_file}")

    def _create_launch_script(self, device, package_dir: Path) -> None:
        """Create the main launch script."""
        threads = device.cpu_cores if hasattr(device, "cpu_cores") else 4
        model_file = "model.gguf"

        script = package_dir / "run.sh"
        script.write_text(f"""#!/bin/bash
# PocketLLM - Auto-generated launch script
# Target: {device.name if hasattr(device, 'name') else device}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="$SCRIPT_DIR/{model_file}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found: $MODEL"
    exit 1
fi

echo "🧊 PocketLLM - Starting model server..."
echo "   Model: {model_file}"
echo "   Device: {device.name if hasattr(device, 'name') else device}"
echo "   Threads: {threads}"

llama-server \\
    -m "$MODEL" \\
    -t {threads} \\
    -c 2048 \\
    --host 0.0.0.0 \\
    --port 8080

# For interactive chat instead of server, use:
# llama-cli -m "$MODEL" -t {threads} -c 2048 -i
""")
        script.chmod(0o755)
        logger.info(f"Created launch script: {script}")

    def _create_archive(self, package_dir: Path, output_dir: Path, device) -> Path:
        """Create a tar.gz archive of the package."""
        device_name = device.id if hasattr(device, "id") else "unknown"
        archive_name = f"pocketllm-{device_name}"
        archive_path = output_dir / archive_name

        with tarfile.open(f"{archive_path}.tar.gz", "w:gz") as tar:
            tar.add(package_dir, arcname=archive_name)

        logger.info(f"Created archive: {archive_path}.tar.gz")
        return Path(f"{archive_path}.tar.gz")

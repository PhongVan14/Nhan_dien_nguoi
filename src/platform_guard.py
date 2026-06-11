from __future__ import annotations

import platform
from pathlib import Path


def is_jetson() -> bool:
    """Return True when running on an NVIDIA Jetson Linux device."""
    if Path("/etc/nv_tegra_release").exists():
        return True

    model_path = Path("/proc/device-tree/model")
    try:
        model = model_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    lowered = model.lower()
    return "nvidia" in lowered and "jetson" in lowered


def platform_summary() -> str:
    return (
        f"{platform.platform()} | machine={platform.machine()} | "
        f"processor={platform.processor() or 'unknown'}"
    )


def require_jetson(*, allow_non_jetson: bool = False) -> None:
    if allow_non_jetson or is_jetson():
        return

    raise SystemExit(
        "This project is configured to run only on NVIDIA Jetson.\n"
        f"Detected: {platform_summary()}\n"
        "Copy the project to the Jetson and run the same command there. "
        "For an intentional laptop smoke test, add --allow-non-jetson."
    )

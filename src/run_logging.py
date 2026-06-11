import contextlib
import json
import platform
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterator, Optional, TextIO


class TeeWriter:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return self._streams[0].isatty()

    @property
    def encoding(self) -> Optional[str]:
        return self._streams[0].encoding


@contextlib.contextmanager
def tee_run_log(script_name: str, log_dir: Path) -> Iterator[Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_{script_name}_{Path.cwd().name}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        sys.stdout = TeeWriter(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = TeeWriter(original_stderr, log_file)  # type: ignore[assignment]
        try:
            print(f"Log file: {log_path}")
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def print_run_context(script_name: str, args: Optional[Namespace] = None) -> None:
    print(f"\n=== {script_name} context ===")
    print(f"command: {' '.join(sys.argv)}")
    print(f"cwd: {Path.cwd()}")
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.platform()}")
    if args is not None:
        print("args:")
        print(json.dumps(_safe_namespace(args), ensure_ascii=False, indent=2))

    _print_torch_context()
    _print_nvidia_smi()
    print(f"=== end context ===\n")


def _safe_namespace(args: Namespace) -> Dict[str, object]:
    output = {}
    for key, value in vars(args).items():
        if key.startswith("_"):
            continue
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, (list, tuple)):
            output[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            output[key] = value
    return output


def _print_torch_context() -> None:
    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch_context_error: {exc}")


def _print_nvidia_smi() -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                "--format=csv",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        print(f"nvidia_smi_error: {exc}")
        return

    if result.stdout.strip():
        print("nvidia-smi:")
        print(result.stdout.strip())
    if result.stderr.strip():
        print("nvidia-smi stderr:")
        print(result.stderr.strip())

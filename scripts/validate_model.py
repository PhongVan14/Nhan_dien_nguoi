from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_three_models import resolve_device
from src.config import load_app_config
from src.platform_guard import require_jetson
from src.run_logging import print_run_context, tee_run_log


def parse_args() -> argparse.Namespace:
    config = load_app_config()
    parser = argparse.ArgumentParser(description="Validate a trained detector on YOLO data.")
    parser.add_argument("--weights", type=Path, help="Path to .pt weights.")
    parser.add_argument("--model", choices=sorted(config.models), help="Use configured weights.")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "val")
    parser.add_argument("--name", default="validation")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "runs" / "logs")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--allow-non-jetson",
        action="store_true",
        help="Allow this script to run outside NVIDIA Jetson.",
    )
    args = parser.parse_args()
    args._config = config
    return args


def main() -> None:
    args = parse_args()

    if args.no_log:
        validate_from_args(args)
        return

    with tee_run_log("validate", args.log_dir) as log_path:
        try:
            print_run_context("validate_model.py", args)
            validate_from_args(args)
            print(f"\nRun completed. Log saved to: {log_path}")
        except BaseException:
            print("\nRun failed. Full traceback:")
            traceback.print_exc()
            print(f"\nLog saved to: {log_path}")
            raise


def validate_from_args(args: argparse.Namespace) -> None:
    require_jetson(allow_non_jetson=args.allow_non_jetson)

    weights = resolve_weights(args)
    device = resolve_device(args.device)

    from ultralytics import YOLO

    print(f"Validating weights={weights}")
    print(f"Using device={device}, split={args.split}, imgsz={args.imgsz}")
    metrics = YOLO(str(weights)).val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )

    print(f"precision: {float(metrics.box.mp):.4f}")
    print(f"recall: {float(metrics.box.mr):.4f}")
    print(f"mAP50: {float(metrics.box.map50):.4f}")
    print(f"mAP50-95: {float(metrics.box.map):.4f}")


def resolve_weights(args: argparse.Namespace) -> Path | str:
    if args.weights:
        return args.weights.resolve()
    if args.model:
        return args._config.models[args.model].weights
    raise SystemExit("Pass --weights path\\to\\best.pt or --model fast|balanced|accurate")


if __name__ == "__main__":
    main()

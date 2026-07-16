from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_three_models import prepare_dataset_config, resolve_device
from src.config import load_app_config
from src.platform_guard import require_jetson
from src.run_logging import print_run_context, tee_run_log


DEFAULT_START_WEIGHTS = {
    "fast": ROOT / "runs" / "train_finetune_20" / "person_counter_fast" / "weights" / "best.pt",
    "balanced": ROOT
    / "runs"
    / "train_balanced_10ep_review"
    / "person_counter_balanced"
    / "weights"
    / "best.pt",
    "accurate": ROOT / "yolov8m.pt",
}


def parse_args() -> argparse.Namespace:
    config = load_app_config()
    choices = ["all", *sorted(config.models)]
    parser = argparse.ArgumentParser(
        description="Fine-tune selected people models for a few more epochs."
    )
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    parser.add_argument("--models", nargs="+", default=["all"], choices=choices)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "train_more")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cache", choices=["false", "ram", "disk"], default="false")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--close-mosaic", type=int, default=5)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--opset", type=int, default=10)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "runs" / "logs")
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write stdout/stderr directly to this file. Useful for background runs.",
    )
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--allow-non-jetson",
        action="store_true",
        help="Allow this script to run outside NVIDIA Jetson.",
    )
    parser.add_argument("--fast-weights", type=Path)
    parser.add_argument("--balanced-weights", type=Path)
    parser.add_argument("--accurate-weights", type=Path)
    args = parser.parse_args()
    args._config = config
    return args


def main() -> None:
    args = parse_args()
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        with args.log_file.open("w", encoding="utf-8", errors="replace") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                try:
                    print_run_context("finetune_more_models.py", args)
                    finetune_from_args(args)
                    print(f"\nRun completed. Log saved to: {args.log_file}")
                except BaseException:
                    print("\nRun failed. Full traceback:")
                    traceback.print_exc()
                    print(f"\nLog saved to: {args.log_file}")
                    raise
        return

    if args.no_log:
        finetune_from_args(args)
        return

    with tee_run_log("finetune_more", args.log_dir) as log_path:
        try:
            print_run_context("finetune_more_models.py", args)
            finetune_from_args(args)
            print(f"\nRun completed. Log saved to: {log_path}")
        except BaseException:
            print("\nRun failed. Full traceback:")
            traceback.print_exc()
            print(f"\nLog saved to: {log_path}")
            raise


def finetune_from_args(args: argparse.Namespace) -> None:
    require_jetson(allow_non_jetson=args.allow_non_jetson)

    data_path = prepare_dataset_config(args.data)
    selected = list(args._config.models) if "all" in args.models else args.models
    device = resolve_device(args.device)
    cache = False if args.cache == "false" else args.cache
    project_path = args.project.resolve()

    from ultralytics import YOLO

    for model_key in selected:
        model_config = args._config.models[model_key]
        start_weights = resolve_start_weights(args, model_key, model_config.weights)
        imgsz = model_config.imgsz
        run_name = f"person_counter_{model_key}_more{args.epochs}ep"

        print(f"\nFine-tuning {model_key}: {model_config.name}")
        print(f"Start weights: {start_weights}")
        print(f"Using device={device}, imgsz={imgsz}, batch={args.batch}")

        model = YOLO(str(start_weights))
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            project=str(project_path),
            name=run_name,
            exist_ok=args.exist_ok,
            device=device,
            workers=args.workers,
            fraction=args.fraction,
            cache=cache,
            cos_lr=args.cos_lr,
            patience=args.patience,
            close_mosaic=args.close_mosaic,
            optimizer="AdamW",
            lr0=args.lr0,
            lrf=args.lrf,
        )

        best_path = project_path / run_name / "weights" / "best.pt"
        print(f"Best weights: {best_path}")
        if not args.no_export:
            export_to_models_dir(
                best_path=best_path,
                model_key=model_key,
                epochs=args.epochs,
                imgsz=imgsz,
                opset=args.opset,
                models_dir=args.models_dir,
            )


def resolve_start_weights(
    args: argparse.Namespace,
    model_key: str,
    configured_weights: str,
) -> Path | str:
    override = {
        "fast": args.fast_weights,
        "balanced": args.balanced_weights,
        "accurate": args.accurate_weights,
    }[model_key]
    if override is not None:
        return override.resolve()

    default = DEFAULT_START_WEIGHTS[model_key]
    if default.exists():
        return default.resolve()
    return configured_weights


def export_to_models_dir(
    *,
    best_path: Path,
    model_key: str,
    epochs: int,
    imgsz: int,
    opset: int,
    models_dir: Path,
) -> None:
    if not best_path.exists():
        raise FileNotFoundError(f"Best weights not found: {best_path}")

    from ultralytics import YOLO

    print(f"Exporting ONNX from: {best_path}")
    exported = YOLO(str(best_path)).export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
    )
    exported_path = Path(exported)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / f"person_counter_{model_key}_finetune_more{epochs}_opset{opset}.onnx"
    shutil.copy2(exported_path, target)
    print(f"Exported ONNX copied to: {target}")


if __name__ == "__main__":
    main()

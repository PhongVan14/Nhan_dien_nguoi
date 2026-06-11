from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import describe_models, load_app_config
from src.platform_guard import require_jetson
from src.run_logging import print_run_context, tee_run_log


def parse_args() -> argparse.Namespace:
    config = load_app_config()
    choices = ["all", *sorted(config.models)]
    parser = argparse.ArgumentParser(description="Train one or all configured people models.")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    parser.add_argument("--models", nargs="+", default=["all"], choices=choices)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for training: auto, cpu, 0, cuda:0.",
    )
    parser.add_argument("--imgsz", type=int, help="Override training image size.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training set to use. Use a small value for smoke tests.",
    )
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "train")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--cache", choices=["false", "ram", "disk"], default="false")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "runs" / "logs")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--allow-non-jetson",
        action="store_true",
        help="Allow this script to run outside NVIDIA Jetson.",
    )
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    args._config = config
    return args


def main() -> None:
    args = parse_args()
    config = args._config

    if args.no_log:
        run_training(args, config)
        return

    with tee_run_log("train", args.log_dir) as log_path:
        try:
            print_run_context("train_three_models.py", args)
            run_training(args, config)
            print(f"\nRun completed. Log saved to: {log_path}")
        except BaseException:
            print("\nRun failed. Full traceback:")
            traceback.print_exc()
            print(f"\nLog saved to: {log_path}")
            raise


def run_training(args: argparse.Namespace, config) -> None:
    if args.list_models:
        print(describe_models(config))
        return

    require_jetson(allow_non_jetson=args.allow_non_jetson)

    try:
        data_path = prepare_dataset_config(args.data)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Dataset error:\n{exc}") from None

    selected = list(config.models) if "all" in args.models else args.models
    project_path = args.project.resolve()
    device = resolve_device(args.device)
    cache = False if args.cache == "false" else args.cache

    from ultralytics import YOLO

    for model_key in selected:
        model_config = config.models[model_key]
        imgsz = args.imgsz or model_config.imgsz
        run_name = f"person_counter_{model_key}"
        print(f"\nTraining {model_key}: {model_config.name}")
        print(f"Using device={device}, imgsz={imgsz}, batch={args.batch}")
        model = YOLO(model_config.weights)
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
        )
        best_path = project_path / run_name / "weights" / "best.pt"
        print(f"Best weights expected at: {best_path}")


def resolve_device(value: str) -> str:
    if value != "auto":
        return value

    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def prepare_dataset_config(data_path: Path) -> Path:
    data_path = data_path.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    raw = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    dataset_root = _resolve_dataset_root(raw.get("path", "."), data_path)

    train_dir = _resolve_split_path(dataset_root, raw.get("train"), "train")
    val_dir = _resolve_split_path(dataset_root, raw.get("val"), "val")

    _require_images(train_dir, "train")
    _require_images(val_dir, "val")

    normalized = dict(raw)
    normalized["path"] = str(dataset_root)
    generated_path = ROOT / "runs" / "dataset_data.resolved.yaml"
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(
        yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return generated_path


def _resolve_dataset_root(path_value: str | Path, data_path: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()

    candidates = [
        (ROOT / candidate).resolve(),
        (data_path.parent / candidate).resolve(),
        (Path.cwd() / candidate).resolve(),
    ]
    for item in candidates:
        if item.exists():
            return item

    return candidates[0]


def _resolve_split_path(dataset_root: Path, split_value: str | None, split_name: str) -> Path:
    if not split_value:
        raise ValueError(f"Missing '{split_name}' split in dataset config")

    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return dataset_root / split_path


def _require_images(directory: Path, split_name: str) -> None:
    image_extensions = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset split '{split_name}' not found: {directory}\n"
            "Put your YOLO images in dataset/images/train and dataset/images/val."
        )

    image_count = sum(
        1
        for file_path in directory.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in image_extensions
    )
    if image_count == 0:
        raise RuntimeError(
            f"Dataset split '{split_name}' has no images: {directory}\n"
            "Add labeled images before training. For each image, add the matching "
            "YOLO label file under dataset/labels/train or dataset/labels/val."
        )


if __name__ == "__main__":
    main()

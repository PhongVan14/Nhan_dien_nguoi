import argparse
import csv
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_logging import print_run_context, tee_run_log

DEFAULT_ARCHIVE = ROOT / "dataset" / "archive.zip"
DEFAULT_OUTPUT = ROOT / "dataset"

SPLITS = {
    "train": ("train/train", "train"),
    "valid": ("valid/valid", "val"),
    "test": ("test/test", "test"),
}
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
CLASS_TO_ID = {"person": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Kaggle/Roboflow people archive to YOLO format."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract images and rewrite labels even when files already exist.",
    )
    parser.add_argument("--log-dir", type=Path, default=ROOT / "runs" / "logs")
    parser.add_argument("--no-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_log:
        prepare_from_args(args)
        return

    with tee_run_log("prepare_dataset", args.log_dir) as log_path:
        try:
            print_run_context("prepare_archive_dataset.py", args)
            prepare_from_args(args)
            print(f"\nRun completed. Log saved to: {log_path}")
        except BaseException:
            print("\nRun failed. Full traceback:")
            traceback.print_exc()
            print(f"\nLog saved to: {log_path}")
            raise


def prepare_from_args(args: argparse.Namespace) -> None:
    archive_path = args.archive.resolve()
    output_dir = args.output.resolve()

    if not archive_path.exists():
        raise SystemExit(f"Archive not found: {archive_path}")

    stats = prepare_dataset(archive_path, output_dir, overwrite=args.overwrite)
    write_data_yaml(output_dir)
    clear_label_caches(output_dir)

    print("Prepared YOLO dataset")
    for split, values in stats.items():
        print(
            f"{split}: {values['images']} images, "
            f"{values['label_files']} label files, {values['boxes']} boxes, "
            f"{values['duplicates_removed']} duplicates removed"
        )
    print(f"Data config: {output_dir / 'data.yaml'}")


def prepare_dataset(
    archive_path,
    output_dir,
    *,
    overwrite=False,
):
    stats = {}
    with ZipFile(archive_path) as archive:
        for source_split, (source_prefix, target_split) in SPLITS.items():
            annotations = read_annotations(archive, f"{source_prefix}/_annotations.csv")
            images = image_entries(archive, source_prefix)

            image_dir = output_dir / "images" / target_split
            label_dir = output_dir / "labels" / target_split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            boxes_written = 0
            duplicate_boxes = 0
            label_files = 0
            for image_name, zip_name in images.items():
                target_image = image_dir / image_name
                if overwrite or not target_image.exists():
                    archive.getinfo(zip_name)
                    with archive.open(zip_name) as source, target_image.open("wb") as target:
                        target.write(source.read())

                labels = annotations.get(image_name, [])
                boxes_written += len(labels)
                duplicate_boxes += getattr(labels, "duplicates", 0)
                if labels or overwrite or not (label_dir / f"{Path(image_name).stem}.txt").exists():
                    (label_dir / f"{Path(image_name).stem}.txt").write_text(
                        "".join(labels),
                        encoding="utf-8",
                    )
                label_files += 1

            stats[target_split] = {
                "images": len(images),
                "label_files": label_files,
                "boxes": boxes_written,
                "duplicates_removed": duplicate_boxes,
            }

    return stats


class LabelRows(list):
    duplicates = 0


def read_annotations(archive, csv_name):
    rows_by_image = defaultdict(set)
    duplicates_by_image = defaultdict(int)
    with archive.open(csv_name) as raw:
        text = (line.decode("utf-8-sig") for line in raw)
        reader = csv.DictReader(line for line in text if line.strip())

        for row in reader:
            class_name = row["class"].strip()
            if class_name not in CLASS_TO_ID:
                continue

            width = float(row["width"])
            height = float(row["height"])
            xmin = _clamp(float(row["xmin"]), 0.0, width)
            ymin = _clamp(float(row["ymin"]), 0.0, height)
            xmax = _clamp(float(row["xmax"]), 0.0, width)
            ymax = _clamp(float(row["ymax"]), 0.0, height)

            box_width = xmax - xmin
            box_height = ymax - ymin
            if box_width <= 0 or box_height <= 0:
                continue

            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            normalized_width = box_width / width
            normalized_height = box_height / height

            label_line = (
                f"{CLASS_TO_ID[class_name]} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{normalized_width:.6f} {normalized_height:.6f}\n"
            )
            filename = row["filename"]
            before = len(rows_by_image[filename])
            rows_by_image[filename].add(label_line)
            if len(rows_by_image[filename]) == before:
                duplicates_by_image[filename] += 1

    result = {}
    for image_name, labels in rows_by_image.items():
        rows = LabelRows(sorted(labels))
        rows.duplicates = duplicates_by_image[image_name]
        result[image_name] = rows
    return result


def image_entries(archive, source_prefix):
    prefix = f"{source_prefix}/"
    images = {}
    for name in archive.namelist():
        path = Path(name)
        if not name.startswith(prefix):
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images[path.name] = name
    return images


def write_data_yaml(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    try:
        dataset_path = resolved_output.relative_to(ROOT).as_posix()
    except ValueError:
        dataset_path = str(resolved_output)

    data = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["person"],
    }
    try:
        yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    except TypeError:
        yaml_text = yaml.safe_dump(data, allow_unicode=True)

    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def clear_label_caches(output_dir: Path) -> None:
    labels_dir = output_dir / "labels"
    for cache_path in labels_dir.glob("*.cache"):
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


if __name__ == "__main__":
    sys.exit(main())

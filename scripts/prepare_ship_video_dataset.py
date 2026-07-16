from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = ".jpg"


@dataclass
class Sample:
    source: Path
    source_index: int
    frame_index: int
    timestamp_seconds: float
    frame: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from ship videos and optionally create starter YOLO labels "
            "with a pretrained COCO boat detector."
        )
    )
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset_ship")
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Seconds between extracted frames from each video.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        help="Optional cap after temporal sampling.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Bootstrap labels with a pretrained detector. Review every label afterward.",
    )
    parser.add_argument("--weights", default="yolov8m.pt")
    parser.add_argument(
        "--source-class-id",
        type=int,
        default=8,
        help="Class to copy from the bootstrap model. COCO boat is 8.",
    )
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", help="Inference device for auto-labeling, e.g. cpu or 0.")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing output directory without deleting files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_seconds <= 0:
        raise SystemExit("--sample-seconds must be greater than 0")
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")
    if args.max_frames_per_video is not None and args.max_frames_per_video <= 0:
        raise SystemExit("--max-frames-per-video must be greater than 0")
    missing = [str(path) for path in args.source if not path.is_file()]
    if missing:
        raise SystemExit("Video file not found:\n" + "\n".join(missing))


def make_output_dirs(output: Path, exist_ok: bool) -> None:
    if output.exists() and any(output.iterdir()) and not exist_ok:
        raise SystemExit(
            f"Output directory is not empty: {output}\n"
            "Use a new directory or pass --exist-ok. Existing files are never deleted."
        )
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_for_sample(source_index: int, source_count: int, sample_index: int, total: int, val_ratio: float) -> str:
    if source_count > 1:
        val_sources = max(1, int(round(source_count * val_ratio)))
        return "val" if source_index >= source_count - val_sources else "train"

    val_samples = max(1, int(round(total * val_ratio)))
    return "val" if sample_index >= total - val_samples else "train"


def iter_samples(video_path: Path, source_index: int, args: argparse.Namespace):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    stride = max(1, int(round(fps * args.sample_seconds)))
    frame_index = 0
    sample_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                yield Sample(
                    source=video_path.resolve(),
                    source_index=source_index,
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    frame=frame,
                )
                sample_index += 1
                if (
                    args.max_frames_per_video is not None
                    and sample_index >= args.max_frames_per_video
                ):
                    break
            frame_index += 1
    finally:
        capture.release()


def expected_sample_count(video_path: Path, args: argparse.Namespace) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = 0
            while capture.grab():
                frame_count += 1
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    stride = max(1, int(round(fps * args.sample_seconds)))
    count = int(math.ceil(frame_count / stride)) if frame_count > 0 else 0
    if args.max_frames_per_video is not None:
        count = min(count, args.max_frames_per_video) if count else args.max_frames_per_video
    return count


def create_bootstrap_model(args: argparse.Namespace):
    if not args.auto_label:
        return None
    from ultralytics import YOLO

    return YOLO(args.weights)


def make_labels(model, frame, args: argparse.Namespace) -> list[str]:
    if model is None:
        return []

    results = model.predict(
        source=frame,
        classes=[args.source_class_id],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    labels = []
    for x_center, y_center, width, height in boxes.xywhn.cpu().tolist():
        labels.append(
            "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
                x_center, y_center, width, height
            )
        )
    return labels


def safe_stem(path: Path) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in path.stem)
    return cleaned.strip("_") or "video"


def write_dataset(args: argparse.Namespace) -> tuple[int, int, int]:
    output = args.output.resolve()
    make_output_dirs(output, args.exist_ok)
    model = create_bootstrap_model(args)
    manifest_rows = []
    train_count = 0
    val_count = 0
    box_count = 0

    for source_index, video_path in enumerate(args.source):
        total_samples = expected_sample_count(video_path, args)
        extracted_from_source = 0
        for sample_index, sample in enumerate(iter_samples(video_path, source_index, args)):
            extracted_from_source += 1
            split = split_for_sample(
                source_index,
                len(args.source),
                sample_index,
                total_samples,
                args.val_ratio,
            )
            name = (
                f"{source_index:02d}_{safe_stem(video_path)}_"
                f"f{sample.frame_index:09d}"
            )
            image_path = output / "images" / split / f"{name}{IMAGE_EXT}"
            label_path = output / "labels" / split / f"{name}.txt"

            written = cv2.imwrite(
                str(image_path),
                sample.frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not written:
                raise RuntimeError(f"Could not write image: {image_path}")

            labels = make_labels(model, sample.frame, args)
            label_path.write_text(
                ("\n".join(labels) + "\n") if labels else "",
                encoding="utf-8",
            )
            box_count += len(labels)
            train_count += split == "train"
            val_count += split == "val"
            manifest_rows.append(
                {
                    "image": str(image_path.relative_to(output)),
                    "source": str(sample.source),
                    "frame": sample.frame_index,
                    "timestamp_seconds": f"{sample.timestamp_seconds:.3f}",
                    "split": split,
                    "bootstrap_boxes": len(labels),
                }
            )
        if extracted_from_source == 0:
            print(f"Warning: no frames extracted from {video_path}")

    if train_count == 0 or val_count == 0:
        raise RuntimeError(
            "Dataset needs at least one train image and one val image. "
            "Use a longer video or lower --sample-seconds."
        )

    data = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "ship"},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    return train_count, val_count, box_count


def main() -> None:
    args = parse_args()
    validate_args(args)
    train_count, val_count, box_count = write_dataset(args)
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Bootstrap boxes: {box_count}")
    print(f"Dataset: {args.output.resolve()}")
    if args.auto_label:
        print("Review and correct all bootstrap labels before training.")
    else:
        print("Labels are empty placeholders. Annotate every visible ship before training.")


if __name__ == "__main__":
    main()

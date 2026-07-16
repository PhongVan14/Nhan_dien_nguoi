from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track ships, then alert when a track persists inside an ROI."
    )
    parser.add_argument("--source", required=True, help="Video path, stream URL, or webcam id.")
    parser.add_argument("--weights", default=str(ROOT / "runs/train/ship_detector/weights/best.pt"))
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", help="Inference device, e.g. cpu or 0.")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--roi",
        type=float,
        nargs="+",
        help=(
            "Polygon as x1 y1 x2 y2 ... Use values from 0 to 1 for normalized "
            "coordinates, or pixel coordinates. Omit to monitor the full frame."
        ),
    )
    parser.add_argument(
        "--min-alert-frames",
        type=int,
        default=15,
        help="Consecutive frames inside the ROI before alerting.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def make_roi(values: list[float] | None, width: int, height: int) -> np.ndarray | None:
    if values is None:
        return None
    if len(values) < 6 or len(values) % 2 != 0:
        raise SystemExit("--roi needs at least three x y coordinate pairs")
    points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    if np.all((points >= 0.0) & (points <= 1.0)):
        points[:, 0] *= width
        points[:, 1] *= height
    return np.rint(points).astype(np.int32)


def inside_roi(center: tuple[int, int], roi: np.ndarray | None) -> bool:
    return roi is None or cv2.pointPolygonTest(roi, center, False) >= 0


def make_writer(path: Path | None, fps: float, width: int, height: int):
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    y = max(22, y)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    if args.min_alert_frames <= 0:
        raise SystemExit("--min-alert-frames must be greater than 0")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    roi = make_roi(args.roi, width, height)
    writer = make_writer(args.output, fps, width, height)
    consecutive: dict[int, int] = {}
    alerted: set[int] = set()
    frame_index = 0
    started_at = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            result = model.track(
                source=frame,
                persist=True,
                classes=[args.class_id],
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                tracker=args.tracker,
                verbose=False,
            )[0]
            seen_ids = set()
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                coords = boxes.xyxy.int().cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                for track_id, (x1, y1, x2, y2), score in zip(ids, coords, scores):
                    seen_ids.add(track_id)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    is_inside = inside_roi(center, roi)
                    consecutive[track_id] = consecutive.get(track_id, 0) + 1 if is_inside else 0
                    is_alert = consecutive[track_id] >= args.min_alert_frames
                    color = (0, 0, 255) if is_alert else (0, 200, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label(
                        frame,
                        f"ship #{track_id} {score:.2f}" + (" ALERT" if is_alert else ""),
                        x1,
                        y1 - 8,
                        color,
                    )
                    if is_alert and track_id not in alerted:
                        alerted.add(track_id)
                        seconds = frame_index / fps
                        print(f"ALERT ship #{track_id} at {seconds:.1f}s")

            for track_id in list(consecutive):
                if track_id not in seen_ids:
                    del consecutive[track_id]

            if roi is not None:
                cv2.polylines(frame, [roi], True, (255, 180, 0), 2)
            draw_label(frame, f"Alerts: {len(alerted)}", 20, 32, (0, 0, 255))
            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("Ship Intrusion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Frames: {frame_index}")
    print(f"Alerts: {len(alerted)}")
    print(f"Time: {time.time() - started_at:.1f}s")
    if args.output:
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

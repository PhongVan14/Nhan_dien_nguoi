from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import describe_models, load_app_config
from src.platform_guard import require_jetson
from src.run_logging import print_run_context, tee_run_log


def parse_args() -> argparse.Namespace:
    config = load_app_config()
    parser = argparse.ArgumentParser(
        description="Detect people and show the current people count."
    )
    parser.add_argument("--source", default="0", help="Video path, stream URL, or webcam id.")
    parser.add_argument(
        "--model",
        default="fast",
        choices=sorted(config.models),
        help="Configured model key.",
    )
    parser.add_argument("--weights", help="Override weights path/name, e.g. models/best.pt.")
    parser.add_argument("--output", type=Path, help="Optional output video path.")
    parser.add_argument("--show", action="store_true", help="Show OpenCV preview window.")
    parser.add_argument("--device", help="Device for inference, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--conf", type=float, help="Override confidence threshold.")
    parser.add_argument("--imgsz", type=int, help="Override inference image size.")
    parser.add_argument(
        "--count-mode",
        choices=["frame", "seen", "max", "line"],
        default="frame",
        help=(
            "frame counts current detections, seen counts tracked IDs, "
            "max shows the largest current-frame count, line counts crossings."
        ),
    )
    parser.add_argument(
        "--line-x",
        type=float,
        help="Vertical counting line x position. Use pixels, or 0-1 as a width ratio.",
    )
    parser.add_argument(
        "--line",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help=(
            "Custom counting line as two points in pixels. Direction is based on "
            "the ordered line: crossing from the right side to the left side is IN."
        ),
    )
    parser.add_argument(
        "--pick-line",
        action="store_true",
        help="Open a video frame and choose the counting line by clicking two points.",
    )
    parser.add_argument(
        "--pick-frame",
        type=int,
        default=0,
        help="Frame index to show when using --pick-line.",
    )
    parser.add_argument("--line-cooldown", type=int, default=12)
    parser.add_argument("--track-distance", type=float, default=120.0)
    parser.add_argument("--max-missing", type=int, default=30)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "runs" / "logs")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--allow-non-jetson",
        action="store_true",
        help="Allow this script to run outside NVIDIA Jetson.",
    )
    parser.add_argument("--list-models", action="store_true", help="Print configured models.")
    args = parser.parse_args()
    args._config = config
    return args


def main() -> None:
    args = parse_args()
    config = args._config

    if args.no_log:
        run_counter(args, config)
        return

    with tee_run_log("count", args.log_dir) as log_path:
        try:
            print_run_context("count_people.py", args)
            run_counter(args, config)
            print(f"\nRun completed. Log saved to: {log_path}")
        except BaseException:
            print("\nRun failed. Full traceback:")
            traceback.print_exc()
            print(f"\nLog saved to: {log_path}")
            raise


def run_counter(args: argparse.Namespace, config) -> None:
    if args.list_models:
        print(describe_models(config))
        return

    require_jetson(allow_non_jetson=args.allow_non_jetson)
    if args.pick_line and args.count_mode != "line":
        raise SystemExit("--pick-line is only useful with --count-mode line")

    model_config = config.models[args.model]
    weights = args.weights or model_config.weights
    imgsz = args.imgsz or model_config.imgsz
    conf = args.conf if args.conf is not None else model_config.conf
    line = tuple(args.line) if args.line else None
    if args.pick_line:
        line = pick_line_from_video(args.source, args.pick_frame)
        print(
            "Picked line: --line "
            f"{line[0]:.0f} {line[1]:.0f} {line[2]:.0f} {line[3]:.0f}"
        )

    from src.video_counter import run_people_counter

    result = run_people_counter(
        weights=weights,
        source=args.source,
        person_class_id=config.person_class_id,
        imgsz=imgsz,
        conf=conf,
        output_path=args.output,
        show=args.show,
        device=args.device,
        count_mode=args.count_mode,
        line=line,
        line_x=args.line_x,
        line_cooldown=args.line_cooldown,
        track_distance=args.track_distance,
        max_missing=args.max_missing,
    )

    print(f"Frames: {result.frames}")
    print(f"Time: {result.seconds:.1f}s")
    if args.count_mode == "line":
        print(f"IN: {result.in_count or 0}")
        print(f"OUT: {result.out_count or 0}")
        print(f"TOTAL: {result.people_count}")
    else:
        print(f"PEOPLE: {result.people_count}")
    if result.output_path:
        print(f"Saved: {result.output_path}")


def pick_line_from_video(source: str, frame_index: int) -> tuple[float, float, float, float]:
    import cv2

    video_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source for line picking: {source}")

    try:
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from source: {source}")

    window_name = "Pick counting line - click 2 points, Enter=accept, r=reset, q=cancel"
    points: list[tuple[int, int]] = []
    preview = frame.copy()

    def redraw() -> None:
        preview[:] = frame
        for point in points:
            cv2.circle(preview, point, 7, (0, 255, 255), -1)
        if len(points) == 2:
            cv2.line(preview, points[0], points[1], (0, 255, 255), 3)
            cv2.putText(
                preview,
                "Enter: accept | r: reset | q: cancel",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                preview,
                "Click 2 points to draw counting line",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 2:
                points.clear()
            points.append((x, y))
            redraw()

    redraw()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32) and len(points) == 2:
            break
        if key == ord("r"):
            points.clear()
            redraw()
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            raise SystemExit("Line picking cancelled.")

    cv2.destroyWindow(window_name)
    (x1, y1), (x2, y2) = points
    return (float(x1), float(y1), float(x2), float(y2))


if __name__ == "__main__":
    main()

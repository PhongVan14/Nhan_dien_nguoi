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

    model_config = config.models[args.model]
    weights = args.weights or model_config.weights
    imgsz = args.imgsz or model_config.imgsz
    conf = args.conf if args.conf is not None else model_config.conf

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
    )

    print(f"Frames: {result.frames}")
    print(f"Time: {result.seconds:.1f}s")
    print(f"PEOPLE: {result.people_count}")
    if result.output_path:
        print(f"Saved: {result.output_path}")


if __name__ == "__main__":
    main()

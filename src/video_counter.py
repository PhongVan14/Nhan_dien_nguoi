from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO


@dataclass(frozen=True)
class CounterRunResult:
    frames: int
    seconds: float
    people_count: int
    output_path: Path | None


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def run_people_counter(
    *,
    weights: str,
    source: str,
    person_class_id: int,
    imgsz: int,
    conf: float,
    output_path: Path | None = None,
    show: bool = False,
    device: str | None = None,
) -> CounterRunResult:
    model = YOLO(weights)
    capture = cv2.VideoCapture(parse_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = _make_writer(output_path, fps, width, height)
    frame_index = 0
    people_count = 0
    started_at = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            result = _detect_frame(
                model=model,
                frame=frame,
                conf=conf,
                imgsz=imgsz,
                person_class_id=person_class_id,
                device=device,
            )

            people_count = _draw_detections(frame, result)
            _draw_overlay(frame, people_count)

            if writer is not None:
                writer.write(frame)

            if show:
                cv2.imshow("People Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    seconds = time.perf_counter() - started_at
    return CounterRunResult(
        frames=frame_index,
        seconds=seconds,
        people_count=people_count,
        output_path=output_path,
    )


def _detect_frame(
    *,
    model: YOLO,
    frame,
    conf: float,
    imgsz: int,
    person_class_id: int,
    device: str | None,
):
    kwargs = {
        "conf": conf,
        "imgsz": imgsz,
        "classes": [person_class_id],
        "verbose": False,
    }
    if device:
        kwargs["device"] = device

    results = model.predict(frame, **kwargs)
    return results[0]


def _make_writer(
    output_path: Path | None,
    fps: float,
    width: int,
    height: int,
):
    if output_path is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def _draw_detections(frame, result) -> int:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0

    xyxy_values = boxes.xyxy.cpu().numpy()
    conf_values = boxes.conf.cpu().numpy()

    people_count = 0
    for xyxy, score in zip(xyxy_values, conf_values):
        x1, y1, x2, y2 = [int(value) for value in xyxy]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (22, 163, 74), 2)
        _put_text(frame, f"person {score:.2f}", (x1, max(22, y1 - 8)))
        people_count += 1

    return people_count


def _draw_overlay(frame, people_count: int) -> None:
    _draw_panel(frame, [f"PEOPLE: {people_count}"], (12, 12))


def _draw_panel(frame, lines: list[str], origin: tuple[int, int]) -> None:
    x, y = origin
    width = 150
    height = 28 + 26 * len(lines)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (230, 230, 230), 1)

    for index, line in enumerate(lines):
        _put_text(frame, line, (x + 12, y + 30 + index * 24), bg=None)


def _put_text(
    frame,
    text: str,
    position: tuple[int, int],
    bg: tuple[int, int, int] | None = (20, 20, 20),
) -> None:
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    if bg is not None:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(
            frame,
            (x - 4, y - text_height - baseline - 4),
            (x + text_width + 4, y + baseline + 4),
            bg,
            -1,
        )
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

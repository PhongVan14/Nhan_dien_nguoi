from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.line_counter import LineCrossingCounter, TrackPoint


LinePoints = tuple[float, float, float, float]


@dataclass(frozen=True)
class CounterRunResult:
    frames: int
    seconds: float
    people_count: int
    output_path: Path | None
    in_count: int | None = None
    out_count: int | None = None


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
    count_mode: str = "frame",
    line: LinePoints | None = None,
    line_x: float | None = None,
    line_cooldown: int = 12,
    track_distance: float = 120.0,
    max_missing: int = 30,
) -> CounterRunResult:
    model = YOLO(weights)
    capture = cv2.VideoCapture(parse_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = _make_writer(output_path, fps, width, height)
    line_counter = None
    if count_mode == "line" and width > 0:
        line_counter = LineCrossingCounter(
            _resolve_counting_line(line, line_x, width, height),
            cooldown_frames=line_cooldown,
            max_missing_frames=max_missing,
        )
    frame_index = 0
    people_count = 0
    max_people_count = 0
    seen_track_ids: set[int] = set()
    started_at = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            if count_mode in {"seen", "line"}:
                result = _track_frame(
                    model=model,
                    frame=frame,
                    conf=conf,
                    imgsz=imgsz,
                    person_class_id=person_class_id,
                    device=device,
                )
                tracks = _extract_tracks(result)
                for track_id, _box, _score in tracks:
                    seen_track_ids.add(track_id)

                _draw_tracks(frame, tracks)
                if count_mode == "line":
                    if line_counter is None:
                        line_counter = LineCrossingCounter(
                            _resolve_counting_line(
                                line,
                                line_x,
                                frame.shape[1],
                                frame.shape[0],
                            ),
                            cooldown_frames=line_cooldown,
                            max_missing_frames=max_missing,
                        )
                    line_counter.update(
                        _track_points_for_line(tracks),
                        frame_index,
                    )
                    people_count = line_counter.total
                    _draw_counting_line(frame, line_counter.line)
                    _draw_line_overlay(frame, line_counter)
                else:
                    people_count = len(seen_track_ids)
                    _draw_overlay(frame, people_count)
            else:
                result = _detect_frame(
                    model=model,
                    frame=frame,
                    conf=conf,
                    imgsz=imgsz,
                    person_class_id=person_class_id,
                    device=device,
                )
                detections = _extract_detections(result)
                current_count = len(detections)
                max_people_count = max(max_people_count, current_count)
                people_count = max_people_count if count_mode == "max" else current_count
                _draw_detections(frame, detections)
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
    in_count = line_counter.in_count if line_counter is not None else None
    out_count = line_counter.out_count if line_counter is not None else None
    return CounterRunResult(
        frames=frame_index,
        seconds=seconds,
        people_count=people_count,
        output_path=output_path,
        in_count=in_count,
        out_count=out_count,
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


def _track_frame(
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
        "persist": True,
        "verbose": False,
    }
    if device:
        kwargs["device"] = device

    results = model.track(frame, **kwargs)
    return results[0]


def _box_center(box: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_bottom_center(box: list[int]) -> tuple[float, float]:
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2.0, float(y2))


def _track_points_for_line(
    tracks: list[tuple[int, list[int], float]],
) -> list[TrackPoint]:
    return [
        TrackPoint(track_id=track_id, point=_box_bottom_center(box))
        for track_id, box, _score in tracks
    ]


class CentroidTracker:
    def __init__(self, max_distance: float = 120.0, max_missing: int = 30):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, detections: list[tuple[list[int], float]]) -> list[tuple[int, list[int], float]]:
        prepared = [
            {"box": box, "score": score, "center": _box_center(box)}
            for box, score in detections
        ]
        unmatched_detections = set(range(len(prepared)))
        unmatched_tracks = set(self.tracks.keys())
        pairs = []

        for track_id, track in self.tracks.items():
            tx, ty = track["center"]
            for detection_index, detection in enumerate(prepared):
                dx = tx - detection["center"][0]
                dy = ty - detection["center"][1]
                distance = (dx * dx + dy * dy) ** 0.5
                if distance <= self.max_distance:
                    pairs.append((distance, track_id, detection_index))

        for _distance, track_id, detection_index in sorted(pairs):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            self.tracks[track_id].update(prepared[detection_index])
            self.tracks[track_id]["missing"] = 0
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in list(unmatched_tracks):
            self.tracks[track_id]["missing"] += 1
            if self.tracks[track_id]["missing"] > self.max_missing:
                del self.tracks[track_id]

        for detection_index in sorted(unmatched_detections):
            self.tracks[self.next_id] = {
                **prepared[detection_index],
                "missing": 0,
            }
            self.next_id += 1

        return [
            (track_id, track["box"], track["score"])
            for track_id, track in self.tracks.items()
            if track["missing"] == 0
        ]


class LeftRightLineCounter:
    def __init__(self, line_x: float, cooldown_frames: int = 12, max_missing_frames: int = 30):
        self.line_x = float(line_x)
        self.cooldown_frames = cooldown_frames
        self.max_missing_frames = max_missing_frames
        self.in_count = 0
        self.out_count = 0
        self.states: dict[int, dict] = {}

    @property
    def total(self) -> int:
        return self.in_count + self.out_count

    def update(self, tracks: list[tuple[int, list[int], float]], frame_index: int) -> None:
        for track_id, box, _score in tracks:
            center_x, _center_y = _box_center(box)
            side = self._side(center_x)
            if side == 0:
                continue

            state = self.states.setdefault(
                track_id,
                {"side": 0, "last_frame": frame_index, "last_count_frame": -100000},
            )
            state["last_frame"] = frame_index

            previous_side = state["side"]
            if previous_side == 0:
                state["side"] = side
                continue

            if previous_side != side:
                if frame_index - state["last_count_frame"] >= self.cooldown_frames:
                    if previous_side < side:
                        self.in_count += 1
                    else:
                        self.out_count += 1
                    state["last_count_frame"] = frame_index
                state["side"] = side

        stale_ids = [
            track_id
            for track_id, state in self.states.items()
            if frame_index - state["last_frame"] > self.max_missing_frames
        ]
        for track_id in stale_ids:
            del self.states[track_id]

    def _side(self, center_x: float) -> int:
        if center_x < self.line_x:
            return -1
        if center_x > self.line_x:
            return 1
        return 0


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


def _extract_detections(result) -> list[tuple[list[int], float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_values = boxes.xyxy.cpu().numpy()
    conf_values = boxes.conf.cpu().numpy()

    detections = []
    for xyxy, score in zip(xyxy_values, conf_values):
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        detections.append(([x1, y1, x2, y2], float(score)))
    return detections


def _extract_tracks(result) -> list[tuple[int, list[int], float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return []

    xyxy_values = boxes.xyxy.cpu().numpy()
    conf_values = boxes.conf.cpu().numpy()
    id_values = boxes.id.cpu().numpy()

    tracks = []
    for track_id, xyxy, score in zip(id_values, xyxy_values, conf_values):
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        tracks.append((int(track_id), [x1, y1, x2, y2], float(score)))
    return tracks


def _draw_detections(frame, detections: list[tuple[list[int], float]]) -> int:
    people_count = 0
    for box, score in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (22, 163, 74), 2)
        _put_text(frame, f"person {score:.2f}", (x1, max(22, y1 - 8)))
        people_count += 1

    return people_count


def _draw_tracks(frame, tracks: list[tuple[int, list[int], float]]) -> None:
    for track_id, box, score in tracks:
        x1, y1, x2, y2 = box
        center_x, center_y = _box_center(box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (22, 163, 74), 2)
        cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 255, 255), -1)
        _put_text(frame, f"person #{track_id} {score:.2f}", (x1, max(22, y1 - 8)))


def _draw_overlay(frame, people_count: int) -> None:
    _draw_panel(frame, [f"PEOPLE: {people_count}"], (12, 12))


def _draw_line_overlay(frame, counter: LineCrossingCounter) -> None:
    _draw_panel(
        frame,
        [f"IN: {counter.in_count}", f"OUT: {counter.out_count}", f"TOTAL: {counter.total}"],
        (12, 12),
    )


def _draw_counting_line(
    frame,
    line: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    height, width = frame.shape[:2]
    (x1, y1), (x2, y2) = line
    p1 = _clamp_point(x1, y1, width, height)
    p2 = _clamp_point(x2, y2, width, height)
    cv2.line(frame, p1, p2, (0, 255, 255), 3)

    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    normal_x = -dy / length
    normal_y = dx / length
    in_position = _clamp_point(mid_x + normal_x * 48, mid_y + normal_y * 48, width, height)
    out_position = _clamp_point(mid_x - normal_x * 48, mid_y - normal_y * 48, width, height)
    _put_text(frame, "IN", in_position, bg=(15, 55, 55))
    _put_text(frame, "OUT", out_position, bg=(55, 55, 15))


def _resolve_line_x(value: float | None, frame_width: int) -> float:
    if value is None:
        return frame_width / 2.0
    if 0.0 < value < 1.0:
        return frame_width * value
    return value


def _resolve_counting_line(
    line: LinePoints | None,
    line_x: float | None,
    frame_width: int,
    frame_height: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if line is not None:
        x1, y1, x2, y2 = line
        return ((x1, y1), (x2, y2))

    x = _resolve_line_x(line_x, frame_width)
    return ((x, 0.0), (x, float(frame_height - 1)))


def _clamp_point(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    clamped_x = max(0, min(frame_width - 1, int(round(x))))
    clamped_y = max(0, min(frame_height - 1, int(round(y))))
    return (clamped_x, clamped_y)


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

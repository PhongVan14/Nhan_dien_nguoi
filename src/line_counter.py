from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Point = tuple[float, float]
Line = tuple[Point, Point]


@dataclass(frozen=True)
class TrackPoint:
    track_id: int
    point: Point


@dataclass(frozen=True)
class CountEvent:
    track_id: int
    direction: str
    frame_index: int
    point: Point


@dataclass
class _TrackState:
    side: int = 0
    last_frame: int = 0
    last_count_frame: int = -10_000


class LineCrossingCounter:
    """Counts tracked objects when their point crosses a virtual line."""

    def __init__(
        self,
        line: Line,
        cooldown_frames: int = 12,
        max_missing_frames: int = 90,
    ) -> None:
        self.line = line
        self.cooldown_frames = cooldown_frames
        self.max_missing_frames = max_missing_frames
        self.in_count = 0
        self.out_count = 0
        self._states: dict[int, _TrackState] = {}

    @property
    def total(self) -> int:
        return self.in_count + self.out_count

    @property
    def active_tracks(self) -> int:
        return len(self._states)

    def update(
        self,
        points: Iterable[TrackPoint],
        frame_index: int,
    ) -> list[CountEvent]:
        events: list[CountEvent] = []

        for track_point in points:
            side = self._side(track_point.point)
            if side == 0:
                continue

            state = self._states.setdefault(track_point.track_id, _TrackState())
            state.last_frame = frame_index

            if state.side == 0:
                state.side = side
                continue

            if state.side != side:
                if frame_index - state.last_count_frame >= self.cooldown_frames:
                    direction = "in" if state.side < side else "out"
                    self._add_count(direction)
                    state.last_count_frame = frame_index
                    events.append(
                        CountEvent(
                            track_id=track_point.track_id,
                            direction=direction,
                            frame_index=frame_index,
                            point=track_point.point,
                        )
                    )
                state.side = side

        self._drop_missing_tracks(frame_index)
        return events

    def _add_count(self, direction: str) -> None:
        if direction == "in":
            self.in_count += 1
        else:
            self.out_count += 1

    def _side(self, point: Point) -> int:
        (x1, y1), (x2, y2) = self.line
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def _drop_missing_tracks(self, frame_index: int) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if frame_index - state.last_frame > self.max_missing_frames
        ]
        for track_id in stale_ids:
            del self._states[track_id]

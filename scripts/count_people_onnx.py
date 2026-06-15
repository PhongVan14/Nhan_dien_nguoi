import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a YOLO ONNX people detector without Ultralytics."
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to .onnx model.")
    parser.add_argument("--source", default="0", help="Video path, stream URL, or webcam id.")
    parser.add_argument("--output", type=Path, help="Optional output video path.")
    parser.add_argument("--show", action="store_true", help="Show OpenCV preview window.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--count-mode",
        choices=["seen", "frame", "max"],
        default="seen",
        help=(
            "seen counts each new tracked person once, frame counts current detections, "
            "max shows the largest current-frame count seen so far."
        ),
    )
    parser.add_argument("--track-distance", type=float, default=120.0)
    parser.add_argument("--max-missing", type=int, default=30)
    parser.add_argument(
        "--min-track-hits",
        type=int,
        default=2,
        help="Count a new person only after the track is detected this many times.",
    )
    parser.add_argument(
        "--max-match-cost",
        type=float,
        default=0.85,
        help="Maximum DeepSORT-style matching cost. Lower is stricter.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class id to keep for multi-class models. COCO person is 0. Use -1 to keep all.",
    )
    parser.add_argument(
        "--backend",
        choices=["opencv", "cuda"],
        default="opencv",
        help="OpenCV DNN backend. Use cuda only if your OpenCV build supports it.",
    )
    return parser.parse_args()


def parse_source(source):
    return int(source) if source.isdigit() else source


def box_center(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def box_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def extract_appearance(frame, box):
    x, y, w, h = box
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(x)))
    y1 = max(0, min(height - 1, int(y)))
    x2 = max(0, min(width, int(x + w)))
    y2 = max(0, min(height, int(y + h)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((256,), dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    feature = hist.reshape(-1).astype(np.float32)
    norm = np.linalg.norm(feature)
    if norm > 0:
        feature = feature / norm
    return feature


def appearance_similarity(feature_a, feature_b):
    if feature_a is None or feature_b is None:
        return 0.0
    similarity = float(np.dot(feature_a, feature_b))
    return max(0.0, min(1.0, similarity))


class DeepSortLiteTracker:
    def __init__(self, max_distance=120.0, max_missing=30, min_hits=2, max_cost=0.85):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.min_hits = max(1, min_hits)
        self.max_cost = max_cost
        self.next_id = 1
        self.tracks = {}
        self.total_seen = 0

    def update(self, frame, detections):
        prepared = []
        for box, score in detections:
            center = box_center(box)
            feature = extract_appearance(frame, box)
            prepared.append(
                {
                    "center": center,
                    "box": box,
                    "score": score,
                    "feature": feature,
                }
            )

        unmatched_detections = set(range(len(prepared)))
        unmatched_tracks = set(self.tracks.keys())
        pairs = []
        for track_id, track in self.tracks.items():
            predicted = self._predict_center(track)
            for det_index, detection in enumerate(prepared):
                cost = self._match_cost(track, predicted, detection)
                if cost <= self.max_cost:
                    pairs.append((cost, track_id, det_index))

        for cost, track_id, det_index in sorted(pairs):
            if track_id not in unmatched_tracks or det_index not in unmatched_detections:
                continue
            self._update_track(track_id, prepared[det_index])
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(det_index)

        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track["missing"] += 1
            if track["missing"] > self.max_missing:
                del self.tracks[track_id]

        for det_index in sorted(unmatched_detections):
            self._new_track(prepared[det_index])
            self.next_id += 1

        active = []
        for track_id, track in self.tracks.items():
            if track["missing"] == 0:
                active.append((track_id, track["box"], track["score"]))
        return active

    def _predict_center(self, track):
        center = track["center"]
        velocity = track["velocity"]
        return (center[0] + velocity[0], center[1] + velocity[1])

    def _match_cost(self, track, predicted_center, detection):
        detection_center = detection["center"]
        dx = predicted_center[0] - detection_center[0]
        dy = predicted_center[1] - detection_center[1]
        distance = (dx * dx + dy * dy) ** 0.5
        iou = box_iou(track["box"], detection["box"])
        appearance = appearance_similarity(track["feature"], detection["feature"])

        if distance > self.max_distance and iou < 0.05 and appearance < 0.65:
            return 999.0

        motion_cost = min(distance / max(1.0, self.max_distance), 1.5)
        iou_cost = 1.0 - iou
        appearance_cost = 1.0 - appearance
        return 0.35 * motion_cost + 0.35 * iou_cost + 0.30 * appearance_cost

    def _update_track(self, track_id, detection):
        track = self.tracks[track_id]
        old_center = track["center"]
        new_center = detection["center"]
        old_velocity = track["velocity"]
        measured_velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1],
        )
        track["velocity"] = (
            0.6 * old_velocity[0] + 0.4 * measured_velocity[0],
            0.6 * old_velocity[1] + 0.4 * measured_velocity[1],
        )
        track["center"] = new_center
        track["box"] = detection["box"]
        track["score"] = detection["score"]
        track["feature"] = 0.8 * track["feature"] + 0.2 * detection["feature"]
        norm = np.linalg.norm(track["feature"])
        if norm > 0:
            track["feature"] = track["feature"] / norm
        track["missing"] = 0
        track["hits"] += 1
        if not track["counted"] and track["hits"] >= self.min_hits:
            track["counted"] = True
            self.total_seen += 1

    def _new_track(self, detection):
        counted = self.min_hits <= 1
        self.tracks[self.next_id] = {
            "center": detection["center"],
            "velocity": (0.0, 0.0),
            "box": detection["box"],
            "score": detection["score"],
            "feature": detection["feature"],
            "missing": 0,
            "hits": 1,
            "counted": counted,
        }
        if counted:
            self.total_seen += 1


def letterbox(frame, imgsz):
    height, width = frame.shape[:2]
    scale = min(float(imgsz) / float(width), float(imgsz) / float(height))
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_x = (imgsz - new_width) // 2
    pad_y = (imgsz - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def make_net(weights, backend):
    net = cv2.dnn.readNetFromONNX(str(weights))
    if backend == "cuda":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def detect_people(net, frame, imgsz, conf_threshold, iou_threshold, class_id):
    input_image, scale, pad_x, pad_y = letterbox(frame, imgsz)
    blob = cv2.dnn.blobFromImage(
        input_image, 1.0 / 255.0, (imgsz, imgsz), swapRB=True, crop=False
    )
    net.setInput(blob)
    output = net.forward()

    boxes = []
    scores = []
    predictions = np.squeeze(output)
    if predictions.ndim == 1:
        predictions = np.expand_dims(predictions, axis=0)
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    height, width = frame.shape[:2]
    for row in predictions:
        if row.shape[0] < 5:
            continue

        if row.shape[0] == 5:
            score = float(row[4])
        else:
            class_scores = row[4:]
            predicted_class = int(np.argmax(class_scores))
            if class_id >= 0 and predicted_class != class_id:
                continue
            score = float(class_scores[predicted_class])

        if score < conf_threshold:
            continue

        cx, cy, box_w, box_h = row[:4]
        x1 = (float(cx) - float(box_w) / 2.0 - pad_x) / scale
        y1 = (float(cy) - float(box_h) / 2.0 - pad_y) / scale
        x2 = (float(cx) + float(box_w) / 2.0 - pad_x) / scale
        y2 = (float(cy) + float(box_h) / 2.0 - pad_y) / scale

        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(0, min(width - 1, int(round(x2))))
        y2 = max(0, min(height - 1, int(round(y2))))
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        if box_width == 0 or box_height == 0:
            continue

        boxes.append([x1, y1, box_width, box_height])
        scores.append(score)

    indexes = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    if len(indexes) == 0:
        return []

    indexes = np.array(indexes).reshape(-1)
    return [(boxes[index], scores[index]) for index in indexes]


def draw_text(frame, text, position, bg=(20, 20, 20)):
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


def draw_people_count(frame, people_count):
    draw_text(frame, "PEOPLE: {}".format(people_count), (20, 36))


def draw_detections(frame, detections):
    for box, score in detections:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (22, 163, 74), 2)
        draw_text(frame, "person {:.2f}".format(score), (x, max(22, y - 8)))


def draw_tracks(frame, tracks):
    for track_id, box, score in tracks:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (22, 163, 74), 2)
        draw_text(
            frame,
            "person #{} {:.2f}".format(track_id, score),
            (x, max(22, y - 8)),
        )


def make_writer(output_path, fps, width, height):
    if output_path is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def main():
    args = parse_args()
    net = make_net(args.weights, args.backend)
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise SystemExit("Cannot open source: {}".format(args.source))

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = make_writer(args.output, fps, width, height)
    tracker = DeepSortLiteTracker(
        args.track_distance,
        args.max_missing,
        args.min_track_hits,
        args.max_match_cost,
    )

    frame_index = 0
    people_count = 0
    max_people_count = 0
    started_at = time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            detections = detect_people(
                net, frame, args.imgsz, args.conf, args.iou, args.class_id
            )
            if args.count_mode == "seen":
                tracks = tracker.update(frame, detections)
                people_count = tracker.total_seen
                draw_tracks(frame, tracks)
            else:
                current_count = len(detections)
                max_people_count = max(max_people_count, current_count)
                people_count = (
                    max_people_count if args.count_mode == "max" else current_count
                )
                draw_detections(frame, detections)

            draw_people_count(frame, people_count)

            if writer is not None:
                writer.write(frame)
            if args.show:
                cv2.imshow("People Counter ONNX", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - started_at
    print("Frames: {}".format(frame_index))
    print("Time: {:.1f}s".format(elapsed))
    print("PEOPLE: {}".format(people_count))
    if args.output:
        print("Saved: {}".format(args.output))


if __name__ == "__main__":
    main()

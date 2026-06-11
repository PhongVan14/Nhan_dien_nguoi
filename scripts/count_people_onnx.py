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


class CentroidTracker:
    def __init__(self, max_distance=120.0, max_missing=30):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks = {}
        self.total_seen = 0

    def update(self, detections):
        prepared = []
        for box, score in detections:
            x, y, w, h = box
            center = (x + w / 2.0, y + h / 2.0)
            prepared.append((center, box, score))

        unmatched_detections = set(range(len(prepared)))
        unmatched_tracks = set(self.tracks.keys())
        pairs = []
        for track_id, track in self.tracks.items():
            tx, ty = track["center"]
            for det_index, (center, _, _) in enumerate(prepared):
                dx = tx - center[0]
                dy = ty - center[1]
                pairs.append((dx * dx + dy * dy, track_id, det_index))

        for distance_sq, track_id, det_index in sorted(pairs):
            if track_id not in unmatched_tracks or det_index not in unmatched_detections:
                continue
            if distance_sq > self.max_distance * self.max_distance:
                continue
            center, box, score = prepared[det_index]
            self.tracks[track_id] = {
                "center": center,
                "box": box,
                "score": score,
                "missing": 0,
            }
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(det_index)

        for track_id in list(unmatched_tracks):
            track = self.tracks[track_id]
            track["missing"] += 1
            if track["missing"] > self.max_missing:
                del self.tracks[track_id]

        for det_index in sorted(unmatched_detections):
            center, box, score = prepared[det_index]
            self.tracks[self.next_id] = {
                "center": center,
                "box": box,
                "score": score,
                "missing": 0,
            }
            self.total_seen += 1
            self.next_id += 1

        active = []
        for track_id, track in self.tracks.items():
            if track["missing"] == 0:
                active.append((track_id, track["box"], track["score"]))
        return active


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
    tracker = CentroidTracker(args.track_distance, args.max_missing)

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
                tracks = tracker.update(detections)
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

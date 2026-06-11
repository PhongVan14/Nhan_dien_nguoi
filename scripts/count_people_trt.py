import argparse
import ctypes
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a TensorRT people counter without PyCUDA."
    )
    parser.add_argument("--engine", type=Path, required=True, help="Path to .engine file.")
    parser.add_argument("--source", default="0", help="Video path, stream URL, or webcam id.")
    parser.add_argument("--output", type=Path, help="Optional output video path.")
    parser.add_argument("--show", action="store_true")
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
    return parser.parse_args()


def parse_source(source):
    return int(source) if source.isdigit() else source


class CudaRuntime:
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []
        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int

    def check(self, code, action):
        if code != 0:
            raise RuntimeError("CUDA {} failed with code {}".format(action, code))

    def malloc(self, nbytes):
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), nbytes), "malloc")
        return pointer

    def free(self, pointer):
        if pointer:
            self.check(self.lib.cudaFree(pointer), "free")

    def memcpy_htod(self, device_pointer, host_array):
        host = host_array.ctypes.data_as(ctypes.c_void_p)
        self.check(
            self.lib.cudaMemcpy(
                device_pointer,
                host,
                host_array.nbytes,
                self.MEMCPY_HOST_TO_DEVICE,
            ),
            "memcpy host to device",
        )

    def memcpy_dtoh(self, host_array, device_pointer):
        host = host_array.ctypes.data_as(ctypes.c_void_p)
        self.check(
            self.lib.cudaMemcpy(
                host,
                device_pointer,
                host_array.nbytes,
                self.MEMCPY_DEVICE_TO_HOST,
            ),
            "memcpy device to host",
        )

    def synchronize(self):
        self.check(self.lib.cudaDeviceSynchronize(), "synchronize")


class TensorRTRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(str(engine_path), "rb") as engine_file:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError("Could not deserialize engine: {}".format(engine_path))

        self.context = self.engine.create_execution_context()
        self.cuda = CudaRuntime()
        self.bindings = [0] * self.engine.num_bindings
        self.host_inputs = {}
        self.host_outputs = {}
        self.device_buffers = {}
        self.input_binding = None
        self.output_bindings = []
        self._allocate_buffers()

    def _binding_shape(self, binding_index):
        shape = tuple(self.engine.get_binding_shape(binding_index))
        if any(dim < 0 for dim in shape):
            shape = tuple(self.context.get_binding_shape(binding_index))
        return shape

    def _allocate_buffers(self):
        for index in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = self._binding_shape(index)
            size = int(trt.volume(shape))
            host = np.empty(size, dtype=dtype)
            device = self.cuda.malloc(host.nbytes)
            self.bindings[index] = int(device.value)
            self.device_buffers[index] = device

            if self.engine.binding_is_input(index):
                self.input_binding = index
                self.host_inputs[index] = host
            else:
                self.output_bindings.append(index)
                self.host_outputs[index] = host

        if self.input_binding is None or not self.output_bindings:
            raise RuntimeError("Engine must have at least one input and one output.")

    def infer(self, input_tensor):
        input_binding = self.input_binding
        host_input = self.host_inputs[input_binding]
        flat_input = np.ascontiguousarray(input_tensor).reshape(-1)
        if flat_input.size != host_input.size:
            raise RuntimeError(
                "Input size mismatch: expected {}, got {}".format(
                    host_input.size, flat_input.size
                )
            )
        np.copyto(host_input, flat_input.astype(host_input.dtype, copy=False))
        self.cuda.memcpy_htod(self.device_buffers[input_binding], host_input)

        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        self.cuda.synchronize()

        outputs = []
        for binding in self.output_bindings:
            host_output = self.host_outputs[binding]
            self.cuda.memcpy_dtoh(host_output, self.device_buffers[binding])
            outputs.append(host_output.copy().reshape(self._binding_shape(binding)))
        return outputs

    def close(self):
        for pointer in self.device_buffers.values():
            self.cuda.free(pointer)
        self.device_buffers = {}


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


def preprocess(frame, imgsz):
    image, scale, pad_x, pad_y = letterbox(frame, imgsz)
    image = image.astype(np.float32) / 255.0
    image = image[:, :, ::-1]
    tensor = np.transpose(image, (2, 0, 1))[None, :, :, :]
    return np.ascontiguousarray(tensor), scale, pad_x, pad_y


def detect_people(runner, frame, imgsz, conf_threshold, iou_threshold, class_id):
    input_tensor, scale, pad_x, pad_y = preprocess(frame, imgsz)
    output = runner.infer(input_tensor)[0]
    predictions = np.squeeze(output)
    if predictions.ndim == 1:
        predictions = np.expand_dims(predictions, axis=0)
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    height, width = frame.shape[:2]
    boxes = []
    scores = []
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
    runner = TensorRTRunner(args.engine)
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
                runner, frame, args.imgsz, args.conf, args.iou, args.class_id
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
                cv2.imshow("People Counter TensorRT", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        runner.close()
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

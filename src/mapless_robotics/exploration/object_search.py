"""OWLv2 detections and depth-backed confirmation of an object goal."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace

import numpy as np

from .goal_query import CACHE_DIR


DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"


@dataclass(frozen=True)
class ObjectDetection:
    box: np.ndarray  # Original image pixels: xmin, ymin, xmax, ymax.
    score: float
    label: str


@dataclass(frozen=True)
class ObjectTarget:
    position: np.ndarray  # World XYZ on the observed object surface.
    score: float
    timestamp: float
    label: str


def filter_detections(boxes, scores, labels, width, height, threshold, limit=20):
    """Clip boxes and suppress duplicates, including across query labels."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=str)
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError("Each detection box needs a score and label")
    finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    valid = (finite & (scores >= threshold) & (boxes[:, 2] > boxes[:, 0])
             & (boxes[:, 3] > boxes[:, 1]))
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    kept = []
    for index in np.argsort(-scores, kind="stable"):
        box = boxes[index]
        duplicate = False
        for detection in kept:
            overlap = np.maximum(0, np.minimum(box[2:], detection.box[2:])
                                 - np.maximum(box[:2], detection.box[:2]))
            intersection = float(np.prod(overlap))
            union = (np.prod(box[2:] - box[:2])
                     + np.prod(detection.box[2:] - detection.box[:2]) - intersection)
            if intersection / union > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(ObjectDetection(box.copy(), float(scores[index]), str(labels[index])))
        if len(kept) >= limit:
            break
    return tuple(kept)


class OWLv2Detector:
    def __init__(self, labels, model=DEFAULT_MODEL, device="auto", threshold=0.2,
                 image_size=None):
        if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("OWLv2 requires nonempty object labels")
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("--object-threshold must be between 0 and 1")
        if image_size is not None and (image_size < 32 or image_size % 16):
            raise ValueError("--object-image-size must be a multiple of 16, at least 32")
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("--object-device cuda requires an available CUDA device")
        self.device = device
        self.labels = tuple(labels)
        self.threshold = float(threshold)
        self.image_size = image_size
        self.texts = [[f"a photo of {label}" for label in self.labels]]
        self.processor = Owlv2Processor.from_pretrained(model, cache_dir=CACHE_DIR)
        self.model = Owlv2ForObjectDetection.from_pretrained(model, cache_dir=CACHE_DIR)
        self.model.to(device=device,
                      dtype=torch.float16 if device == "cuda" else torch.float32).eval()
        # Keep object inference off the default stream used by OpenFrontier.
        # Wait once for weights uploaded during initialization, before inference.
        self._stream = torch.cuda.Stream(device=device) if device == "cuda" else None
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream(device))

    def __call__(self, rgb):
        import torch
        from PIL import Image

        height, width = rgb.shape[:2]
        options = ({"size": {"height": self.image_size, "width": self.image_size}}
                   if self.image_size is not None else {})
        inputs = self.processor(images=Image.fromarray(rgb), text=self.texts,
                                return_tensors="pt", **options)
        stream = torch.cuda.stream(self._stream) if self._stream is not None else nullcontext()
        with torch.inference_mode(), stream:
            inputs = inputs.to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
            outputs = self.model(**inputs, interpolate_pos_encoding=self.image_size is not None)
            result = self.processor.post_process_grounded_object_detection(
                outputs, threshold=self.threshold, target_sizes=[(height, width)])[0]
            labels = [self.labels[int(index)] for index in result["labels"].detach().cpu().tolist()]
            return filter_detections(result["boxes"].detach().cpu().numpy(),
                                     result["scores"].detach().cpu().numpy(), labels,
                                     width, height, self.threshold)


class ObjectSearchWorker:
    """One OWLv2 capture in flight, independent of frontier analysis and control."""

    def __init__(self, detector):
        self.detector = detector
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="object-search")
        self.future = None
        self.frame_index = None
        self.discard_pending = False

    def submit(self, frame_index, analysis):
        if self.future is not None or not analysis.frontier_analysis:
            return False
        self.frame_index = frame_index
        self.discard_pending = False
        self.future = self.executor.submit(self._process, analysis)
        return True

    def _process(self, analysis):
        detections = self.detector(analysis.observation.rgb)
        return replace(analysis, object_detections=tuple(detections), object_analysis=True)

    def poll(self):
        if self.future is None or not self.future.done():
            return None, None
        future, frame_index = self.future, self.frame_index
        self.future = self.frame_index = None
        if self.discard_pending:
            return None, None
        return frame_index, future.result()

    def discard(self):
        """Ignore an old-view result after recovery, movement, or rejection."""
        self.discard_pending = True

    def close(self):
        if self.future is not None:
            self.future.cancel()
        self.executor.shutdown(wait=True)
        self.future = None


def detection_position(detection, depth, camera_pose, intrinsics, max_depth=5.0):
    """Project median central box depth using Habitat's camera axes."""
    if depth is None:
        return None
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape
    box = np.asarray(detection.box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    center = (box[:2] + box[2:]) / 2
    half = (box[2:] - box[:2]) * 0.15
    lo = np.maximum(np.floor(center - half).astype(int), 0)
    hi = np.minimum(np.ceil(center + half).astype(int), [width, height])
    if np.any(hi <= lo):
        return None
    patch = depth[lo[1]:hi[1], lo[0]:hi[0]]
    valid = np.isfinite(patch) & (patch > 0) & (patch <= max_depth)
    if np.count_nonzero(valid) < max(5, patch.size / 2):
        return None
    distance = float(np.median(patch[valid]))
    u, v = center
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    local = np.array([(u - cx) * distance / fx, -(v - cy) * distance / fy, -distance])
    return camera_pose[:3, :3] @ local + camera_pose[:3, 3]


def approach_position(target_position, robot_pose, stand_off=1.0):
    """Ground-plane approach point on the observer's side of the object."""
    position = np.asarray(robot_pose[:3, 3], dtype=float)
    delta = (np.asarray(target_position) - position)[[0, 2]]
    distance = float(np.linalg.norm(delta))
    approach = position.copy()
    if distance > stand_off:
        approach[[0, 2]] += delta * ((distance - stand_off) / distance)
    return approach


class ObjectSearch:
    """Confirm one spatially consistent object across distinct captures."""

    def __init__(self, query, confirmations=3, consistency_radius=0.75):
        if not query.strip():
            raise ValueError("--object-query must contain a request")
        if confirmations < 1 or not np.isfinite(consistency_radius) or consistency_radius <= 0:
            raise ValueError("Object confirmations and consistency radius must be positive")
        self.query = query.strip()
        self.labels = ()
        self.required_confirmations = confirmations
        self.radius = float(consistency_radius)
        self.target = self.candidate = None
        self.count = 0
        self.last_capture = self.last_seen = -float("inf")
        self.rejected = []
        self.status = "Searching"

    def reset_candidate(self):
        self.candidate = None
        self.count = 0

    def update(self, result, intrinsics):
        timestamp = result.observation.timestamp
        if timestamp <= self.last_capture:
            return self.target
        self.last_capture = timestamp
        located = []
        for detection in result.object_detections:
            position = detection_position(detection, result.depth,
                                          result.observation.camera_pose, intrinsics)
            if position is not None and not any(
                    np.linalg.norm(position - previous) <= self.radius for previous in self.rejected):
                located.append((detection, position))
        reference = self.target.position if self.target is not None else self.candidate
        matches = [(detection, position) for detection, position in located
                   if reference is not None and np.linalg.norm(position - reference) <= self.radius]
        if self.target is not None:
            if matches:
                self.last_seen = timestamp
            return self.target
        if not located:
            self.reset_candidate()
            self.status = "Searching"
            return None
        detection, position = max(matches or located, key=lambda item: item[0].score)
        if matches:
            self.candidate = (self.candidate * self.count + position) / (self.count + 1)
            self.count += 1
        else:
            self.candidate = position.copy()
            self.count = 1
        self.status = f"Confirming {self.count}/{self.required_confirmations}"
        if self.count >= self.required_confirmations:
            self.target = ObjectTarget(self.candidate.copy(), detection.score, timestamp, detection.label)
            self.last_seen = timestamp
            self.status = "Confirmed; approaching"
        return self.target

    def reject(self):
        if self.target is not None:
            self.rejected.append(self.target.position.copy())
        self.target = None
        self.reset_candidate()
        self.last_seen = -float("inf")
        self.status = "Searching"

    def to_dict(self):
        return {"query": self.query, "labels": list(self.labels), "status": self.status,
                "required_confirmations": self.required_confirmations, "confirmations": self.count,
                "target": None if self.target is None else {
                    "position": self.target.position.tolist(), "score": self.target.score,
                    "timestamp": self.target.timestamp, "label": self.target.label},
                "rejected_positions": [position.tolist() for position in self.rejected]}

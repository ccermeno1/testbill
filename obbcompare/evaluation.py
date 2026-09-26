"""Evaluation on a YOLOv8-OBB dataset: loading, image transforms and rotated-box metrics.

The AP is the one of the three training repos (port of mmrotate ``eval_rbbox_map``, VOC
"area" mode): per image the detections are sorted by score and each one is matched to the
ground truth box it overlaps most; it is a TP if that IoU >= thr and the box was not taken
yet. mAP@.5:.95 is the mean over IoU 0.50, 0.55, ..., 0.95. Same numbers as the repos'
``evaluate.py`` for the same detections.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .detector import Detections, rotated_iou

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}
IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)


# ---------------------------------------------------------------------- dataset
@dataclass
class Sample:
    image_path: Path
    polys: np.ndarray    # (k, 4, 2) ground truth corners, in pixels of the image as loaded
    labels: np.ndarray   # (k,)


def poly_to_rbox(polys: np.ndarray) -> np.ndarray:
    """(k, 4, 2) -> (k, 5) cx, cy, w, h, angle_rad via the minimum-area rectangle."""
    out = np.zeros((len(polys), 5), np.float32)
    for i, p in enumerate(polys.astype(np.float32)):
        (cx, cy), (w, h), a = cv2.minAreaRect(p)
        out[i] = cx, cy, w, h, np.radians(a)
    return out


def load_image(path: Path) -> np.ndarray:
    """BGR uint8 with the EXIF orientation applied (HEIC through pillow-heif)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def load_yolo_obb(root: Path, split: str | None = None) -> list[Sample]:
    """Roboflow YOLOv8-OBB export: ``<split>/images`` + ``<split>/labels`` with
    ``cls x1 y1 ... x4 y4`` normalised to the image size. Without ``split`` every split
    folder found is used."""
    root = Path(root)
    splits = [split] if split else [d.name for d in sorted(root.iterdir()) if (d / "images").is_dir()]
    samples = []
    for s in splits:
        for img_path in sorted((root / s / "images").iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            with Image.open(img_path) as im:
                w, h = ImageOps.exif_transpose(im).size
            label_path = root / s / "labels" / (img_path.stem + ".txt")
            rows = []
            if label_path.exists():
                rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
            labels = np.array([int(r[0]) for r in rows], np.int64)
            polys = np.array([[float(v) for v in r[1:9]] for r in rows], np.float32).reshape(-1, 4, 2)
            polys *= np.array([w, h], np.float32)
            samples.append(Sample(img_path, polys, labels))
    return samples


# ---------------------------------------------------------------------- image transforms (studies)
def resize_long_side(img: np.ndarray, polys: np.ndarray, long_side: int):
    """Downscale the photo so its long side is ``long_side`` (simulates a lower-resolution
    camera or a smaller image); the model then scales it to its own input size."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img, polys
    out = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return out, polys * s


def rotate(img: np.ndarray, polys: np.ndarray, angle_deg: float, fill=(114, 114, 114)):
    """Rotate counter-clockwise by ``angle_deg`` around the centre, enlarging the canvas so
    nothing is cropped. The ground-truth corners are rotated with the same matrix."""
    if angle_deg % 360 == 0:
        return img, polys
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    flags = cv2.INTER_LINEAR if angle_deg % 90 else cv2.INTER_NEAREST
    out = cv2.warpAffine(img, m, (nw, nh), flags=flags, borderValue=fill)
    pts = polys.reshape(-1, 2) @ m[:, :2].T + m[:, 2]
    return out, pts.reshape(-1, 4, 2).astype(np.float32)


def downscale_for_study(img: np.ndarray, polys: np.ndarray, long_side: int = 1600):
    """The models work at <= 1024 px: shrinking huge phone photos first makes the studies
    much faster without changing what the network sees in any meaningful way."""
    return resize_long_side(img, polys, long_side)


# ---------------------------------------------------------------------- matching and AP
@dataclass
class ImageResult:
    """Detections of one image (score >= the low AP threshold) and its ground truth."""
    boxes: np.ndarray    # (n, 5)
    scores: np.ndarray   # (n,)
    labels: np.ndarray   # (n,)
    gt: np.ndarray       # (k, 5)
    gt_labels: np.ndarray
    ious: np.ndarray | None = None  # (n, k) cached

    @classmethod
    def from_detections(cls, dets: Detections, gt_polys: np.ndarray, gt_labels: np.ndarray) -> "ImageResult":
        order = np.argsort(-dets.scores, kind="stable")
        r = cls(dets.boxes[order], dets.scores[order], dets.labels[order], poly_to_rbox(gt_polys), gt_labels)
        r.ious = iou_matrix(r.boxes, r.gt)
        return r


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((len(a), len(b)), np.float32)
    if len(b):
        for i, box in enumerate(a):
            out[i] = rotated_iou(box, b)
    return out


def match(r: ImageResult, iou_thr: float, score_thr: float = 0.0, cls: int | None = None):
    """mmrotate matching. Returns tp (n,) bool over the kept detections, the matched gt index
    per detection (-1 = FP) and the number of gt boxes of the class."""
    keep = r.scores >= score_thr
    gt_mask = np.ones(len(r.gt), bool) if cls is None else r.gt_labels == cls
    if cls is not None:
        keep &= r.labels == cls
    idx = np.nonzero(keep)[0]
    gt_idx = np.nonzero(gt_mask)[0]
    tp = np.zeros(len(idx), bool)
    matched = np.full(len(idx), -1)
    if len(gt_idx) and len(idx):
        ious = r.ious[np.ix_(idx, gt_idx)]
        covered = np.zeros(len(gt_idx), bool)
        for i in range(len(idx)):
            j = int(ious[i].argmax())
            if ious[i, j] >= iou_thr and not covered[j]:
                covered[j] = True
                tp[i] = True
                matched[i] = gt_idx[j]
    return tp, matched, idx, int(gt_mask.sum())


def _ap_area(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def pr_curve(results: list[ImageResult], iou_thr: float = 0.5, cls: int = 0):
    """Precision / recall / score arrays over all detections sorted by score."""
    tps, scores, n_gt = [], [], 0
    for r in results:
        tp, _, idx, g = match(r, iou_thr, cls=cls)
        tps.append(tp)
        scores.append(r.scores[idx])
        n_gt += g
    scores = np.concatenate(scores) if scores else np.zeros(0)
    order = np.argsort(-scores, kind="stable")
    tp = np.concatenate(tps)[order].astype(np.float64) if tps else np.zeros(0)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, np.finfo(np.float32).eps)
    return precision, recall, scores[order], n_gt


def average_precision(results: list[ImageResult], iou_thr: float, num_classes: int = 1) -> float:
    aps = []
    for c in range(num_classes):
        p, r, _, n_gt = pr_curve(results, iou_thr, c)
        aps.append(_ap_area(r, p) if n_gt else 0.0)
    return float(np.mean(aps))


def prf_at(results: list[ImageResult], score_thr: float, iou_thr: float = 0.5):
    """Precision, recall, F1 counting only detections with score >= score_thr."""
    tp = fp = n_gt = 0
    for r in results:
        t, _, _, g = match(r, iou_thr, score_thr)
        tp += int(t.sum())
        fp += int((~t).sum())
        n_gt += g
    p = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    f1 = 2 * p * rec / max(p + rec, 1e-12)
    return p, rec, f1, tp, fp, n_gt - tp


def prf_curve(results: list[ImageResult], thresholds: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """(len(thresholds), 3) precision, recall, F1."""
    return np.array([prf_at(results, t, iou_thr)[:3] for t in thresholds])


def summarize(results: list[ImageResult], score_thr: float, num_classes: int = 1) -> dict:
    p, r, f1, tp, fp, fn = prf_at(results, score_thr)
    conf = np.concatenate([x.scores[x.scores >= score_thr] for x in results]) if results else np.zeros(0)
    return {
        "precision": p, "recall": r, "f1": f1,
        "mAP50": average_precision(results, 0.5, num_classes),
        "mAP75": average_precision(results, 0.75, num_classes),
        "mAP50-95": float(np.mean([average_precision(results, t, num_classes) for t in IOU_THRS])),
        "conf_mean": float(conf.mean()) if len(conf) else np.nan,
        "conf_median": float(np.median(conf)) if len(conf) else np.nan,
        "conf_std": float(conf.std()) if len(conf) else np.nan,
        "TP": tp, "FP": fp, "FN": fn,
    }


def matched_pairs(results: list[ImageResult], score_thr: float, iou_thr: float = 0.5):
    """For every TP at the operating point: its score, IoU, angle error (deg) and the
    relative size of the banknote (sqrt of its area over the image area is computed
    by the caller). Also the scores of the FPs and the indices of the missed gt boxes."""
    rows, fp_scores, missed = [], [], []
    for i, r in enumerate(results):
        tp, matched, idx, _ = match(r, iou_thr, score_thr)
        for k, d in enumerate(idx):
            if tp[k]:
                j = matched[k]
                rows.append((i, j, r.scores[d], r.ious[d, j], angle_error_deg(r.boxes[d], r.gt[j])))
            else:
                fp_scores.append(r.scores[d])
        found = set(matched[tp].tolist())
        missed += [(i, j) for j in range(len(r.gt)) if j not in found]
    return rows, np.array(fp_scores), missed


def angle_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Difference between the directions of the LONG sides of two boxes, in [0, 90] degrees
    (a rectangle looks the same rotated 180 degrees)."""
    def long_side_angle(box):
        ang = box[4] if box[2] >= box[3] else box[4] + np.pi / 2
        return np.degrees(ang) % 180
    d = abs(long_side_angle(a) - long_side_angle(b)) % 180
    return float(min(d, 180 - d))

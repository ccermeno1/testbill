"""
Valida el fallback de rbox_iou contra Shapely (referencia exacta) con cajas aleatorias,
incluyendo casos degenerados (cajas identicas, disjuntas, una dentro de otra, rotadas 90).

    python ext_op_fallback/test_rbox_iou.py
"""
import math
import sys
import time

import numpy as np
import paddle
from shapely.geometry import Polygon

import ext_op
from ext_op import matched_rbox_iou, rbox_iou


def corners_np(r):
    cx, cy, w, h, a = r
    c, s = math.cos(a), math.sin(a)
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
    return np.stack([cx + dx * c - dy * s, cy + dx * s + dy * c], axis=1)


def iou_shapely(r1, r2):
    p1, p2 = Polygon(corners_np(r1)), Polygon(corners_np(r2))
    inter = p1.intersection(p2).area
    union = p1.area + p2.area - inter
    return inter / union if union > 0 else 0.0


def main():
    print("IS_FALLBACK =", ext_op.IS_FALLBACK)
    rng = np.random.default_rng(0)
    M, N = 12, 300
    b1 = np.stack([rng.uniform(0, 400, M), rng.uniform(0, 400, M), rng.uniform(5, 250, M),
                   rng.uniform(5, 250, M), rng.uniform(-math.pi, math.pi, M)], axis=1).astype(np.float32)
    b2 = np.stack([rng.uniform(0, 400, N), rng.uniform(0, 400, N), rng.uniform(5, 250, N),
                   rng.uniform(5, 250, N), rng.uniform(-math.pi, math.pi, N)], axis=1).astype(np.float32)
    # casos especiales
    b2[0] = b1[0]                                   # identica -> 1
    b2[1] = b1[1] * np.array([1, 1, 0.5, 0.5, 1])   # contenida (mismo centro y angulo) -> 0.25
    b2[2] = b1[2] + np.array([0, 0, 0, 0, math.pi / 2], dtype=np.float32)  # girada 90
    b2[3] = b1[3] + np.array([1000, 1000, 0, 0, 0], dtype=np.float32)      # disjunta -> 0
    b2[4] = b1[4] + np.array([b1[4][2], 0, 0, 0, 0], dtype=np.float32)     # tocando por una arista (a=0 no garantizado)

    ref = np.array([[iou_shapely(r1, r2) for r2 in b2] for r1 in b1], dtype=np.float32)
    out = rbox_iou(paddle.to_tensor(b1), paddle.to_tensor(b2)).numpy()
    err = np.abs(out - ref)
    print(f"rbox_iou [{M}x{N}]  max_err={err.max():.2e}  mean_err={err.mean():.2e}")
    print("  casos: identica=%.4f contenida=%.4f disjunta=%.4f" % (out[0, 0], out[1, 1], out[3, 3]))

    m_out = matched_rbox_iou(paddle.to_tensor(b1), paddle.to_tensor(b2[:M])).numpy()
    m_ref = np.array([iou_shapely(r1, r2) for r1, r2 in zip(b1, b2[:M])], dtype=np.float32)
    m_err = np.abs(m_out - m_ref).max()
    print(f"matched_rbox_iou [{M}]  max_err={m_err:.2e}")

    # tamano realista del asignador: 9 gt x 8400 anchors
    g = paddle.to_tensor(rng.uniform(0, 640, (9, 5)).astype(np.float32))
    p = paddle.to_tensor(rng.uniform(0, 640, (8400, 5)).astype(np.float32))
    rbox_iou(g, p)
    if paddle.device.is_compiled_with_cuda():
        paddle.device.synchronize()
    t0 = time.time()
    for _ in range(10):
        rbox_iou(g, p)
    if paddle.device.is_compiled_with_cuda():
        paddle.device.synchronize()
    print(f"rbox_iou [9x8400] en {paddle.device.get_device()}: {(time.time() - t0) / 10 * 1000:.1f} ms/llamada")

    ok = err.max() < 1e-3 and m_err < 1e-3
    print("OK" if ok else "FALLO")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

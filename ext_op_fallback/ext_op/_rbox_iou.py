"""
IoU exacta entre rectangulos rotados en Paddle puro (sin ops custom compiladas).

Mismo algoritmo que ppdet/ext_op/csrc/rbox_iou (adaptado de detectron2):
  1. vertices de cada caja a partir de (cx, cy, w, h, angulo_en_radianes)
  2. puntos candidatos del poligono interseccion = intersecciones arista-arista (16)
     + vertices de una caja dentro de la otra (4 + 4)
  3. la interseccion de dos convexos es convexa: se ordenan los candidatos validos
     por angulo alrededor de su centroide y se aplica la formula del cordon (shoelace)

Todo vectorizado sobre un eje de pares, asi que sirve para [M, N] (todos contra
todos) y para [K] (emparejados).
"""
import paddle

# pares (gt x pred) procesados a la vez en rbox_iou; ~100k pares ~ 0.7 GB de temporales
MAX_PAIRS = 100_000


def rbox2corners(rbox):
    """[..., 5] (cx, cy, w, h, a[rad]) -> [..., 4, 2] con la misma parametrizacion
    que get_rotated_vertices() en rbox_iou_utils.h (orden consecutivo de vertices)."""
    cx, cy, w, h, a = [rbox[..., i] for i in range(5)]
    c2 = paddle.cos(a) * 0.5
    s2 = paddle.sin(a) * 0.5
    p0x = cx - s2 * h - c2 * w
    p0y = cy + c2 * h - s2 * w
    p1x = cx + s2 * h - c2 * w
    p1y = cy - c2 * h - s2 * w
    p2x = 2 * cx - p0x
    p2y = 2 * cy - p0y
    p3x = 2 * cx - p1x
    p3y = 2 * cy - p1y
    xs = paddle.stack([p0x, p1x, p2x, p3x], axis=-1)
    ys = paddle.stack([p0y, p1y, p2y, p3y], axis=-1)
    return paddle.stack([xs, ys], axis=-1)


def _edge_intersections(c1, c2):
    """c1, c2: [P, 4, 2] -> puntos [P, 16, 2], mascara [P, 16]."""
    p1 = c1                                   # [P,4,2]
    d1 = paddle.roll(c1, -1, axis=1) - c1     # [P,4,2]
    p3 = c2
    d2 = paddle.roll(c2, -1, axis=1) - c2
    # broadcast: aristas de c1 en eje 1, de c2 en eje 2 -> [P,4,4,2]
    p1 = p1.unsqueeze(2)
    d1 = d1.unsqueeze(2)
    p3 = p3.unsqueeze(1)
    d2 = d2.unsqueeze(1)
    dp = p3 - p1

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    den = cross(d1, d2)                       # [P,4,4]
    nonpar = paddle.abs(den) > 1e-12
    den_safe = paddle.where(nonpar, den, paddle.ones_like(den))
    t = cross(dp, d2) / den_safe
    u = cross(dp, d1) / den_safe
    valid = nonpar & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    pts = p1 + t.unsqueeze(-1) * d1           # [P,4,4,2]
    P = pts.shape[0]
    return pts.reshape([P, 16, 2]), valid.reshape([P, 16])


def _corners_inside(pts, box):
    """pts: [P,4,2] puntos; box: [P,4,2] vertices consecutivos de un rectangulo.
    Devuelve mascara [P,4] de puntos dentro del rectangulo (bordes incluidos)."""
    a = box[:, 0:1, :]                        # [P,1,2]
    ab = box[:, 1:2, :] - a
    ad = box[:, 3:4, :] - a
    ap = pts - a                              # [P,4,2]
    proj_ab = (ap * ab).sum(-1)
    proj_ad = (ap * ad).sum(-1)
    len_ab = (ab * ab).sum(-1)
    len_ad = (ad * ad).sum(-1)
    eps = 1e-6
    return ((proj_ab >= -eps) & (proj_ab <= len_ab + eps) &
            (proj_ad >= -eps) & (proj_ad <= len_ad + eps))


def _convex_area(pts, mask):
    """pts: [P,K,2], mask: [P,K] -> area [P] del poligono convexo formado por los
    puntos validos (0 si hay menos de 3)."""
    P, K, _ = pts.shape
    maskf = mask.astype(pts.dtype).unsqueeze(-1)
    num_valid = maskf.sum(axis=1)                                  # [P,1]
    center = (pts * maskf).sum(axis=1) / paddle.clip(num_valid, min=1.0)
    diff = pts - center.unsqueeze(1)
    ang = paddle.atan2(diff[..., 1], diff[..., 0])                 # [P,K]
    ang = paddle.where(mask, ang, paddle.full_like(ang, 1e6))      # invalidos fuera del orden

    # Orden angular sin argsort (muy lento en Paddle para muchas filas cortas):
    # rank_i = numero de puntos validos j que van antes de i (empate -> por indice).
    ai = ang.unsqueeze(2)                                          # [P,K,1]
    aj = ang.unsqueeze(1)                                          # [P,1,K]
    idx = paddle.arange(K, dtype='int64')
    tie = (idx.unsqueeze(1) > idx.unsqueeze(0)).unsqueeze(0)       # [1,K,K]  j < i
    before = (aj < ai) | ((aj == ai) & tie)                        # [P,K,K]
    before = before & mask.unsqueeze(1)                            # solo j validos
    rank = before.astype(pts.dtype).sum(axis=2)                    # [P,K]

    # sucesor ciclico de cada vertice valido: el de rango (rank+1) mod num_valid
    succ_rank = rank + 1
    succ_rank = paddle.where(succ_rank >= num_valid, paddle.zeros_like(succ_rank), succ_rank)
    onehot = (rank.unsqueeze(1) == succ_rank.unsqueeze(2)) & mask.unsqueeze(1)   # [P,K(i),K(j)]
    succ = paddle.matmul(onehot.astype(pts.dtype), pts)            # [P,K,2]
    cross = pts[..., 0] * succ[..., 1] - pts[..., 1] * succ[..., 0]
    area = 0.5 * paddle.abs((cross * maskf.squeeze(-1)).sum(axis=1))
    # con < 3 vertices validos el "poligono" es un punto o segmento: area 0
    return paddle.where(num_valid.squeeze(-1) >= 3, area, paddle.zeros_like(area))


def _pairwise_iou_flat(r1, r2):
    """r1, r2: [P, 5] emparejados -> iou [P]."""
    c1 = rbox2corners(r1)
    c2 = rbox2corners(r2)
    inter_pts, inter_mask = _edge_intersections(c1, c2)
    in12 = _corners_inside(c1, c2)
    in21 = _corners_inside(c2, c1)
    pts = paddle.concat([inter_pts, c1, c2], axis=1)               # [P,24,2]
    mask = paddle.concat([inter_mask, in12, in21], axis=1)         # [P,24]
    inter = _convex_area(pts, mask)
    area1 = paddle.abs(r1[:, 2] * r1[:, 3])
    area2 = paddle.abs(r2[:, 2] * r2[:, 3])
    union = area1 + area2 - inter
    iou = inter / paddle.clip(union, min=1e-9)
    # cajas degeneradas -> 0 (como el kernel oficial)
    iou = paddle.where((area1 < 1e-14) | (area2 < 1e-14), paddle.zeros_like(iou), iou)
    return paddle.clip(iou, 0.0, 1.0)


def rbox_iou(rbox1, rbox2):
    """rbox1: [M, 5], rbox2: [N, 5] -> IoU [M, N]. Angulo en radianes."""
    rbox1 = paddle.cast(rbox1, 'float32')
    rbox2 = paddle.cast(rbox2, 'float32')
    M, N = rbox1.shape[0], rbox2.shape[0]
    if M == 0 or N == 0:
        return paddle.zeros([M, N], dtype='float32')
    # por trozos para acotar la memoria (los tensores intermedios son ~ pares x 24 x 24)
    rows_per_chunk = max(1, MAX_PAIRS // N)
    outs = []
    for i in range(0, M, rows_per_chunk):
        r1c = rbox1[i:i + rows_per_chunk]
        m = r1c.shape[0]
        r1 = r1c.unsqueeze(1).expand([m, N, 5]).reshape([m * N, 5])
        r2 = rbox2.unsqueeze(0).expand([m, N, 5]).reshape([m * N, 5])
        outs.append(_pairwise_iou_flat(r1, r2).reshape([m, N]))
    return outs[0] if len(outs) == 1 else paddle.concat(outs, axis=0)


def matched_rbox_iou(rbox1, rbox2):
    """rbox1, rbox2: [K, 5] -> IoU emparejada [K]."""
    rbox1 = paddle.cast(rbox1, 'float32')
    rbox2 = paddle.cast(rbox2, 'float32')
    if rbox1.shape[0] == 0:
        return paddle.zeros([0], dtype='float32')
    return _pairwise_iou_flat(rbox1, rbox2)

"""Carga de pesos ajenos en los modelos propios. Con mapeo explicito, sin magia.

Dos casos, y en los dos lo que no encaja se dice, no se tapa:

- **YOLOX de Megvii (COCO) en la cabeza propia.** `yolox_s.pth.tar`, Apache-2.0.
  Nuestro `CSPDarknet` + `PAFPN` es la misma red que su `YOLOPAFPN`: medido,
  354 tensores y 7.066.683 parametros en los dos, formas identicas. Solo cambian
  los nombres: ellos envuelven el backbone como `backbone.backbone.*` y llaman a
  los modulos del cuello `lateral_conv0`, `C3_p4`, `reduce_conv1`...; aqui son
  `backbone.*`, `neck.lateral_c5`, `neck.p4`, `neck.lateral_c4`... La tabla de
  abajo es ese mapeo, y se comprueba forma a forma al cargar.

  Su cabeza (`head.*`) NO se carga: es la de COCO, 80 clases y sin angulo. Lo
  que se hereda es "saber ver", no "saber donde esta el billete".

- **DDGRCF (DOTA) en el port.** Mismos nombres por construccion; ver
  `ddgrcf.load_pretrained`. Aqui solo se despacha.
"""

from __future__ import annotations

from pathlib import Path

import torch

#: Modulo del cuello de Megvii -> modulo del nuestro. Emparejados por funcion:
#: el 1x1 que reduce C5, el C3 tras concatenar con C4, etc. Verificado con las
#: formas: las ocho parejas tienen tensores identicos.
_MEGVII_NECK = {
    "lateral_conv0": "lateral_c5",
    "C3_p4": "p4",
    "reduce_conv1": "lateral_c4",
    "C3_p3": "p3",
    "bu_conv2": "down_p3",
    "C3_n3": "n4",
    "bu_conv1": "down_n4",
    "C3_n4": "n5",
}


def remap_megvii_key(key: str) -> str | None:
    """Clave del checkpoint de Megvii -> clave nuestra, o None si es su cabeza."""
    if key.startswith("backbone.backbone."):
        return "backbone." + key[len("backbone.backbone.") :]
    if key.startswith("backbone."):
        module, _, tail = key[len("backbone.") :].partition(".")
        if module not in _MEGVII_NECK:
            raise KeyError(
                f"modulo del cuello desconocido en el checkpoint: {module!r}. "
                "O no es un YOLOX de Megvii o su PAFPN ha cambiado"
            )
        return f"neck.{_MEGVII_NECK[module]}.{tail}"
    return None  # head.*: la cabeza de COCO, que no se quiere


def load_megvii_yolox(model, path: str | Path) -> dict:
    """Carga backbone y cuello desde un `yolox_*.pth.tar` de Megvii.

    Devuelve `{"loaded": n, "skipped_head": n, "mismatched": [...]}`. Es un
    error, no un aviso, que un tensor del backbone o del cuello quede sin
    cubrir: significaria que la variante no coincide (cargar `yolox_s` en
    `nano`, por ejemplo), y entrenar "preentrenado a medias" sin saberlo es
    peor que entrenar de cero.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    own = model.state_dict()

    mapped, skipped_head, mismatched = {}, 0, []
    for key, value in state.items():
        target = remap_megvii_key(key)
        if target is None:
            skipped_head += 1
            continue
        if target not in own:
            mismatched.append(f"{key} -> {target}: no existe")
        elif own[target].shape != value.shape:
            mismatched.append(
                f"{key} -> {target}: forma {tuple(value.shape)} != {tuple(own[target].shape)}"
            )
        else:
            mapped[target] = value

    expected = [k for k in own if k.startswith(("backbone.", "neck."))]
    uncovered = [k for k in expected if k not in mapped]
    if mismatched or uncovered:
        raise RuntimeError(
            f"{Path(path).name} no encaja con {model.variant!r}: "
            f"{len(mismatched)} tensores con conflicto y {len(uncovered)} sin cubrir. "
            f"Primeros: {(mismatched + uncovered)[:3]}. Comprueba que la variante "
            "del checkpoint (nano/tiny/s) es la del modelo"
        )
    model.load_state_dict(mapped, strict=False)
    return {"loaded": len(mapped), "skipped_head": skipped_head, "mismatched": mismatched}


def load_pretrained(model, path: str | Path) -> str:
    """Despacha por arquitectura y devuelve una nota para el registro."""
    from testbank.models.ddgrcf import DdgrcfYoloxObb
    from testbank.models.ddgrcf import load_pretrained as load_ddgrcf

    if isinstance(model, DdgrcfYoloxObb):
        skipped = load_ddgrcf(model, path)
        return (
            f"preentreno DOTA cargado desde {Path(path).name}; "
            f"{len(skipped)} tensores saltados (capa de clase): {sorted(skipped)[:2]}"
        )
    report = load_megvii_yolox(model, path)
    return (
        f"preentreno COCO (Megvii) cargado desde {Path(path).name}: "
        f"{report['loaded']} tensores de backbone+cuello; su cabeza "
        f"({report['skipped_head']} tensores) descartada"
    )


__all__ = ["load_megvii_yolox", "load_pretrained", "remap_megvii_key"]

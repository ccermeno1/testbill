"""Index of pixel dimensions, reading only headers with Pillow.

It was needed earlier than planned. It was scheduled for bbox_coco, the only
format that declared requires_image_size, but the CANONICAL ORDER needs it too:
normalized coordinates scale x and y by different factors, so without the image
aspect the "longest side" is not the longest side.

`Image.open` does not decode the pixel data until asked, so reading `.size`
costs as much as reading the header.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DEFAULT_INDEX_NAME = "image_sizes.json"


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height


def read_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size  # (width, height)


def load_index(path: str | Path) -> dict[str, tuple[int, int]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: (int(value[0]), int(value[1])) for key, value in raw.items()}


class SizeIndex:
    """Resolves the aspect of a sample, building the index if needed."""

    def __init__(self, index: dict[str, tuple[int, int]] | None = None):
        self._index: dict[str, tuple[int, int]] = dict(index or {})

    @classmethod
    def for_samples(cls, samples, *, cache_path: str | Path | None = None) -> SizeIndex:
        if cache_path is not None and Path(cache_path).is_file():
            cached = load_index(cache_path)
            missing = [s for s in samples if s.sample_id not in cached]
            if not missing:
                return cls(cached)
            cached.update(
                {s.sample_id: read_size(s.image_path) for s in missing}
            )
            Path(cache_path).write_text(
                json.dumps({k: list(v) for k, v in sorted(cached.items())}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return cls(cached)
        index = {s.sample_id: read_size(s.image_path) for s in samples}
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(
                json.dumps({k: list(v) for k, v in sorted(index.items())}, indent=2)
                + "\n",
                encoding="utf-8",
            )
        return cls(index)

    def size(self, sample_id: str) -> tuple[int, int]:
        try:
            return self._index[sample_id]
        except KeyError:
            raise KeyError(
                f"'{sample_id}' is not in the size index; rebuild it"
            ) from None

    def aspect(self, sample_id: str) -> float:
        width, height = self.size(sample_id)
        return width / height

    def __len__(self) -> int:
        return len(self._index)

__all__ = ["DEFAULT_INDEX_NAME", "ImageSize", "SizeIndex", "load_index", "read_size"]

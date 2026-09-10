"""Configuracion validada con pydantic. Cada ejecucion guarda la config resuelta."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    root: Path = Path("data/raw")
    splits_dir: Path = Path("splits")
    derived_dir: Path = Path("data/derived")


class SplitConfig(StrictModel):
    """Solo se usa en modo crear. En modo adoptar la particion ya viene dada."""

    train: float = 0.70
    valid: float = 0.15
    test: float = 0.15
    seed: int = 20260910
    group_key: str = "none"
    group_regex: str | None = None
    group_manifest: Path | None = None
    i_confirm_independence: bool = False
    folds: int = 5

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "SplitConfig":
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"los porcentajes de particion deben sumar 1, suman {total}")
        return self

    @property
    def ratios(self) -> tuple[float, float, float]:
        return (self.train, self.valid, self.test)


class AnnotationPolicyConfig(StrictModel):
    """Umbral de visibilidad de la politica de anotacion.

    Se aplica de verdad al anotar, no aqui: el codigo solo puede senalar
    anotaciones que la contradicen. Documentado en el README como guia.
    """

    visibility_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)
    #: Filtro EN CARGA: se descarta toda anotacion cuya area sea menor que esta
    #: fraccion del area de la mayor anotacion de su misma imagen, de modo que
    #: en cada imagen se conserva el billete de delante. Es una aproximacion a
    #: `visibility_threshold`, no una medida de oclusion, y no borra nada: los
    #: ficheros de origen quedan intactos y lo descartado se reporta.
    #: Subirlo o bajarlo no requiere reexportar el dataset.
    min_relative_area: float = Field(default=0.25, ge=0.0, le=1.0)


class CropConfig(StrictModel):
    """El margen decide cobertura y contaminacion, asi que va en la config.

    Sin fijarlo y registrarlo, dos ejecuciones no son comparables: el margen
    intercambia una metrica por la otra directamente.
    """

    margin: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Barrido con el que se reporta, para que el intercambio quede a la vista.
    margin_sweep: tuple[float, ...] = (0.0, 0.05, 0.10)


class ContaminationConfig(StrictModel):
    """Dos umbrales, porque la distribucion es bimodal, no continua.

    Medido con un detector PERFECTO (prediccion = verdad) sobre train+valid del
    export actual, a margen 0.05, con `contamination_floor`:

        un billete   n=301   p95 = 0.0000
        abanico      n=378   p95 = 0.8906

    Un umbral unico seria inutil en los dos sentidos a la vez: exigente de mas
    para los abanicos, donde ni un detector perfecto puede bajar del 0.89 --
    si un billete esta parcialmente tapado, su caja contiene por fuerza pixeles
    del que lo tapa -- y flojo de mas para las imagenes de un solo billete,
    donde el suelo es cero exacto y cualquier contaminacion es un error real.

    Aviso al leer el umbral de abanico: entre el suelo (0.89) y el techo (1.0)
    quedan 11 puntos, asi que discrimina poco. Para comparar candidatos en
    abanicos mira la MEDIANA, que el detector perfecto deja en 0.023 y tiene
    recorrido de sobra.

    Los dos numeros salen de los datos, asi que hay que rederivarlos cuando
    cambie el export: `contamination_floor` en `metrics/crop.py` los recalcula.
    """

    #: Suelo medido: 0.0000. Cualquier contaminacion aqui es un error real.
    single_max: float = Field(default=0.01, ge=0.0, le=1.0)
    #: Suelo medido: 0.8906. Deja ~3 puntos de holgura, un cuarto del recorrido
    #: que queda hasta 1.0.
    fan_max: float = Field(default=0.92, ge=0.0, le=1.0)
    #: Percentil sobre el que se aplican los dos umbrales.
    percentile: float = Field(default=95.0, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _fan_is_not_stricter_than_single(self) -> "ContaminationConfig":
        if self.fan_max < self.single_max:
            raise ValueError(
                "fan_max no puede ser menor que single_max: un abanico nunca "
                "puede salir mas limpio que una imagen de un solo billete"
            )
        return self


class MetricsConfig(StrictModel):
    match_iou: float = Field(default=0.5, gt=0.0, lt=1.0)
    coverage_target: float = Field(default=0.98, gt=0.0, le=1.0)
    coverage_percentile: float = Field(default=5.0, ge=0.0, le=100.0)
    contamination: ContaminationConfig = ContaminationConfig()
    bootstrap_samples: int = 2000
    #: La unidad de remuestreo es la imagen, no la deteccion: varios billetes de
    #: una misma imagen estan correlacionados y remuestrear detecciones estrecha
    #: los intervalos artificialmente.
    bootstrap_unit: str = "image"
    seed: int = 20260910

    @field_validator("bootstrap_unit")
    @classmethod
    def _only_image(cls, value: str) -> str:
        if value != "image":
            raise ValueError(
                "la unidad de bootstrap tiene que ser 'image': remuestrear "
                "detecciones ignora la correlacion dentro de cada imagen"
            )
        return value


class DetectorConfig(StrictModel):
    """Lo que se le pasa al entrenador de turno. Va en la config y se registra:
    dos ejecuciones con epochs distintos no son comparables."""

    #: El despliegue es movil, asi que se priorizan nano/small.
    variant: str = "nano"
    epochs: int = Field(default=100, gt=0)
    image_size: int = Field(default=640, gt=0)
    batch_size: int = Field(default=8, gt=0)
    #: Umbral de confianza en INFERENCIA. Bajo a proposito: las metricas de
    #: deteccion necesitan la cola de baja confianza para trazar la curva
    #: precision-recall; recortarla arriba infla el AP artificialmente.
    confidence_threshold: float = Field(default=0.01, gt=0.0, lt=1.0)
    #: IoU del NMS rotado.
    nms_iou: float = Field(default=0.5, gt=0.0, lt=1.0)


class VizConfig(StrictModel):
    #: Muestra fija de validacion, siempre la misma, para comparar a ojo.
    sample_count: int = 12
    seed: int = 20260910
    output_dir: Path = Path("runs/_inspection")
    dpi: int = 110


class Config(StrictModel):
    data: DataConfig = DataConfig()
    splits: SplitConfig = SplitConfig()
    annotation_policy: AnnotationPolicyConfig = AnnotationPolicyConfig()
    crop: CropConfig = CropConfig()
    detector: DetectorConfig = DetectorConfig()
    metrics: MetricsConfig = MetricsConfig()
    viz: VizConfig = VizConfig()
    runs_dir: Path = Path("runs")

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def dump_yaml(self) -> str:
        return yaml.safe_dump(
            yaml.safe_load(self.model_dump_json()), sort_keys=False, allow_unicode=True
        )

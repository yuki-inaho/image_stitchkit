"""
Native stitching; input snapshots, immutable output views, explicit CPU/CUDA selection.
"""

import enum
import os
import pathlib
from typing import Annotated

import numpy
from numpy.typing import NDArray


nanobind_version: str = '2.15.0'

class StitchError(RuntimeError):
    pass

class Method(enum.Enum):
    rew = 0

    niswgsp = 1

    gesgsp = 2

class Device(enum.Enum):
    cpu = 0

    cuda = 1

    auto = 2

class StructureMode(enum.Enum):
    edges = 0

    lineae = 1

    none = 2

class Options:
    def __init__(self) -> None: ...

    def validate(self) -> None: ...

    @property
    def method(self) -> Method: ...

    @method.setter
    def method(self, arg: Method, /) -> None: ...

    @property
    def device(self) -> Device: ...

    @device.setter
    def device(self, arg: Device, /) -> None: ...

    @property
    def structures(self) -> StructureMode: ...

    @structures.setter
    def structures(self, arg: StructureMode, /) -> None: ...

    @property
    def grid_size(self) -> int: ...

    @grid_size.setter
    def grid_size(self, arg: int, /) -> None: ...

    @property
    def max_features(self) -> int: ...

    @max_features.setter
    def max_features(self, arg: int, /) -> None: ...

    @property
    def ratio_threshold(self) -> float: ...

    @ratio_threshold.setter
    def ratio_threshold(self, arg: float, /) -> None: ...

    @property
    def ransac_threshold(self) -> float: ...

    @ransac_threshold.setter
    def ransac_threshold(self, arg: float, /) -> None: ...

    @property
    def ransac_iterations(self) -> int: ...

    @ransac_iterations.setter
    def ransac_iterations(self, arg: int, /) -> None: ...

    @property
    def minimum_matches(self) -> int: ...

    @minimum_matches.setter
    def minimum_matches(self, arg: int, /) -> None: ...

    @property
    def seed(self) -> int: ...

    @seed.setter
    def seed(self, arg: int, /) -> None: ...

    @property
    def use_apap(self) -> bool: ...

    @use_apap.setter
    def use_apap(self, arg: bool, /) -> None: ...

    @property
    def apap_sigma(self) -> float: ...

    @apap_sigma.setter
    def apap_sigma(self, arg: float, /) -> None: ...

    @property
    def alignment_weight(self) -> float: ...

    @alignment_weight.setter
    def alignment_weight(self, arg: float, /) -> None: ...

    @property
    def local_similarity_weight(self) -> float: ...

    @local_similarity_weight.setter
    def local_similarity_weight(self, arg: float, /) -> None: ...

    @property
    def global_beta(self) -> float: ...

    @global_beta.setter
    def global_beta(self, arg: float, /) -> None: ...

    @property
    def global_gamma(self) -> float: ...

    @global_gamma.setter
    def global_gamma(self, arg: float, /) -> None: ...

    @property
    def structure_weight(self) -> float: ...

    @structure_weight.setter
    def structure_weight(self, arg: float, /) -> None: ...

    @property
    def tps_lambda(self) -> float: ...

    @tps_lambda.setter
    def tps_lambda(self, arg: float, /) -> None: ...

    @property
    def max_tps_controls(self) -> int: ...

    @max_tps_controls.setter
    def max_tps_controls(self, arg: int, /) -> None: ...

    @property
    def max_structures(self) -> int: ...

    @max_structures.setter
    def max_structures(self, arg: int, /) -> None: ...

    @property
    def feather_width(self) -> float: ...

    @feather_width.setter
    def feather_width(self, arg: float, /) -> None: ...

    @property
    def max_canvas_pixels(self) -> int: ...

    @max_canvas_pixels.setter
    def max_canvas_pixels(self, arg: int, /) -> None: ...

    @property
    def lineae_model(self) -> pathlib.Path: ...

    @lineae_model.setter
    def lineae_model(self, arg: str | os.PathLike, /) -> None: ...

    @property
    def lineae_variant(self) -> str: ...

    @lineae_variant.setter
    def lineae_variant(self, arg: str, /) -> None: ...

    @property
    def lineae_device(self) -> Device: ...

    @lineae_device.setter
    def lineae_device(self, arg: Device, /) -> None: ...

    @property
    def lineae_threshold(self) -> float: ...

    @lineae_threshold.setter
    def lineae_threshold(self, arg: float, /) -> None: ...

class PreparedScene:
    @property
    def image_count(self) -> int: ...

    @property
    def registration_ms(self) -> float: ...

    @property
    def feature_extractions(self) -> int: ...

class Result:
    @property
    def image(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None, None), order='C', writable=False)]: ...

    @property
    def mask(self) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None), order='C', writable=False)]: ...

    @property
    def origin(self) -> tuple[float, float]: ...

    def report_json(self) -> str: ...

    def save(self, path: str | os.PathLike) -> None: ...

def stitch(images: list, options: Options) -> Result: ...

def prepare(images: list, options: Options) -> PreparedScene: ...

def compose(scene: PreparedScene, options: Options) -> Result: ...

def load(path: str | os.PathLike, max_side: int = 0) -> Annotated[NDArray[numpy.uint8], dict(shape=(None, None, None), order='C', writable=False)]: ...

def capabilities() -> dict: ...

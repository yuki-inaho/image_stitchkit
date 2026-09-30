"""C++ image stitching with immutable configuration and explicit image ownership.

Images are RGB uint8 arrays, never BGR. Non-contiguous arrays are explicitly
made contiguous, then copied into C++ before computation releases the GIL.
Returned image/mask arrays are read-only views with independent lifetime owners.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from . import _native

__version__: str = _native.__version__
StitchError = _native.StitchError
PreparedScene = _native.PreparedScene
Result = _native.Result
_defaults = _native.Options()


@dataclass(frozen=True, slots=True)
class Options:
    """Validated immutable settings. Defaults are taken from C++, not duplicated.

    Registration settings cannot change when composing an existing PreparedScene.
    Method, geometry and device settings can change independently. CUDA currently
    covers sampling/compositing only, not feature extraction or sparse solving.
    """
    method: Literal["rew", "niswgsp", "gesgsp"] = _defaults.method.name
    device: Literal["cpu", "cuda", "auto"] = _defaults.device.name
    structures: Literal["edges", "lineae", "none"] = _defaults.structures.name
    grid_size: int = _defaults.grid_size
    max_features: int = _defaults.max_features
    ratio_threshold: float = _defaults.ratio_threshold
    ransac_threshold: float = _defaults.ransac_threshold
    ransac_iterations: int = _defaults.ransac_iterations
    minimum_matches: int = _defaults.minimum_matches
    seed: int = _defaults.seed
    use_apap: bool = _defaults.use_apap
    apap_sigma: float = _defaults.apap_sigma
    alignment_weight: float = _defaults.alignment_weight
    local_similarity_weight: float = _defaults.local_similarity_weight
    global_beta: float = _defaults.global_beta
    global_gamma: float = _defaults.global_gamma
    structure_weight: float = _defaults.structure_weight
    tps_lambda: float = _defaults.tps_lambda
    max_tps_controls: int = _defaults.max_tps_controls
    max_structures: int = _defaults.max_structures
    feather_width: float = _defaults.feather_width
    max_canvas_pixels: int = _defaults.max_canvas_pixels
    lineae_model: str | Path = ""
    lineae_variant: str = _defaults.lineae_variant
    lineae_device: Literal["cpu", "cuda", "auto"] = _defaults.lineae_device.name
    lineae_threshold: float = _defaults.lineae_threshold

    def __post_init__(self) -> None:
        _to_native(self).validate()

    def with_updates(self, **changes: Any) -> Options:
        """Return a new validated configuration; never mutate one in-flight."""
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        return {f.name: str(getattr(self, f.name)) if f.name == "lineae_model"
                else getattr(self, f.name) for f in fields(self)}


def _to_native(options: Options) -> _native.Options:
    if not isinstance(options, Options):
        raise TypeError("options must be stitchkit.Options")
    native = _native.Options()
    enum_types = {"method": _native.Method, "device": _native.Device,
                  "lineae_device": _native.Device, "structures": _native.StructureMode}
    for f in fields(options):
        value = getattr(options, f.name)
        if f.name in enum_types:
            if not isinstance(value, str) or value not in enum_types[f.name].__members__:
                raise ValueError(f"Invalid {f.name}: {value!r}")
            value = enum_types[f.name].__members__[value]
        elif f.name == "lineae_model":
            if not isinstance(value, (str, Path)):
                raise TypeError("lineae_model must be a path or string")
            value = str(value)
        elif f.type == "int":
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{f.name} must be an integer")
            value = int(value)
        elif f.type == "float":
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{f.name} must be a real number")
            value = float(value)
        elif f.type == "bool" and not isinstance(value, bool):
            raise TypeError(f"{f.name} must be a boolean")
        elif f.type == "str" and not isinstance(value, str):
            raise TypeError(f"{f.name} must be a string")
        setattr(native, f.name, value)
    return native


def _arrays(images: Sequence[NDArray[np.uint8]]) -> list[NDArray[np.uint8]]:
    if isinstance(images, np.ndarray):
        raise TypeError("Pass a sequence of 2..16 RGB images, not one ndarray")
    arrays = list(images)
    if not 2 <= len(arrays) <= 16:
        raise ValueError("Expected 2 to 16 ordered RGB images")
    for index, a in enumerate(arrays):
        if not isinstance(a, np.ndarray) or a.dtype != np.uint8:
            raise TypeError(f"Image {index}: expected a NumPy uint8 array, no implicit dtype conversion")
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"Image {index}: expected RGB shape (H,W,3); RGBA/grayscale are not implicit")
        if not 2 <= a.shape[0] <= 32768 or not 2 <= a.shape[1] <= 32768 or a.shape[0] * a.shape[1] > 64_000_000:
            raise ValueError(f"Image {index}: dimensions exceed the C++ image limits")
        arrays[index] = np.ascontiguousarray(a)
    return arrays


def stitch(images: Sequence[NDArray[np.uint8]], options: Options | None = None) -> Result:
    """Snapshot and stitch an ordered, overlapping sequence; no subprocess is used."""
    return _native.stitch(_arrays(images), _to_native(options if options is not None else Options()))


def prepare(images: Sequence[NDArray[np.uint8]], options: Options | None = None) -> PreparedScene:
    """Own an input snapshot and compute correspondences once for mode comparisons."""
    return _native.prepare(_arrays(images), _to_native(options if options is not None else Options()))


def compose(scene: PreparedScene, options: Options | None = None) -> Result:
    """Estimate a warp and render using existing correspondences, without re-matching.

    Pass the same registration options used for prepare(). Defaults are not
    inferred from the scene; a mismatch raises StitchError rather than reusing
    correspondences computed under different settings.
    """
    if not isinstance(scene, PreparedScene):
        raise TypeError("scene must be a PreparedScene returned by prepare")
    return _native.compose(scene, _to_native(options if options is not None else Options()))


def load(path: str | Path, *, max_side: int = 0) -> NDArray[np.uint8]:
    """Read PNG/JPEG with the C++ codec, optionally resize. Input PNG alpha is flattened on white."""
    if isinstance(max_side, bool) or not isinstance(max_side, Integral):
        raise TypeError("max_side must be an integer")
    return _native.load(Path(path), int(max_side))


def report(result: Result) -> dict[str, Any]:
    """Parse the same locale-independent JSON schema used by the native CLI."""
    return json.loads(result.report_json())


def capabilities() -> dict[str, Any]:
    """Return build features and actual runtime device availability, not just requested flags."""
    return dict(_native.capabilities())


__all__ = ["Options", "Result", "PreparedScene", "StitchError", "stitch", "prepare", "compose",
           "load", "report", "capabilities", "__version__"]

"""Small, reusable measurement primitives; plotting stays outside the native API.

No pandas or plotting dependency is imported here. Timed regions contain only the
public native call, not disk I/O, JSON parsing, comparisons, or plotting. A scope
named 'prepared' deliberately excludes preparation and must not be presented as
an end-to-end latency. Wall time includes the Python adapter and input snapshots.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter_ns, process_time_ns
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray
from . import Options, Result, compose, prepare, report, stitch

METHODS = ("rew", "niswgsp", "gesgsp")
SCOPES = ("end_to_end", "prepared")

@dataclass
class BenchmarkRun:
    records: list[dict[str, Any]]
    outputs: dict[str, Result]
    preparation: dict[str, Any]


def benchmark_images(images: Sequence[NDArray[np.uint8]], *, case: str,
                     options: Options | None = None, repeats: int = 5,
                     warmups: int = 1, seed: int = 314159) -> BenchmarkRun:
    """Interleave all method/scope combinations after untimed warmups.

    Five repeats are an exploratory comparison, not a precise tail-latency or
    statistical significance estimate. Input order and algorithm seed are fixed;
    only measurement order is randomized. Output equality is checked outside the
    timed region on every repeat to catch state leakage and reuse regressions.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a nonnegative integer")
    base = options if options is not None else Options()
    started = perf_counter_ns()
    scene = prepare(images, base)
    preparation = {"case": case, "prepare_wall_ms": (perf_counter_ns()-started)/1e6,
                   "registration_ms": scene.registration_ms,
                   "feature_extractions": scene.feature_extractions,
                   "images": scene.image_count,
                   "input_bytes": sum(a.nbytes for a in images)}
    configs = {method: base.with_updates(method=method) for method in METHODS}
    jobs = [(method, scope) for method in METHODS for scope in SCOPES]
    def invoke(method: str, scope: str) -> Result:
        return stitch(images, configs[method]) if scope == "end_to_end" else compose(scene, configs[method])
    for _ in range(warmups):
        for method, scope in jobs:
            invoke(method, scope)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    outputs: dict[str, Result] = {}
    for repeat in range(repeats):
        for index in rng.permutation(len(jobs)):
            method, scope = jobs[int(index)]
            cpu_start, wall_start = process_time_ns(), perf_counter_ns()
            result = invoke(method, scope)
            wall_ms = (perf_counter_ns()-wall_start)/1e6
            cpu_ms = (process_time_ns()-cpu_start)/1e6
            d = report(result)
            if method in outputs:
                if not np.array_equal(result.image, outputs[method].image) or not np.array_equal(result.mask, outputs[method].mask):
                    raise RuntimeError(f"Non-deterministic or reuse-mismatched output: {case}/{method}/{scope}")
            else:
                outputs[method] = result
            record = {"case": case, "method": method, "scope": scope,
                      "repeat": repeat, "execution_order": len(records), "wall_ms": wall_ms,
                      "process_cpu_ms": cpu_ms, "prepare_wall_ms": preparation["prepare_wall_ms"],
                      "registration_reused": d["registration_reused"],
                      "feature_extractions": d["feature_extractions"],
                      "actual_device": d["actual_device"], "feature_matches": d["feature_matches"],
                      "fitted_alignment_rmse_px": d["alignment_rmse_after"],
                      "structure_equations": d["structure_equations"],
                      "output_width": d["width"], "output_height": d["height"],
                      "valid_pixels": d["valid_pixels"]}
            record.update({f"{name}_ms": value for name, value in d["timing_ms"].items()})
            records.append(record)
    return BenchmarkRun(records, outputs, preparation)


def reference_metrics(result: Result, truth: NDArray[np.uint8], *, border: int = 3) -> dict[str, float | int]:
    """RGB RMSE on covered reference pixels PLUS reference-domain coverage.

    The panorama and truth must share the first-image coordinate frame. We do not
    align the output to the truth after stitching; doing so would hide geometric
    errors. Integer origins are enforced. Coverage is reported separately so that
    a small surviving patch cannot masquerade as a high-quality full panorama.
    """
    if not isinstance(truth, np.ndarray) or truth.dtype != np.uint8 or truth.ndim != 3 or truth.shape[2] != 3:
        raise TypeError("truth must be RGB uint8")
    if isinstance(border, bool) or not isinstance(border, int) or border < 0 or 2*border >= min(truth.shape[:2]):
        raise ValueError("border does not define a nonempty evaluation region")
    ox, oy = result.origin
    if abs(ox-round(ox)) > 1e-6 or abs(oy-round(oy)) > 1e-6:
        raise ValueError("Noninteger origin requires an explicit resampling protocol")
    h,w=truth.shape[:2]
    yy,xx=np.mgrid[border:h-border,border:w-border]
    px,py=xx-int(round(ox)),yy-int(round(oy))
    image,mask=result.image,result.mask
    inside=(px>=0)&(py>=0)&(px<image.shape[1])&(py<image.shape[0])
    safe_x=np.clip(px,0,image.shape[1]-1);safe_y=np.clip(py,0,image.shape[0]-1)
    valid=inside & (mask[safe_y,safe_x] > 0)
    count=int(valid.sum()); support=int(valid.size)
    if not count:raise ValueError("No valid pixels in the reference domain")
    delta=image[safe_y[valid],safe_x[valid]].astype(np.float64)-truth[yy[valid],xx[valid]].astype(np.float64)
    return {"rgb_rmse_255":float(np.sqrt(np.mean(delta*delta))),
            "coverage_fraction":count/support,"covered_pixels":count,"reference_pixels":support}

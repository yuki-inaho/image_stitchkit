"""Compare two native CLI builds with identical inputs and randomized paired trials.

Requires the notebook dependency group (NumPy, pandas, Pillow). Both executables
must be built in Release with the same compiler and dependency versions for a
meaningful code-change comparison. This tool does not enforce that condition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter_ns

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Original v0.1 stitch executable")
    parser.add_argument("--current", type=Path, default=root / "build/release/stitch")
    parser.add_argument("--output", type=Path, default=root / "results/refactor_comparison")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    executables = {"baseline_0.1": args.baseline.resolve(), "refactor_0.2": args.current.resolve()}
    for executable in executables.values():
        if not executable.is_file():
            parser.error(f"Executable not found: {executable}")
    assets = [root / f"tests/assets/synthetic/view{i}.png" for i in range(3)]
    for asset in assets:
        if not asset.is_file():
            parser.error(f"Test asset not found: {asset}")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    reference: np.ndarray | None = None
    rng = np.random.default_rng(args.seed)
    # Warmup is excluded. Each measured round contains both versions.
    for repetition in range(-1, args.repeats):
        for version in rng.permutation(list(executables)):
            image, report = out / f"{version}.png", out / f"{version}.json"
            command = [str(executables[version]), "--method", "rew", "--device", "cpu",
                       "--max-side", "0", "--output", str(image), "--report", str(report),
                       *(str(path) for path in assets)]
            start = perf_counter_ns()
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
            wall_ms = (perf_counter_ns() - start) / 1e6
            diagnostics = json.loads(report.read_text(encoding="utf-8"))
            with Image.open(image) as loaded:
                pixels = np.asarray(loaded.convert("RGBA")).copy()
            # Validate every output outside the timed region. Do not drop failures.
            if reference is None:
                reference = pixels
            else:
                np.testing.assert_array_equal(pixels, reference)
            if repetition >= 0:
                records.append({"version": version, "repeat": repetition,
                                "registration_ms": diagnostics["timing_ms"]["registration"],
                                "cli_wall_including_io_ms": wall_ms,
                                "feature_extractions_reported": diagnostics.get("feature_extractions")})
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(out / "before_after_raw.csv", index=False)
    summary = frame.groupby("version")[["registration_ms", "cli_wall_including_io_ms"]].median()
    summary.to_csv(out / "before_after_medians.csv")
    protocol = {"method": "rew", "case": "3-image synthetic chain", "warmups": 1,
                "repeats": args.repeats, "order_seed": args.seed, "output_rgba_identical": True,
                "notes": ["registration_ms is native stage timing; CLI wall includes process startup and image I/O",
                          "Use the same compiler, Release configuration, and external libraries for both builds",
                          "Baseline feature extraction count is not instrumented; do not invent an observed count"]}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string())
    print("All output RGBA arrays are identical")


if __name__ == "__main__":
    main()

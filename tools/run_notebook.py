"""Execute the notebook with the current uv interpreter and export HTML with embedded outputs."""
from __future__ import annotations
import argparse
import json
from importlib.metadata import distribution
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path("notebooks/stitchkit_evaluation.ipynb"))
    parser.add_argument("--export-only", action="store_true", help="Export an already executed notebook without new measurements")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (args.notebook if args.notebook.is_absolute() else root / args.notebook).resolve()
    # A private kernelspec binds execution to this exact interpreter, not a global kernel.
    with tempfile.TemporaryDirectory(prefix="stitchkit-kernel-") as folder:
        base = Path(folder)
        spec = base / "kernels/stitchkit"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (stitchkit)", "language": "python"}))
        old_path = os.environ.get("JUPYTER_PATH")
        data_paths = [str(base)]
        # Also support dependencies inherited from another environment during offline validation.
        dist = distribution("nbconvert")
        for entry in dist.files or []:
            if str(entry).replace("\\", "/").endswith("share/jupyter/nbconvert/templates/lab/conf.json"):
                data_paths.append(str(dist.locate_file(entry).resolve().parents[3]))
                break
        if old_path: data_paths.append(old_path)
        os.environ["JUPYTER_PATH"] = os.pathsep.join(data_paths)
        try:
            nb = nbformat.read(source, as_version=4)
            if not args.export_only:
                client = NotebookClient(nb, timeout=600, kernel_name="stitchkit",
                                        resources={"metadata":{"path":str(root)}}, allow_errors=False)
                client.execute()
                nbformat.write(nb, source)
            exporter = HTMLExporter()
            exporter.exclude_input_prompt = True
            body, _ = exporter.from_notebook_node(nb)
            source.with_suffix(".html").write_text(body, encoding="utf-8")
        finally:
            if old_path is None: os.environ.pop("JUPYTER_PATH",None)
            else: os.environ["JUPYTER_PATH"] = old_path
    print(f"{source}: HTML exported (new execution: {not args.export_only})")

if __name__ == "__main__":
    main()

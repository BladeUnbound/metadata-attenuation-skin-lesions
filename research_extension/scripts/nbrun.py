"""Minimal notebook executor (the environment has nbformat but no nbclient/ipykernel).

Executes code cells in order in one shared namespace with the notebook's directory as cwd. Each cell's stdout is
recorded as a stream output, every open matplotlib figure as a PNG display_data, and an exception as an error
output, after which execution stops. Writes the executed notebook in place.

Usage: python nbrun.py notebook.ipynb
"""
import base64
import contextlib
import io
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat


def run(path):
    nb = nbformat.read(path, as_version=4)
    ns = {"__name__": "__main__"}
    os.chdir(os.path.dirname(os.path.abspath(path)))
    count, ok = 0, True
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        cell.execution_count = count
        cell.outputs = []
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(cell.source, f"<cell {count}>", "exec"), ns)
        except Exception as e:  # record and stop
            ok = False
            if buf.getvalue():
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue()))
            cell.outputs.append(nbformat.v4.new_output("error", ename=type(e).__name__, evalue=str(e),
                                                       traceback=traceback.format_exc().splitlines()))
            break
        if buf.getvalue():
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue()))
        for num in plt.get_fignums():
            fig = plt.figure(num)
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
            cell.outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(png.getvalue()).decode()}))
            plt.close(fig)
    nbformat.write(nb, path)
    print(("OK" if ok else "FAILED"), "executed", count, "cells:", path)
    return ok


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1]) else 1)

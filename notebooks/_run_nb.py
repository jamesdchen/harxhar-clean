"""Execute one notebook; embed savefig PNGs so plots show in the ipynb."""

from __future__ import annotations

import base64
import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat.v4 import new_output


def embed_savefig_pngs(nb, nb_path: Path) -> None:
    """Attach image/png from fig.savefig(OUT / \"name.png\") if missing."""
    repo = nb_path.parent.parent
    # OUT is always results/<stem-specific>; resolve from savefig strings.
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        names = re.findall(r'fig\.savefig\(\s*OUT\s*/\s*"([^"]+\.png)"', src)
        if not names:
            continue
        outs = list(cell.get("outputs") or [])
        have = False
        for o in outs:
            data = (o.get("data") or {}) if hasattr(o, "get") else {}
            if "image/png" in data:
                have = True
                break
        if have:
            continue
        for name in names:
            candidates = [
                repo / "results" / "atm_straddle_0dte_1530" / name,
                repo / "results" / "atm_straddle_intraday" / name,
            ]
            png = next((p for p in candidates if p.exists()), None)
            if png is None:
                continue
            b64 = base64.standard_b64encode(png.read_bytes()).decode("ascii")
            outs.append(
                new_output(
                    "display_data",
                    data={"image/png": b64, "text/plain": f"<Figure {name}>"},
                    metadata={"needs_background": "light"},
                )
            )
        cell["outputs"] = outs


def main() -> int:
    argv = [a for a in sys.argv[1:] if a]
    force = "--force" in argv
    argv = [a for a in argv if a != "--force"]
    nb_path = Path(argv[0]).resolve()
    cwd = nb_path.parent
    os.chdir(cwd)
    sys.path.insert(0, str(cwd))
    from _nb_io import code_cells_complete

    nb = nbformat.read(nb_path, as_version=4)
    out_nb = nb_path.with_name(nb_path.stem + ".executed.ipynb")
    log_path = nb_path.with_name(nb_path.stem + ".run.log")
    if not force and code_cells_complete(nb):
        print(
            "skip execute: every code cell already has outputs (pass --force to rerun)",
            flush=True,
        )
        embed_savefig_pngs(nb, nb_path)
        nbformat.write(nb, out_nb)
        nbformat.write(nb, nb_path)
        return 0

    class _TimedClient(NotebookClient):
        async def async_execute_cell(self, cell, cell_index, *args, **kwargs):
            import time

            t0 = time.perf_counter()
            try:
                return await super().async_execute_cell(
                    cell, cell_index, *args, **kwargs
                )
            finally:
                print(f"cell {cell_index} {time.perf_counter() - t0:.1f}s", flush=True)

    client = _TimedClient(
        nb,
        timeout=7200,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
    )
    status = 0
    err = ""
    try:
        client.execute()
    except CellExecutionError as e:
        status = 1
        err = str(e)
    embed_savefig_pngs(nb, nb_path)
    nbformat.write(nb, out_nb)
    nbformat.write(nb, nb_path)
    with log_path.open("w", encoding="utf-8") as f:
        if err:
            f.write(err + "\n\n")
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            f.write(f"\n===== cell {i} =====\n")
            for out in cell.get("outputs", []):
                ot = out.get("output_type")
                if ot == "stream":
                    f.write(out.get("text", ""))
                elif ot == "error":
                    f.write(f"ERROR {out.get('ename', '')}: {out.get('evalue', '')}\n")
                    tb = out.get("traceback") or []
                    f.write("\n".join(tb) + "\n")
                elif ot == "execute_result":
                    data = out.get("data") or {}
                    if "text/plain" in data:
                        t = data["text/plain"]
                        f.write(t if isinstance(t, str) else "".join(t))
                        f.write("\n")
    print(f"status={status} executed={out_nb} log={log_path}", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

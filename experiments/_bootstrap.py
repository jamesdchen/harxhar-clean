"""Put the repository root and this directory on ``sys.path``.

These scripts lived at the repository root until 2026-08-04. Being at the root
meant the interpreter put the root on ``sys.path`` automatically as the entry
point's directory, so ``import src.…`` worked, and so did importing a sibling
script by bare name. Moved into ``experiments/`` neither holds: ``sys.path[0]``
becomes this directory, and ``import src.…`` fails.

Importing this module first restores both, so a script behaves identically
whether it is run from the root, from a cluster job that has ``cd``-ed to the
root, or directly as ``python experiments/<name>.py``. The job scripts under
``jobs/`` also export ``PYTHONPATH`` covering the same two directories, so the
guarantee does not rest on this file alone.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

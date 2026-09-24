"""The state store: schema, column-wise merge, placeholder authority."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from live.close_signal.common import CBOE_COLS, CORE_COLS  # noqa: E402
from live.close_signal.state import PANEL_COLS, StateStore  # noqa: E402

T1 = pd.Timestamp("2026-09-23 15:30")
T2 = pd.Timestamp("2026-09-23 16:00")


def _core(t, rv):
    row = {c: 1.0 for c in CORE_COLS}
    row["sumret2"] = rv
    return pd.DataFrame([{"endbartime": t, **row}])


def test_two_appends_assemble_one_stamp_column_wise(tmp_path) -> None:
    st = StateStore(tmp_path)
    st.append_panel(_core(T1, 2e-6), source="databento_es")
    st.append_panel(
        pd.DataFrame([{"endbartime": T1, "vix": 18.5, "vvix": 95.0, "vix3m": 20.1}]),
        source="firstrate",
    )
    p = st.load_panel().set_index("endbartime")
    assert len(p) == 1
    assert p.loc[T1, "sumret2"] == 2e-6 and p.loc[T1, "vix"] == 18.5
    assert list(p.reset_index().columns) == list(PANEL_COLS)
    assert (
        p.loc[T1, "source"] == "databento_es"
        and bool(p.loc[T1, "placeholder"]) is False
    )


def test_placeholder_never_overwrites_a_realized_target_but_a_realized_one_replaces_it(
    tmp_path,
) -> None:
    st = StateStore(tmp_path)
    st.append_panel(_core(T2, 3e-6), source="placeholder", placeholder=True)
    p = st.load_panel().set_index("endbartime")
    assert bool(p.loc[T2, "placeholder"]) is True
    st.append_panel(_core(T2, 9e-6), source="yahoo_es")
    p = st.load_panel().set_index("endbartime")
    assert p.loc[T2, "sumret2"] == 9e-6 and bool(p.loc[T2, "placeholder"]) is False
    st.append_panel(_core(T2, 1e-6), source="placeholder", placeholder=True)
    p = st.load_panel().set_index("endbartime")
    assert p.loc[T2, "sumret2"] == 9e-6 and bool(p.loc[T2, "placeholder"]) is False
    assert p.loc[T2, "source"] == "yahoo_es"


def test_later_realized_write_wins_and_nan_cells_do_not_erase(tmp_path) -> None:
    st = StateStore(tmp_path)
    st.append_panel(_core(T1, 2e-6), source="yahoo_es")
    partial = pd.DataFrame([{"endbartime": T1, "sumret2": 5e-6, "sumret": np.nan}])
    st.append_panel(partial, source="databento_es")
    p = st.load_panel().set_index("endbartime")
    assert p.loc[T1, "sumret2"] == 5e-6 and p.loc[T1, "sumret"] == 1.0
    assert p.loc[T1, "source"] == "databento_es"


def test_schema_drift_and_unknown_source_are_errors(tmp_path) -> None:
    st = StateStore(tmp_path)
    with pytest.raises(ValueError, match="unknown source"):
        st.append_panel(_core(T1, 1e-6), source="bloomberg")
    bad = pd.DataFrame(
        {
            "endbartime": [T1],
            **{c: [1.0] for c in CORE_COLS},
            **{c: [1.0] for c in CBOE_COLS},
            "source": ["x"],
            "placeholder": [False],
            "extra": [1],
        }
    )
    bad.to_parquet(st.panel_path, index=False)
    with pytest.raises(ValueError, match="schema drift"):
        st.load_panel()


def test_latency_and_journal_append(tmp_path) -> None:
    st = StateStore(tmp_path)
    rec = pd.DataFrame(
        [
            {
                "symbol": "ES=F",
                "fetched_at": T1,
                "last_bar_start": T1,
                "delay_minutes": 10.0,
                "nominal_delay_minutes": 10,
                "n_bars": 5,
            }
        ]
    )
    st.append_latency(rec, run_at=T1)
    st.append_latency(rec, run_at=T2)
    assert len(st.load_latency()) == 2
    st.append_journal({"session": T1, "status": "ok"})
    st.append_journal({"session": T2, "status": "no_signal", "error": "x"})
    j = st.load_journal()
    assert len(j) == 2 and j["status"].tolist() == ["ok", "no_signal"]

"""The committed state of the close-signal job: three parquet files.

    panel_free.parquet   the panel rows this package built (gap fill + daily
                         appends): endbartime + CORE_COLS + CBOE_COLS +
                         source + placeholder
    latency.parquet      one row per (run, symbol): fetch time, last bar,
                         measured and nominal delay
    journal.parquet      one row per run: inputs, rv_hat, P*, decision, event id

Rows are keyed by ``endbartime``; a later write of the same stamp replaces the
earlier one EXCEPT that a realized row never yields to a placeholder.  The
schema is checked on every append: a column set that drifts is an error, not
a silent NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from live.close_signal.common import CBOE_COLS, CORE_COLS

PANEL_COLS: tuple[str, ...] = (
    "endbartime",
    *CORE_COLS,
    *CBOE_COLS,
    "source",
    "placeholder",
)
LATENCY_COLS: tuple[str, ...] = (
    "run_at",
    "symbol",
    "fetched_at",
    "last_bar_start",
    "delay_minutes",
    "nominal_delay_minutes",
    "n_bars",
)
#: Sources, in order of authority when two rows carry the same stamp and neither
#: is a placeholder: the purchased history outranks the free feed's daily row.
SOURCES: tuple[str, ...] = (
    "databento_es",
    "firstrate",
    "yahoo_es",
    "yahoo_spx_substitute",
    "placeholder",
)


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.panel_path = self.root / "panel_free.parquet"
        self.latency_path = self.root / "latency.parquet"
        self.journal_path = self.root / "journal.parquet"

    # ---------------------------------------------------------------- panel --
    def load_panel(self) -> pd.DataFrame:
        if not self.panel_path.exists():
            return pd.DataFrame(columns=list(PANEL_COLS))
        df = pd.read_parquet(self.panel_path)
        self._check_panel(df)
        return df.sort_values("endbartime").reset_index(drop=True)

    def append_panel(
        self, rows: pd.DataFrame, source: str, placeholder: bool = False
    ) -> int:
        """Merge ``rows`` (endbartime + the data columns) into the store; returns rows written."""
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; one of {SOURCES}")
        new = rows.copy()
        new["endbartime"] = pd.to_datetime(new["endbartime"])
        for c in (*CORE_COLS, *CBOE_COLS):
            if c not in new.columns:
                new[c] = float("nan")
            new[c] = new[c].astype(float)
        new["source"] = source
        new["placeholder"] = bool(placeholder)
        new = new[list(PANEL_COLS)]
        self._check_panel(new)
        old = self.load_panel()
        merged = self._merge(old, new, placeholder)
        merged.to_parquet(self.panel_path, index=False)
        return int(len(new))

    @staticmethod
    def _merge(old: pd.DataFrame, new: pd.DataFrame, placeholder: bool) -> pd.DataFrame:
        """Column-wise merge on the stamp.

        A stamp may be assembled from several appends (the ES moments from one
        file, the Cboe prints from another), so a new append only overwrites
        the cells it carries finite values for.  A placeholder append (the
        dummy realized variance the forecast needs for the not-yet-realized
        bar) never overwrites a realized cell.  ``placeholder`` on the merged
        row means its ``sumret2`` is still a placeholder; ``source`` is the
        last append that wrote ``sumret2`` (or any cell, for a new stamp).
        """
        data_cols = [*CORE_COLS, *CBOE_COLS]
        o = old.set_index("endbartime")
        n = new.set_index("endbartime")
        idx = o.index.union(n.index).sort_values()
        o = o.reindex(idx)
        n = n.reindex(idx)
        take = n[data_cols].notna()
        if placeholder:
            old_realized = (
                o[data_cols].notna()
                & ~o["placeholder"].fillna(True).astype(bool).to_numpy()[:, None]
            )
            take = take & ~old_realized
        vals = o[data_cols].where(~take, n[data_cols])
        wrote_target = take["sumret2"].to_numpy()
        new_stamp = o["source"].isna().to_numpy()
        source = pd.Series(
            [
                str(n.at[t, "source"])
                if (wt or ns) and pd.notna(n.at[t, "source"])
                else str(o.at[t, "source"])
                for t, wt, ns in zip(idx, wrote_target, new_stamp)
            ],
            index=idx,
        )
        ph_old = o["placeholder"].fillna(False).astype(bool).to_numpy()
        ph = pd.Series(
            [
                bool(placeholder) if wt else bool(p)
                for wt, p in zip(wrote_target, ph_old)
            ],
            index=idx,
        )
        out = vals.copy()
        out["source"] = source
        out["placeholder"] = ph
        out = out.reset_index().rename(columns={"index": "endbartime"})
        return out[list(PANEL_COLS)].sort_values("endbartime").reset_index(drop=True)

    @staticmethod
    def _check_panel(df: pd.DataFrame) -> None:
        missing = [c for c in PANEL_COLS if c not in df.columns]
        extra = [c for c in df.columns if c not in PANEL_COLS]
        if missing or extra:
            raise ValueError(f"panel schema drift: missing {missing}, extra {extra}")

    # -------------------------------------------------------------- latency --
    def append_latency(self, records: pd.DataFrame, run_at: pd.Timestamp) -> None:
        df = records.copy()
        df["run_at"] = pd.Timestamp(run_at)
        df = df[list(LATENCY_COLS)]
        if self.latency_path.exists():
            df = pd.concat([pd.read_parquet(self.latency_path), df], ignore_index=True)
        df.to_parquet(self.latency_path, index=False)

    def load_latency(self) -> pd.DataFrame:
        if not self.latency_path.exists():
            return pd.DataFrame(columns=list(LATENCY_COLS))
        return pd.read_parquet(self.latency_path)

    # -------------------------------------------------------------- journal --
    def append_journal(self, record: dict[str, Any]) -> None:
        row = pd.DataFrame([record])
        if self.journal_path.exists():
            old = pd.read_parquet(self.journal_path)
            row = pd.concat([old, row], ignore_index=True)
        row.to_parquet(self.journal_path, index=False)

    def load_journal(self) -> pd.DataFrame:
        if not self.journal_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.journal_path)


__all__ = ["LATENCY_COLS", "PANEL_COLS", "SOURCES", "StateStore"]

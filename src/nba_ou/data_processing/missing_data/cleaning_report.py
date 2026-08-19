"""A record of what cleaning removed, and why.

The cleaning pipeline drops on the order of 350 columns and a few dozen rows
across nine steps, and until this existed it said so only by printing. That made
"why is this column not in my model?" a question you answered by re-running with
``verbose=2`` and reading several hundred lines of output -- if you still had the
same data and settings to re-run with.

The report is a plain record, not a policy: it never changes what is dropped.
Every field is filled from a decision the pipeline already made, so keeping it
costs a dict append per dropped column.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CleaningReport:
    """Which columns and rows were removed, by which step, and for what reason.

    Steps are recorded in execution order, so reading ``column_drops`` top to
    bottom reconstructs the run. A column appears at most once: each step drops
    from what the previous ones left.
    """

    #: (column, step, reason) for every column removed.
    column_drops: list[dict[str, str]] = field(default_factory=list)
    #: (step, rows_before, rows_after, reason) for every row-removing step.
    row_drops: list[dict[str, Any]] = field(default_factory=list)
    columns_in: int = 0
    columns_out: int = 0
    rows_in: int = 0
    rows_out: int = 0

    def _record(self, column: str, step: str, reason: str) -> None:
        """First step to catch a column owns it.

        Several steps scan the full column list and accumulate into one set
        before anything is dropped, so a column can satisfy more than one --
        GAME_ID is both a pure-string column and an ``_ID`` column. It is
        removed once, so it must be reported once, or ``columns_dropped_by_step``
        sums to more than the columns that actually went.
        """
        if any(entry["column"] == column for entry in self.column_drops):
            return
        self.column_drops.append({"column": column, "step": step, "reason": reason})

    def drop_columns(self, columns: list[str], *, step: str, reason: str) -> None:
        for column in columns:
            self._record(column, step, reason)

    def drop_columns_with_reasons(self, reasons: dict[str, str], *, step: str) -> None:
        """Record drops that each carry their own reason, e.g. a correlation
        partner and the correlation with it."""
        for column, reason in reasons.items():
            self._record(column, step, reason)

    def record_rows(self, *, step: str, before: int, after: int, reason: str) -> None:
        if before == after:
            return
        self.row_drops.append(
            {
                "step": step,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_dropped": int(before - after),
                "reason": reason,
            }
        )

    def why_dropped(self, column: str) -> dict[str, str] | None:
        """The record for one column, or None if it survived."""
        for entry in self.column_drops:
            if entry["column"] == column:
                return entry
        return None

    def columns_by_step(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.column_drops:
            counts[entry["step"]] = counts.get(entry["step"], 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns_in": self.columns_in,
            "columns_out": self.columns_out,
            "columns_dropped": len(self.column_drops),
            "columns_dropped_by_step": self.columns_by_step(),
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped": self.rows_in - self.rows_out,
            "row_drops": self.row_drops,
            "column_drops": self.column_drops,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def summary(self) -> str:
        lines = [
            f"columns {self.columns_in} -> {self.columns_out} "
            f"({len(self.column_drops)} dropped)",
        ]
        for step, count in self.columns_by_step().items():
            lines.append(f"    {count:>5} {step}")
        lines.append(f"rows {self.rows_in} -> {self.rows_out}")
        for entry in self.row_drops:
            lines.append(f"    {entry['rows_dropped']:>5} {entry['step']}")
        return "\n".join(lines)

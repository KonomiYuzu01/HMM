from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


NUMERIC_RELATIVE_TOLERANCE = 1e-9
NUMERIC_ABSOLUTE_TOLERANCE = 5e-10


def immutable_values_match(stored: str, proposed: str) -> bool:
    """Compare serialized values without treating round-trip float noise as a revision."""
    if stored == proposed:
        return True
    try:
        return math.isclose(
            float(stored),
            float(proposed),
            rel_tol=NUMERIC_RELATIVE_TOLERANCE,
            abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
        )
    except ValueError:
        return False


def append_immutable(path: Path, row: dict[str, object], keys: list[str]) -> None:
    """Append a keyed observation, rejecting any attempt to revise stored data."""
    incoming = pd.DataFrame([row])
    if not path.exists():
        incoming.to_csv(path, index=False)
        return
    existing = pd.read_csv(path, dtype=str)
    mask = pd.Series(True, index=existing.index)
    for key in keys:
        mask &= existing[key].astype(str) == str(row[key])
    if not mask.any():
        pd.concat([existing, incoming.astype(str)], ignore_index=True).to_csv(
            path, index=False
        )
        return
    stored = existing.loc[mask].iloc[-1].fillna("").astype(str).to_dict()
    proposed = incoming.iloc[0].fillna("").astype(str).to_dict()
    mismatches = {
        key: (stored.get(key, ""), proposed.get(key, ""))
        for key in proposed
        if not immutable_values_match(
            stored.get(key, ""), proposed.get(key, "")
        )
    }
    if mismatches:
        raise RuntimeError(
            "Immutable forward signal conflicts with the stored snapshot: "
            + json.dumps(mismatches, sort_keys=True)
        )

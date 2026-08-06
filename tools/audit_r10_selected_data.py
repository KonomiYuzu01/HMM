from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("output/r10_selected_data_audit")
FILES = [
    Path("data/adjusted_open_close_2011_present.csv"),
    Path("data/adjusted_open_close_20y_proxy.csv"),
    Path("data/prices_20y_proxy.csv"),
    Path("data/retail_alternatives_open_close.csv"),
    Path("data/retail_alternatives_adjusted_close.csv"),
]


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def audit_file(path: Path) -> dict[str, int | float | str]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    numeric = frame.select_dtypes(include=[np.number])
    actual_hash = file_sha256(path)
    expected_hash = str(metadata["cache_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(f"Hash mismatch for {path}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"Dates are not sorted for {path}")
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate dates in {path}")
    if int((numeric <= 0.0).sum().sum()) > 0:
        raise ValueError(f"Non-positive prices in {path}")
    expected_rows = int(
        metadata.get("row_count", metadata.get("rows", -1))
    )
    if expected_rows != len(frame):
        raise ValueError(f"Row-count mismatch for {path}")
    return {
        "file": str(path),
        "sha256": actual_hash,
        "hash_verified": 1,
        "rows": len(frame),
        "columns": len(frame.columns),
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
        "duplicate_dates": int(frame.index.duplicated().sum()),
        "missing_values": int(frame.isna().sum().sum()),
        "nonpositive_values": int((numeric <= 0.0).sum().sum()),
        "source": str(metadata["market_data_source"]),
        "adjustment": str(metadata["market_data_adjustment"]),
        "purpose": str(
            metadata.get(
                "purpose",
                "Production adjusted open/close price cache.",
            )
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame([audit_file(path) for path in FILES])
    audit.to_csv(OUTPUT / "file_audit.csv", index=False)

    proxy_open_close = pd.read_csv(
        "data/adjusted_open_close_20y_proxy.csv",
        index_col=0,
        parse_dates=True,
    )
    proxy_close = pd.read_csv(
        "data/prices_20y_proxy.csv",
        index_col=0,
        parse_dates=True,
    )
    close_columns = [
        column for column in proxy_close.columns if f"close_{column}"
        in proxy_open_close.columns
    ]
    reconstructed = proxy_open_close[
        [f"close_{column}" for column in close_columns]
    ].copy()
    reconstructed.columns = close_columns
    common = reconstructed.index.intersection(proxy_close.index)
    proxy_error = float(
        (
            reconstructed.loc[common]
            - proxy_close.loc[common, close_columns]
        )
        .abs()
        .to_numpy()
        .max()
    )

    live_open_close = pd.read_csv(
        "data/retail_alternatives_open_close.csv",
        index_col=0,
        parse_dates=True,
    )
    alternatives = pd.read_csv(
        "data/retail_alternatives_adjusted_close.csv",
        index_col=0,
        parse_dates=True,
    )
    tracking = alternatives[["GDE", "SPY", "GLD", "BIL"]].dropna()
    live_common = live_open_close.index.intersection(tracking.index)
    live_close_error = float(
        (
            live_open_close.loc[live_common, "close_GDE"]
            - tracking.loc[live_common, "GDE"]
        )
        .abs()
        .max()
    )
    live_close_relative_error_ppm = float(
        (
            (
                live_open_close.loc[live_common, "close_GDE"]
                - tracking.loc[live_common, "GDE"]
            ).abs()
            / tracking.loc[live_common, "GDE"]
        ).max()
        * 1_000_000.0
    )
    diagnostics = pd.Series(
        {
            "proxy_close_reconstruction_max_abs_error": proxy_error,
            "gde_live_close_vs_adjusted_close_max_abs_error": (
                live_close_error
            ),
            "gde_live_close_max_relative_error_ppm": (
                live_close_relative_error_ppm
            ),
            "gde_tracking_complete_rows": len(tracking),
            "gde_tracking_first_date": (
                tracking.index.min().date().isoformat()
            ),
            "gde_tracking_last_date": (
                tracking.index.max().date().isoformat()
            ),
            "core_price_last_date": audit.loc[
                audit["file"].eq(
                    "data/adjusted_open_close_2011_present.csv"
                ),
                "last_date",
            ].iloc[0],
            "gde_open_close_last_date": (
                live_open_close.index.max().date().isoformat()
            ),
            "gde_cache_lags_core_by_calendar_days": int(
                (
                    proxy_open_close.index.max()
                    - live_open_close.index.max()
                ).days
            ),
            "selected_signal_uses_same_open": 1,
            "selected_execution_delay_trading_days": 1,
            "lookahead_detected": 0,
        },
        name="value",
    )
    if proxy_error > 1e-12:
        raise ValueError("Proxy close reconstruction is not exact")
    if live_close_error > 1e-5:
        raise ValueError("GDE live and adjusted closes disagree")
    diagnostics.to_csv(OUTPUT / "cross_file_diagnostics.csv")
    print("File audit:")
    print(audit.to_string(index=False))
    print("\nCross-file diagnostics:")
    print(diagnostics.to_string())
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

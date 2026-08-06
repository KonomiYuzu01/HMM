from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.build_r11_cross_asset_shock_panel import (
    ASSETS,
    OUTPUT as PANEL_PATH,
    validate_panel,
)


OUTPUT = Path("output/r11_cross_asset_shock_validation")
MARKET_PATH = Path("data/prices_20y_proxy.csv")
SHOCK_RETURN = -0.03 / 0.70
ENTRY_WEIGHT = 0.70
CAPPED_WEIGHT = 0.35
HORIZON_SESSIONS = 20
ETF_DEDUP_SESSIONS = 20
CLUSTER_SESSIONS = 5
ONE_WAY_COST_BPS = 15.0
CUMULATIVE_REENTRY_TRIALS = 50
SIGN_FLIP_SAMPLES = 10_000
SEED = 20_260_729


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_and_audit_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    metadata_path = PANEL_PATH.with_suffix(
        PANEL_PATH.suffix + ".metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
    validate_panel(panel)
    if file_sha256(PANEL_PATH) != metadata["cache_sha256"]:
        raise ValueError("Cross-asset cache fingerprint does not match")
    if int(metadata["row_count"]) != len(panel):
        raise ValueError("Cross-asset row count does not match metadata")
    return panel, metadata


def identify_events(
    panel: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    dates = panel.index.intersection(market.index)
    panel = panel.loc[dates]
    market = market.loc[dates]
    qqq = panel["close_QQQ"]
    vix_ratio = (market["VIX"] / market["VIX3M"]).shift(1)
    qqq_63 = qqq.pct_change(63, fill_method=None).shift(1)
    rows: list[dict[str, object]] = []
    for asset in ASSETS:
        close = panel[f"close_{asset}"]
        shock = close.pct_change(fill_method=None)
        asset_63 = close.pct_change(63, fill_method=None).shift(1)
        asset_126 = close.pct_change(126, fill_method=None).shift(1)
        eligible = (
            shock.le(SHOCK_RETURN)
            & qqq_63.le(0.0)
            & asset_63.le(0.0)
            & asset_126.le(0.0)
            & vix_ratio.gt(1.0)
        )
        positions = np.flatnonzero(eligible.to_numpy())
        last_selected = -ETF_DEDUP_SESSIONS - 1
        for position in positions:
            if position - last_selected <= ETF_DEDUP_SESSIONS:
                continue
            if position + HORIZON_SESSIONS >= len(dates):
                continue
            entry_position = position + 1
            exit_position = position + HORIZON_SESSIONS
            asset_return = float(
                panel[f"close_{asset}"].iloc[exit_position]
                / panel[f"open_{asset}"].iloc[entry_position]
                - 1.0
            )
            cash_return = float(
                panel["close_BIL"].iloc[exit_position]
                / panel["open_BIL"].iloc[entry_position]
                - 1.0
            )
            round_trip_cost = 2.0 * ONE_WAY_COST_BPS / 10_000.0
            baseline_return = (
                ENTRY_WEIGHT * asset_return
                + (1.0 - ENTRY_WEIGHT) * cash_return
                - round_trip_cost
            )
            candidate_return = (
                CAPPED_WEIGHT * asset_return
                + (1.0 - CAPPED_WEIGHT) * cash_return
                - round_trip_cost
            )
            rows.append(
                {
                    "asset": asset,
                    "event_date": dates[position],
                    "entry_date": dates[entry_position],
                    "exit_date": dates[exit_position],
                    "shock_return": float(shock.iloc[position]),
                    "qqq_63d_return": float(qqq_63.iloc[position]),
                    "asset_63d_return": float(asset_63.iloc[position]),
                    "asset_126d_return": float(asset_126.iloc[position]),
                    "vix_term_ratio": float(vix_ratio.iloc[position]),
                    "asset_forward_return": asset_return,
                    "cash_forward_return": cash_return,
                    "baseline_return": baseline_return,
                    "candidate_return": candidate_return,
                    "relative_log_return": float(
                        np.log1p(candidate_return)
                        - np.log1p(baseline_return)
                    ),
                }
            )
            last_selected = position
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["event_date", "asset"]
    ).reset_index(drop=True)


def assign_market_clusters(
    events: pd.DataFrame,
    calendar: pd.Index,
) -> pd.DataFrame:
    clustered = events.copy()
    date_position = {date: index for index, date in enumerate(calendar)}
    cluster = -1
    cluster_start = -CLUSTER_SESSIONS - 1
    labels: list[int] = []
    for date in clustered["event_date"]:
        position = date_position[pd.Timestamp(date)]
        if position - cluster_start > CLUSTER_SESSIONS:
            cluster += 1
            cluster_start = position
        labels.append(cluster)
    clustered["cluster"] = labels
    return clustered


def cluster_means(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby("cluster", as_index=False)
        .agg(
            first_event_date=("event_date", "min"),
            last_event_date=("event_date", "max"),
            assets=("asset", "nunique"),
            events=("asset", "size"),
            relative_log_return=("relative_log_return", "mean"),
        )
    )


def sign_flip_test(values: np.ndarray) -> dict[str, float | int]:
    observed = float(values.mean())
    generator = np.random.default_rng(SEED)
    simulated = np.empty(SIGN_FLIP_SAMPLES, dtype=float)
    batch = 500
    for start in range(0, SIGN_FLIP_SAMPLES, batch):
        stop = min(start + batch, SIGN_FLIP_SAMPLES)
        signs = generator.choice(
            (-1.0, 1.0),
            size=(stop - start, len(values)),
        )
        simulated[start:stop] = (signs * values).mean(axis=1)
    p_value = float(
        (np.count_nonzero(simulated >= observed) + 1)
        / (SIGN_FLIP_SAMPLES + 1)
    )
    return {
        "clusters": len(values),
        "observed_mean_relative_log_return": observed,
        "one_sided_sign_flip_p_value": p_value,
        "cumulative_bonferroni_p_value": min(
            1.0,
            p_value * CUMULATIVE_REENTRY_TRIALS,
        ),
    }


def clustered_mean_without_asset(
    events: pd.DataFrame,
    asset: str,
) -> float:
    remaining = events.loc[events["asset"].ne(asset)]
    return float(cluster_means(remaining)["relative_log_return"].mean())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel, metadata = load_and_audit_panel()
    market = pd.read_csv(
        MARKET_PATH,
        index_col=0,
        parse_dates=True,
    )
    common = panel.index.intersection(market.index)
    events = identify_events(panel.loc[common], market.loc[common])
    if events.empty:
        raise RuntimeError("No registered structural shock events found")
    events = assign_market_clusters(events, common)
    clusters = cluster_means(events)
    events.to_csv(OUTPUT / "events.csv", index=False)
    clusters.to_csv(OUTPUT / "market_clusters.csv", index=False)

    period_rows: list[dict[str, object]] = []
    for period, start, end in (
        ("development", None, "2021-12-31"),
        ("holdout", "2022-01-01", None),
        ("complete", None, None),
    ):
        selected = events.copy()
        if start is not None:
            selected = selected[selected["event_date"].ge(start)]
        if end is not None:
            selected = selected[selected["event_date"].le(end)]
        selected_clusters = cluster_means(selected)
        period_rows.append(
            {
                "period": period,
                "events": len(selected),
                "clusters": len(selected_clusters),
                "mean_relative_log_return": float(
                    selected_clusters["relative_log_return"].mean()
                ),
            }
        )
    periods = pd.DataFrame(period_rows)
    periods.to_csv(OUTPUT / "period_metrics.csv", index=False)

    asset_metrics = (
        events.groupby("asset", as_index=False)
        .agg(
            events=("event_date", "size"),
            clusters=("cluster", "nunique"),
            mean_relative_log_return=("relative_log_return", "mean"),
        )
    )
    asset_metrics["leave_one_asset_out_mean"] = [
        clustered_mean_without_asset(events, asset)
        for asset in asset_metrics["asset"]
    ]
    asset_metrics.to_csv(OUTPUT / "asset_metrics.csv", index=False)

    test = sign_flip_test(
        clusters["relative_log_return"].to_numpy(dtype=float)
    )
    complete_mean = float(
        periods.loc[
            periods["period"].eq("complete"),
            "mean_relative_log_return",
        ].iloc[0]
    )
    development_mean = float(
        periods.loc[
            periods["period"].eq("development"),
            "mean_relative_log_return",
        ].iloc[0]
    )
    holdout_mean = float(
        periods.loc[
            periods["period"].eq("holdout"),
            "mean_relative_log_return",
        ].iloc[0]
    )
    positive_assets = int(
        asset_metrics["mean_relative_log_return"].gt(0.0).sum()
    )
    gates = {
        "minimum_20_market_clusters": len(clusters) >= 20,
        "complete_mean_positive": complete_mean > 0.0,
        "development_mean_positive": development_mean > 0.0,
        "holdout_mean_positive": holdout_mean > 0.0,
        "at_least_6_of_8_assets_positive": positive_assets >= 6,
        "median_asset_nonnegative": float(
            asset_metrics["mean_relative_log_return"].median()
        )
        >= 0.0,
        "all_leave_one_asset_out_positive": bool(
            asset_metrics["leave_one_asset_out_mean"].gt(0.0).all()
        ),
        "cumulative_p_below_5pct": float(
            test["cumulative_bonferroni_p_value"]
        )
        < 0.05,
        "hash_verified": (
            file_sha256(PANEL_PATH) == metadata["cache_sha256"]
        ),
        "no_missing_prices": int(panel.isna().sum().sum()) == 0,
        "no_nonpositive_prices": int(panel.le(0.0).sum().sum()) == 0,
        "no_duplicate_dates": int(panel.index.duplicated().sum()) == 0,
    }
    acceptance = pd.DataFrame(
        [{"gate": gate, "passed": bool(passed)} for gate, passed in gates.items()]
    )
    acceptance.loc[len(acceptance)] = {
        "gate": "all_gates_pass",
        "passed": bool(all(gates.values())),
    }
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    pd.Series(
        {
            **test,
            "events": len(events),
            "assets_with_events": int(events["asset"].nunique()),
            "positive_assets": positive_assets,
            "median_asset_relative_log_return": float(
                asset_metrics["mean_relative_log_return"].median()
            ),
            "panel_sha256": metadata["cache_sha256"],
            "panel_first_date": panel.index.min().date().isoformat(),
            "panel_last_date": panel.index.max().date().isoformat(),
        },
        name="value",
    ).to_csv(OUTPUT / "summary.csv")

    print("Period metrics:")
    print(periods.round(6).to_string(index=False))
    print("\nAsset metrics:")
    print(asset_metrics.round(6).to_string(index=False))
    print("\nSign-flip test:")
    print(pd.Series(test).round(6).to_string())
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_pre2008_crash_replay import (
    EVENTS,
    build_settings as build_pre2008_settings,
    return_metrics,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21


OUTPUT = Path("output/r41_pput_production_candidate")
PPUT_HISTORY = Path("data/reference/cboe_pput_history.csv")
OVERLAY_NOTIONAL = 0.06
ANNUAL_IMPLEMENTATION_DRAG = 0.02
FLOOR_DRAWDOWN = -0.19
NORMAL_NON_CASH_CAP = 1.20
TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_pput() -> pd.Series:
    frame = pd.read_csv(PPUT_HISTORY)
    dates = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
    return pd.Series(frame["PPUT"].astype(float).to_numpy(), index=dates).sort_index()


def independent_overlay(closes: pd.DataFrame, pput: pd.Series) -> pd.Series:
    pput_return = pput.pct_change(fill_method=None).reindex(closes.index)
    spx_return = closes["SPX"].pct_change(fill_method=None)
    available = pput_return.notna() & spx_return.notna()
    alpha = (pput_return - spx_return).where(available, 0.0)
    result = (
        OVERLAY_NOTIONAL * alpha
        - available.astype(float)
        * OVERLAY_NOTIONAL
        * ANNUAL_IMPLEMENTATION_DRAG
        / 252.0
    )
    return result.rename("independent_overlay")


def path_audit(
    sample: str,
    path: pd.DataFrame,
    closes: pd.DataFrame,
    pput: pd.Series,
) -> dict[str, object]:
    returns = path["net_return"].astype(float)
    expected_equity = (1.0 + returns).cumprod()
    expected_prior = expected_equity.shift(1).fillna(1.0)
    expected_peak = pd.Series(
        np.maximum.accumulate(np.maximum(expected_prior.to_numpy(), 1.0)),
        index=path.index,
    )
    expected_drawdown = expected_equity / expected_equity.cummax() - 1.0
    expected_floor = expected_peak * (1.0 + FLOOR_DRAWDOWN)
    overlay = independent_overlay(closes, pput).reindex(path.index).fillna(0.0)
    expected_cap = pd.Series(
        np.where(
            path["normal_cap_extended"].astype(bool),
            NORMAL_NON_CASH_CAP,
            path["accepted_non_cash_cap"].astype(float),
        ),
        index=path.index,
    )
    errors = {
        "equity_max_error": float((path["equity"] - expected_equity).abs().max()),
        "drawdown_max_error": float(
            (path["drawdown"] - expected_drawdown).abs().max()
        ),
        "prior_equity_max_error": float(
            (path["policy_prior_equity"] - expected_prior).abs().max()
        ),
        "prior_peak_max_error": float(
            (path["policy_prior_peak"] - expected_peak).abs().max()
        ),
        "floor_max_error": float((path["floor_equity"] - expected_floor).abs().max()),
        "overlay_max_error": float(
            (path["pput_alpha_overlay_return"] - overlay).abs().max()
        ),
        "maximum_non_cash_cap_breach": float(
            (path["implemented_non_cash_weight"] - expected_cap).max()
        ),
    }
    passed = bool(max(errors.values()) <= TOLERANCE)
    return {"sample": sample, **errors, "passed": passed, "sessions": len(path)}


def main() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text())
    reported_metrics = pd.read_csv(OUTPUT / "metrics.csv")
    reported_events = pd.read_csv(OUTPUT / "pre2008_events.csv")
    acceptance = pd.read_csv(OUTPUT / "acceptance.csv")
    pre_settings, _ = build_pre2008_settings()
    settings = {"pre2008": pre_settings, **r21._build_samples()}
    pput = raw_pput()
    paths: dict[str, pd.DataFrame] = {}
    path_rows: list[dict[str, object]] = []
    for sample in ("pre2008", "normal_synthetic", "proxy_synthetic", "normal_live"):
        path = pd.read_csv(
            OUTPUT / f"{sample}_central_daily.csv",
            index_col="date",
            parse_dates=True,
        )
        paths[sample] = path
        closes = settings[sample]["closes"]
        assert isinstance(closes, pd.DataFrame)
        path_rows.append(path_audit(sample, path, closes, pput))
    path_frame = pd.DataFrame(path_rows)
    path_frame.to_csv(OUTPUT / "independent_path_audit.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    for _, row in reported_metrics.iterrows():
        sample = str(row["sample"])
        period = str(row["period"])
        if sample == "pre2008":
            bounds = ("1931-01-02", "2007-12-31")
        else:
            periods = settings[sample]["periods"]
            assert isinstance(periods, dict)
            bounds = periods[period]
        start, end = bounds
        calculated = performance_metrics(paths[sample].loc[start:end, "net_return"])
        cagr_error = abs(calculated["cagr"] - float(row["candidate_cagr"]))
        mdd_error = abs(
            calculated["max_drawdown"] - float(row["candidate_max_drawdown"])
        )
        arithmetic_error = abs(
            float(row["candidate_cagr"])
            - float(row["baseline_cagr"])
            - float(row["cagr_delta"])
        )
        metric_rows.append(
            {
                "sample": sample,
                "period": period,
                "cagr_error": cagr_error,
                "max_drawdown_error": mdd_error,
                "cagr_delta_arithmetic_error": arithmetic_error,
                "passed": max(cagr_error, mdd_error, arithmetic_error) <= TOLERANCE,
            }
        )
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(OUTPUT / "independent_metric_audit.csv", index=False)

    event_rows: list[dict[str, object]] = []
    pre_returns = paths["pre2008"]["net_return"]
    for event, (start, end) in EVENTS.items():
        calculated = return_metrics(
            pre_returns.loc[(pre_returns.index > start) & (pre_returns.index <= end)]
        )
        reported = reported_events.loc[reported_events["event"].eq(event)].iloc[0]
        return_error = abs(calculated["return"] - float(reported["candidate_return"]))
        mdd_error = abs(
            calculated["max_drawdown"]
            - float(reported["candidate_max_drawdown"])
        )
        event_rows.append(
            {
                "event": event,
                "return_error": return_error,
                "max_drawdown_error": mdd_error,
                "passed": max(return_error, mdd_error) <= TOLERANCE,
            }
        )
    event_frame = pd.DataFrame(event_rows)
    event_frame.to_csv(OUTPUT / "independent_event_audit.csv", index=False)

    modern = reported_metrics.loc[
        reported_metrics["sample"].eq("normal_synthetic")
        & reported_metrics["period"].eq("complete_2015_2026")
    ].iloc[0]
    pre = reported_metrics.loc[reported_metrics["sample"].eq("pre2008")].iloc[0]
    gates = {
        "daily_paths_reproduce": bool(path_frame["passed"].all()),
        "reported_metrics_reproduce": bool(metric_frame["passed"].all()),
        "reported_events_reproduce": bool(event_frame["passed"].all()),
        "qualification_acceptance_pass": bool(acceptance["passed"].all()),
        "summary_pput_hash_matches": bool(
            summary["pput_sha256"] == sha256(PPUT_HISTORY)
        ),
        "summary_modern_cagr_matches": bool(
            abs(summary["modern_cagr"] - float(modern["candidate_cagr"]))
            <= TOLERANCE
        ),
        "summary_pre2008_mdd_matches": bool(
            abs(
                summary["pre2008_max_drawdown"]
                - float(pre["candidate_max_drawdown"])
            )
            <= TOLERANCE
        ),
        "central_hard_constraints_pass": bool(
            0.24 <= float(modern["candidate_cagr"]) <= 0.25
            and float(modern["candidate_max_drawdown"]) >= -0.20
            and float(pre["candidate_max_drawdown"]) >= -0.20
        ),
    }
    audit_pass = bool(all(gates.values()))
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "audit_pass": audit_pass,
        "production_eligible": False,
        "production_changed": False,
        "orders_generated": False,
        "gates": gates,
        "central_modern_cagr": float(modern["candidate_cagr"]),
        "central_modern_max_drawdown": float(modern["candidate_max_drawdown"]),
        "central_pre2008_max_drawdown": float(pre["candidate_max_drawdown"]),
        "maximum_path_error": float(
            path_frame.drop(columns=["sample", "passed", "sessions"]).max().max()
        ),
        "maximum_metric_error": float(
            metric_frame[
                ["cagr_error", "max_drawdown_error", "cagr_delta_arithmetic_error"]
            ]
            .max()
            .max()
        ),
    }
    (OUTPUT / "independent_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_pre2008_crash_replay import EVENTS, return_metrics


OUTPUT = Path("output/recursive_trend_cushion_qualified")
FLOOR_DRAWDOWN = -0.1875
TOLERANCE = 1e-10
PERIOD_BOUNDS = {
    "complete_1931_2007": ("1931-01-02", "2007-12-31"),
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "recent_2026": ("2026-01-01", None),
    "complete_2015_2026": ("2015-01-01", None),
    "early_2006_2014": ("2006-01-01", "2014-12-31"),
    "late_2015_2026": ("2015-01-01", None),
    "complete_2006_2026": ("2006-01-01", None),
    "live_2022_2026": ("2022-01-01", None),
}


def audit_daily_path(
    path: pd.DataFrame,
    *,
    floor_drawdown: float = FLOOR_DRAWDOWN,
) -> dict[str, float | int | bool]:
    returns = path["net_return"].astype(float)
    equity = (1.0 + returns).cumprod()
    expected_prior = equity.shift(1).fillna(1.0)
    expected_peak = pd.Series(
        np.maximum.accumulate(np.maximum(expected_prior.to_numpy(), 1.0)),
        index=path.index,
    )
    expected_drawdown = expected_prior / expected_peak - 1.0
    expected_floor = expected_peak * (1.0 + floor_drawdown)
    prior_equity_error = float(
        (path["policy_prior_equity"] - expected_prior).abs().max()
    )
    prior_peak_error = float(
        (path["policy_prior_peak"] - expected_peak).abs().max()
    )
    drawdown_error = float(
        (path["recursive_prior_drawdown"] - expected_drawdown).abs().max()
    )
    floor_error = float((path["floor_equity"] - expected_floor).abs().max())
    non_cash = 1.0 - path["implemented_cash_weight"].astype(float)
    cap_breach = float(
        (non_cash - path["accepted_non_cash_cap"].astype(float)).max()
    )
    active = path["state"].ne("normal")
    r38_active_violations = int(
        path.loc[active, "r38_incremental_enabled"].astype(bool).sum()
    )
    floor_breaches = int((expected_prior < expected_floor - TOLERANCE).sum())
    passed = bool(
        prior_equity_error <= TOLERANCE
        and prior_peak_error <= TOLERANCE
        and drawdown_error <= TOLERANCE
        and floor_error <= TOLERANCE
        and cap_breach <= TOLERANCE
        and r38_active_violations == 0
    )
    return {
        "passed": passed,
        "prior_equity_max_error": prior_equity_error,
        "prior_peak_max_error": prior_peak_error,
        "drawdown_max_error": drawdown_error,
        "floor_max_error": floor_error,
        "maximum_non_cash_cap_breach": cap_breach,
        "r38_active_violations": r38_active_violations,
        "floor_breaches": floor_breaches,
        "sessions": len(path),
    }


def main(
    output: Path = OUTPUT,
    *,
    floor_drawdown: float = FLOOR_DRAWDOWN,
) -> None:
    metrics = pd.read_csv(output / "metrics.csv")
    events = pd.read_csv(output / "pre2008_events.csv")
    neighborhood = pd.read_csv(output / "parameter_neighborhood.csv")
    acceptance = pd.read_csv(output / "acceptance.csv")
    path_rows: list[dict[str, object]] = []
    loaded: dict[str, pd.DataFrame] = {}
    for sample in (
        "pre2008",
        "normal_live",
        "normal_synthetic",
        "proxy_synthetic",
    ):
        path = pd.read_csv(
            output / f"{sample}_central_daily.csv",
            index_col="date",
            parse_dates=True,
        )
        loaded[sample] = path
        path_rows.append(
            {
                "sample": sample,
                **audit_daily_path(path, floor_drawdown=floor_drawdown),
            }
        )

    metric_rows: list[dict[str, object]] = []
    sample_map = {
        "pre2008": "pre2008",
        "normal_live": "normal_live",
        "normal_synthetic": "normal_synthetic",
        "proxy_synthetic": "proxy_synthetic",
    }
    for _, reported in metrics.iterrows():
        sample = str(reported["sample"])
        period = str(reported["period"])
        path = loaded[sample_map[sample]]
        if period not in PERIOD_BOUNDS:
            raise ValueError(f"Missing independent bounds for {period}")
        start, end = PERIOD_BOUNDS[period]
        selected = path.loc[start:end, "net_return"]
        calculated = performance_metrics(selected)
        cagr_error = abs(float(calculated["cagr"]) - float(reported["candidate_cagr"]))
        mdd_error = abs(
            float(calculated["max_drawdown"])
            - float(reported["candidate_max_drawdown"])
        )
        delta_error = abs(
            (
                float(reported["candidate_cagr"])
                - float(reported["baseline_cagr"])
            )
            - float(reported["cagr_delta"])
        )
        metric_rows.append(
            {
                "sample": sample,
                "period": period,
                "cagr_error": cagr_error,
                "max_drawdown_error": mdd_error,
                "cagr_delta_arithmetic_error": delta_error,
                "passed": max(cagr_error, mdd_error, delta_error) <= TOLERANCE,
            }
        )

    event_rows: list[dict[str, object]] = []
    pre_returns = loaded["pre2008"]["net_return"]
    for event, (start, end) in EVENTS.items():
        selected = pre_returns.loc[
            (pre_returns.index > start) & (pre_returns.index <= end)
        ]
        calculated = return_metrics(selected)
        reported = events.loc[events["event"].eq(event)].iloc[0]
        return_error = abs(
            calculated["return"] - float(reported["candidate_return"])
        )
        mdd_error = abs(
            calculated["max_drawdown"]
            - float(reported["candidate_max_drawdown"])
        )
        event_rows.append(
            {
                "event": event,
                "return_error": return_error,
                "max_drawdown_error": mdd_error,
                "recalculated_max_drawdown": calculated["max_drawdown"],
                "passed": max(return_error, mdd_error) <= TOLERANCE,
            }
        )

    path_audit = pd.DataFrame(path_rows)
    metric_audit = pd.DataFrame(metric_rows)
    event_audit = pd.DataFrame(event_rows)
    neighbor_pre = neighborhood.loc[neighborhood["sample"].eq("pre2008")]
    neighbor_modern = neighborhood.loc[
        neighborhood["sample"].eq("normal_synthetic")
    ]
    central_pre = metrics.loc[metrics["sample"].eq("pre2008")].iloc[0]
    central_modern = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["period"].str.startswith("complete_")
    ].iloc[0]
    central_proxy = metrics.loc[
        metrics["sample"].eq("proxy_synthetic")
        & metrics["period"].str.startswith("complete_")
    ].iloc[0]
    gates = {
        "daily_paths_pass": bool(path_audit["passed"].all()),
        "reported_metrics_reproduce": bool(metric_audit["passed"].all()),
        "reported_events_reproduce": bool(event_audit["passed"].all()),
        "complete_pre2008_mdd_pass": bool(
            central_pre["candidate_max_drawdown"] >= -0.20
        ),
        "all_events_mdd_pass": bool(
            event_audit["recalculated_max_drawdown"].ge(-0.20).all()
        ),
        "modern_cagr_cost_pass": bool(central_modern["cagr_delta"] >= -0.02),
        "proxy_cagr_cost_pass": bool(central_proxy["cagr_delta"] >= -0.02),
        "neighbor_pre2008_pass": bool(
            neighbor_pre["candidate_max_drawdown"].ge(-0.22).all()
        ),
        "neighbor_dotcom_pass": bool(
            neighbor_pre["dotcom_max_drawdown"].ge(-0.22).all()
        ),
        "neighbor_modern_cost_pass": bool(
            neighbor_modern["cagr_delta"].ge(-0.025).all()
        ),
        "qualification_acceptance_pass": bool(acceptance["passed"].all()),
    }
    audit_pass = bool(all(gates.values()))
    path_audit.to_csv(output / "independent_path_audit.csv", index=False)
    metric_audit.to_csv(output / "independent_metric_audit.csv", index=False)
    event_audit.to_csv(output / "independent_event_audit.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "audit_pass": audit_pass,
        "production_eligible": False,
        "production_changed": False,
        "orders_generated": False,
        "gates": gates,
        "central_complete_pre2008_mdd": float(
            central_pre["candidate_max_drawdown"]
        ),
        "central_modern_cagr": float(central_modern["candidate_cagr"]),
        "central_modern_cagr_delta": float(central_modern["cagr_delta"]),
        "central_proxy_cagr_delta": float(central_proxy["cagr_delta"]),
        "minimum_neighbor_pre2008_mdd": float(
            neighbor_pre["candidate_max_drawdown"].min()
        ),
        "minimum_neighbor_modern_cagr_delta": float(
            neighbor_modern["cagr_delta"].min()
        ),
    }
    (output / "independent_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

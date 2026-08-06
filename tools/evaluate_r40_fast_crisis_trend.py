from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_bear_recovery_governor import base_components
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
from tools.evaluate_r40_normal_cap_frontier import NormalCapCandidate, simulate_candidate
import tools.evaluate_r21_current_engine_industry_momentum as r21


OUTPUT = Path("output/r40_fast_crisis_trend")
TREND_ASSETS = ("QQQ", "SEMIS", "BOND", "GOLD", "OIL", "USD")
HORIZON_ENSEMBLES = {
    "fast": (20, 60),
    "balanced": (20, 60, 120),
    "medium": (60, 120, 252),
}
RISK_TARGETS = (0.04, 0.06, 0.08, 0.10)
VOLATILITY_DAYS = 63
PER_ASSET_NOTIONAL_CAP = 0.75
GROSS_NOTIONAL_CAP = 2.50
ONE_WAY_COST_BPS = 3.0


@dataclass(frozen=True)
class FastTrendCandidate:
    name: str
    horizons: tuple[int, ...]
    annual_risk_target: float


def causal_fast_trend_overlay(
    *,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    dates: pd.DatetimeIndex,
    horizons: tuple[int, ...],
    annual_risk_target: float,
    one_way_cost_bps: float = ONE_WAY_COST_BPS,
) -> pd.DataFrame:
    if not horizons or min(horizons) < 2:
        raise ValueError("trend horizons must contain windows of at least two days")
    if annual_risk_target <= 0.0:
        raise ValueError("annual_risk_target must be positive")
    if one_way_cost_bps < 0.0:
        raise ValueError("one_way_cost_bps cannot be negative")
    required = set(TREND_ASSETS) | {"CASH"}
    missing = sorted(required.difference(opens.columns) | required.difference(closes.columns))
    if missing:
        raise ValueError(f"missing trend assets: {missing}")

    excess_prices = closes.loc[:, TREND_ASSETS].div(closes["CASH"], axis=0)
    votes = []
    for horizon in horizons:
        moving_average = excess_prices.rolling(horizon, min_periods=horizon).mean()
        votes.append(
            pd.DataFrame(
                np.where(excess_prices > moving_average, 1.0, -1.0),
                index=excess_prices.index,
                columns=excess_prices.columns,
            ).where(moving_average.notna())
        )
    score = sum(votes) / len(votes)

    open_returns = opens.loc[:, TREND_ASSETS].pct_change(fill_method=None)
    cash_returns = opens["CASH"].pct_change(fill_method=None)
    excess_returns = open_returns.sub(cash_returns, axis=0)
    annual_volatility = (
        excess_returns.rolling(VOLATILITY_DAYS, min_periods=VOLATILITY_DAYS)
        .std(ddof=1)
        * sqrt(252.0)
    )
    risk_per_asset = annual_risk_target / sqrt(len(TREND_ASSETS))
    raw_positions = score * risk_per_asset / annual_volatility.clip(lower=0.03)
    raw_positions = raw_positions.clip(
        lower=-PER_ASSET_NOTIONAL_CAP,
        upper=PER_ASSET_NOTIONAL_CAP,
    )
    gross = raw_positions.abs().sum(axis=1)
    gross_scale = (GROSS_NOTIONAL_CAP / gross).clip(upper=1.0).fillna(0.0)
    raw_positions = raw_positions.mul(gross_scale, axis=0)

    # A close signal from t-2 can be traded at the open of t-1 and earns the
    # open(t-1)-to-open(t) return recorded on t. This intentionally forfeits
    # the first overnight move after a signal rather than assuming close fills.
    positions = raw_positions.shift(2).reindex(dates).fillna(0.0)
    realized_excess = excess_returns.reindex(dates).fillna(0.0)
    turnover = positions.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = positions.iloc[0].abs().sum()
    cost = turnover * one_way_cost_bps / 10_000.0
    gross_return = (positions * realized_excess).sum(axis=1)
    net_return = gross_return - cost
    result = pd.DataFrame(
        {
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "cost": cost,
            "gross_notional": positions.abs().sum(axis=1),
        },
        index=dates,
    )
    for asset in TREND_ASSETS:
        result[f"position_{asset}"] = positions[asset]
    return result


def evaluate_sample(
    sample: str,
    settings: dict[str, object],
    candidates: tuple[FastTrendCandidate, ...],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    scenario = COST_SCENARIOS[0]
    components = base_components(settings, scenario)
    base, diagnostics = simulate_candidate(
        settings,
        scenario,
        components,
        NormalCapCandidate("r40", 1.0),
    )
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    dates = base.index
    if sample == "pre2008":
        bounds = ("1931-01-02", "2007-12-31")
        period = "complete_1931_2007"
    else:
        periods = settings["periods"]
        assert isinstance(periods, dict)
        period = next(name for name in periods if name.startswith("complete_"))
        bounds = periods[period]
    start, end = bounds
    baseline_returns = base["net_return"].loc[start:end]
    rows: list[dict[str, object]] = []
    overlays: list[pd.DataFrame] = []
    for candidate in candidates:
        overlay = causal_fast_trend_overlay(
            opens=opens,
            closes=closes,
            dates=dates,
            horizons=candidate.horizons,
            annual_risk_target=candidate.annual_risk_target,
        )
        combined = base["net_return"] + overlay["net_return"]
        base_metrics = performance_metrics(baseline_returns)
        candidate_metrics = performance_metrics(combined.loc[start:end])
        rows.append(
            {
                "sample": sample,
                "period": period,
                "candidate": candidate.name,
                "horizons": "/".join(map(str, candidate.horizons)),
                "annual_risk_target": candidate.annual_risk_target,
                "base_cagr": base_metrics["cagr"],
                "candidate_cagr": candidate_metrics["cagr"],
                "cagr_delta": candidate_metrics["cagr"] - base_metrics["cagr"],
                "base_max_drawdown": base_metrics["max_drawdown"],
                "candidate_max_drawdown": candidate_metrics["max_drawdown"],
                "base_sharpe": base_metrics["sharpe"],
                "candidate_sharpe": candidate_metrics["sharpe"],
                "annualized_overlay_turnover": float(
                    overlay.loc[start:end, "turnover"].mean() * 252.0
                ),
                "annualized_overlay_cost": float(
                    overlay.loc[start:end, "cost"].mean() * 252.0
                ),
                "mean_gross_notional": float(
                    overlay.loc[start:end, "gross_notional"].mean()
                ),
                "controlled_fraction": float(
                    diagnostics.loc[start:end, "state"].ne("normal").mean()
                ),
            }
        )
        if sample == "normal_synthetic":
            saved = overlay.copy()
            saved["base_net_return"] = base["net_return"]
            saved["combined_net_return"] = combined
            saved["candidate"] = candidate.name
            overlays.append(saved)
    return rows, pd.concat(overlays) if overlays else pd.DataFrame()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = tuple(
        FastTrendCandidate(
            f"{ensemble}_risk{int(risk * 100):02d}", horizons, risk
        )
        for ensemble, horizons in HORIZON_ENSEMBLES.items()
        for risk in RISK_TARGETS
    )
    samples = r21._build_samples()
    settings_by_sample = {
        "normal_synthetic": samples["normal_synthetic"],
        "proxy_synthetic": samples["proxy_synthetic"],
    }
    pre_settings, _ = build_pre2008_settings()
    settings_by_sample["pre2008"] = pre_settings

    rows: list[dict[str, object]] = []
    modern_overlay = pd.DataFrame()
    for sample, settings in settings_by_sample.items():
        print(f"evaluating {sample}", flush=True)
        sample_rows, overlay = evaluate_sample(sample, settings, candidates)
        rows.extend(sample_rows)
        if sample == "normal_synthetic":
            modern_overlay = overlay
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    modern_overlay.to_csv(OUTPUT / "normal_synthetic_overlays.csv")
    modern = metrics.loc[metrics["sample"].eq("normal_synthetic")]
    proxy = metrics.loc[metrics["sample"].eq("proxy_synthetic")]
    pre = metrics.loc[metrics["sample"].eq("pre2008")]
    frontier = modern.merge(
        proxy[["candidate", "candidate_cagr", "candidate_max_drawdown"]],
        on="candidate",
        suffixes=("_modern", "_proxy"),
    ).merge(
        pre[["candidate", "candidate_cagr", "candidate_max_drawdown"]],
        on="candidate",
    ).rename(
        columns={
            "candidate_cagr": "candidate_cagr_pre2008",
            "candidate_max_drawdown": "candidate_max_drawdown_pre2008",
        }
    )
    frontier["target_pass"] = (
        frontier["candidate_cagr_modern"].between(0.24, 0.25, inclusive="both")
        & frontier["candidate_max_drawdown_pre2008"].ge(-0.20)
    )
    frontier.to_csv(OUTPUT / "frontier.csv", index=False)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate_count": len(candidates),
        "target_candidates": frontier.loc[
            frontier["target_pass"], "candidate"
        ].tolist(),
        "production_changed": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(frontier.round(6).to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

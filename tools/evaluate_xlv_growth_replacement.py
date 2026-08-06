from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    metric_delta,
    relative_log_return,
)
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
from tools.evaluate_r39_relative_damage_concentration_veto import (
    _simulate_account,
    _staged_inputs,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21


OUTPUT = Path("output/xlv_growth_replacement")
XLV_PRICES = Path("data/sector_hmm_shadow_prices.csv")
CENTRAL_FRACTION = 0.10
SENSITIVITY_FRACTIONS = (0.05, CENTRAL_FRACTION, 0.15)
XLV_COST_MULTIPLIER = 1.0
CUMULATIVE_TRIALS = 120
TRADING_EPSILON = 1e-14


def load_xlv_prices(path: Path = XLV_PRICES) -> pd.Series:
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    if "XLV" not in frame:
        raise ValueError(f"{path} does not contain XLV")
    series = pd.to_numeric(frame["XLV"], errors="coerce").dropna()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    if series.index.has_duplicates or not series.index.is_monotonic_increasing:
        raise ValueError("XLV prices have invalid dates")
    if series.le(0.0).any():
        raise ValueError("XLV prices must be positive")
    return series.rename("XLV")


def xlv_growth_replacement(
    baseline_returns: pd.Series,
    targets: pd.DataFrame,
    execution_daily: pd.DataFrame,
    closes: pd.DataFrame,
    xlv_prices: pd.Series,
    *,
    fraction: float,
    xlv_one_way_cost_bps: float,
) -> pd.DataFrame:
    """Apply a conservative, self-financed XLV replacement at target level.

    XLV replaces the same fraction of the existing QQQ+SEMIS sleeve.  The
    baseline return already includes every production cost; this marginal test
    charges the XLV adjustment separately and deliberately takes no credit for
    reduced QQQ/SEMIS trading.  It is therefore a research screen, not an
    execution simulator or a production allocation rule.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if xlv_one_way_cost_bps < 0.0:
        raise ValueError("xlv_one_way_cost_bps must be non-negative")
    required = {"QQQ", "SEMIS"}
    if missing := sorted(required.difference(targets.columns)):
        raise ValueError(f"Targets missing {missing}")
    if missing := sorted(required.difference(closes.columns)):
        raise ValueError(f"Closes missing {missing}")
    if "turnover" not in execution_daily:
        raise ValueError("Execution data missing turnover")

    index = (
        baseline_returns.index.intersection(targets.index)
        .intersection(execution_daily.index)
        .intersection(closes.index)
        .intersection(xlv_prices.index)
    )
    if index.empty:
        raise ValueError("No common dates for XLV replacement")
    target = targets.loc[index, ["QQQ", "SEMIS"]].astype(float)
    growth_weight = target.sum(axis=1)
    if growth_weight.lt(-1e-12).any():
        raise ValueError("Growth sleeve cannot be negative")
    qqq_share = target["QQQ"].div(growth_weight.where(growth_weight > 0.0))
    qqq_share = qqq_share.fillna(0.0)
    semis_share = 1.0 - qqq_share
    growth_return = (
        qqq_share * closes.loc[index, "QQQ"].pct_change(fill_method=None)
        + semis_share
        * closes.loc[index, "SEMIS"].pct_change(fill_method=None)
    ).fillna(0.0)
    xlv_return = xlv_prices.loc[index].pct_change(fill_method=None).fillna(0.0)
    replacement_weight = (fraction * growth_weight).rename(
        "xlv_replacement_weight"
    )
    base_trade = execution_daily.loc[index, "turnover"].gt(
        TRADING_EPSILON
    )
    trade_weight = replacement_weight.diff().abs().where(base_trade, 0.0)
    if len(trade_weight):
        trade_weight.iloc[0] = replacement_weight.iloc[0]
    xlv_cost = (
        trade_weight * xlv_one_way_cost_bps / 10_000.0
    ).rename("xlv_extra_trading_cost")
    baseline = baseline_returns.reindex(index).rename("baseline_return")
    if fraction == 0.0:
        candidate = baseline.copy().rename("candidate_return")
    else:
        candidate = (
            baseline
            + replacement_weight * (xlv_return - growth_return)
            - xlv_cost
        ).rename("candidate_return")
    frame = pd.concat(
        [
            baseline,
            candidate,
            growth_weight.rename("growth_weight"),
            replacement_weight,
            qqq_share.rename("qqq_share_within_replaced_growth"),
            semis_share.rename("semis_share_within_replaced_growth"),
            growth_return.rename("replaced_growth_return"),
            xlv_return.rename("xlv_return"),
            base_trade.rename("base_trade"),
            trade_weight.rename("xlv_one_way_turnover"),
            xlv_cost,
        ],
        axis=1,
    ).dropna(subset=["baseline_return", "candidate_return"])
    if fraction == 0.0 and not np.allclose(
        frame["baseline_return"], frame["candidate_return"]
    ):
        raise AssertionError("Zero XLV replacement must equal the baseline")
    if frame["xlv_replacement_weight"].gt(
        frame["growth_weight"] + 1e-12
    ).any():
        raise AssertionError("XLV replacement increased the growth budget")
    if frame["candidate_return"].le(-1.0).any():
        raise RuntimeError("XLV replacement produced a total-loss day")
    return frame


def _period_rows(
    sample: str,
    scenario: str,
    fraction: float,
    periods: dict[str, tuple[str, str]],
    path: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        selected = path.loc[start:end]
        if selected.empty:
            continue
        rows.append(
            {
                "sample": sample,
                "scenario": scenario,
                "xlv_fraction_of_growth": fraction,
                "period": period,
                "annualized_relative_log_return": float(
                    relative_log_return(
                        selected["baseline_return"],
                        selected["candidate_return"],
                    ).mean()
                    * 252.0
                ),
                **metric_delta(
                    selected["baseline_return"],
                    selected["candidate_return"],
                ),
            }
        )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    xlv_prices = load_xlv_prices()
    samples = r21._build_samples()
    metric_rows: list[dict[str, object]] = []
    paths: dict[tuple[str, float], pd.DataFrame] = {}

    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            targets, daily, _ = _staged_inputs(settings, scenario)
            baseline = _simulate_account(settings, scenario, targets, daily)
            closes = settings["closes"]
            assert isinstance(closes, pd.DataFrame)
            for fraction in SENSITIVITY_FRACTIONS:
                path = xlv_growth_replacement(
                    baseline["net_return"],
                    targets,
                    daily,
                    closes,
                    xlv_prices,
                    fraction=fraction,
                    xlv_one_way_cost_bps=(
                        scenario.base_one_way_cost_bps
                        * XLV_COST_MULTIPLIER
                    ),
                )
                metric_rows.extend(
                    _period_rows(
                        sample,
                        scenario.name,
                        fraction,
                        periods,
                        path,
                    )
                )
                if scenario.name == "current_liquidity":
                    paths[(sample, fraction)] = path
                    path.to_csv(
                        OUTPUT / f"{sample}_xlv{int(fraction * 100)}.csv",
                        index_label="date",
                    )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    family_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic", "normal_live"):
        candidate_paths = [paths[(sample, value)] for value in SENSITIVITY_FRACTIONS]
        common = candidate_paths[0].index
        for path in candidate_paths[1:]:
            common = common.intersection(path.index)
        relatives = np.column_stack(
            [
                relative_log_return(
                    path.loc[common, "baseline_return"],
                    path.loc[common, "candidate_return"],
                )
                for path in candidate_paths
            ]
        )
        for block_days in (21, 63):
            check = circular_family_reality_check(
                relatives,
                SENSITIVITY_FRACTIONS.index(CENTRAL_FRACTION),
                block_days,
            )
            family_rows.append(
                {
                    "sample": sample,
                    "block_days": block_days,
                    "declared_family": ",".join(
                        f"{fraction:.0%}" for fraction in SENSITIVITY_FRACTIONS
                    ),
                    "central_fraction": CENTRAL_FRACTION,
                    "cumulative_trials": CUMULATIVE_TRIALS,
                    **check,
                    "cumulative_trial_adjusted_p_value": min(
                        1.0,
                        float(check["familywise_reality_check_p_value"])
                        * CUMULATIVE_TRIALS,
                    ),
                }
            )
    family = pd.DataFrame(family_rows)
    family.to_csv(OUTPUT / "family_reality_check.csv", index=False)

    central = metrics.loc[
        (metrics["xlv_fraction_of_growth"] == CENTRAL_FRACTION)
        & (metrics["scenario"] == "current_liquidity")
    ]

    def one(sample: str, period: str) -> pd.Series:
        rows = central.loc[
            (central["sample"] == sample) & (central["period"] == period)
        ]
        if len(rows) != 1:
            raise ValueError(f"Expected one central row for {sample} {period}")
        return rows.iloc[0]

    holdout = one("normal_synthetic", "holdout_2022_2025")
    proxy = one("proxy_synthetic", "complete_2006_2026")
    live = one("normal_live", "live_2022_2026")
    stress_holdout = metrics.loc[
        (metrics["sample"] == "normal_synthetic")
        & (metrics["scenario"] == "cost_stress")
        & (metrics["xlv_fraction_of_growth"] == CENTRAL_FRACTION)
        & (metrics["period"] == "holdout_2022_2025")
    ].iloc[0]
    qualification_checks = {
        "holdout_positive_cagr_delta": bool(holdout["cagr_delta"] > 0.0),
        "holdout_nonnegative_sharpe_delta": bool(
            holdout["sharpe_delta"] >= 0.0
        ),
        "holdout_drawdown_not_worse_than_25bp": bool(
            holdout["max_drawdown_delta"] >= -0.0025
        ),
        "proxy_positive_cagr_delta": bool(proxy["cagr_delta"] > 0.0),
        "live_nonnegative_sharpe_delta": bool(live["sharpe_delta"] >= 0.0),
        "stressed_holdout_drawdown_not_worse_than_25bp": bool(
            stress_holdout["max_drawdown_delta"] >= -0.0025
        ),
        "family_reality_check_after_cumulative_trials": bool(
            family["cumulative_trial_adjusted_p_value"].le(0.05).all()
        ),
    }
    passed = all(qualification_checks.values())
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_marginal_replacement_screen",
        "production_changed": False,
        "orders_generated": False,
        "candidate": "XLV replaces a fixed fraction of QQQ+SEMIS",
        "central_fraction_of_growth": CENTRAL_FRACTION,
        "sensitivity_fractions": list(SENSITIVITY_FRACTIONS),
        "latest_xlv_price_date": xlv_prices.index.max().date().isoformat(),
        "cost_treatment": (
            "XLV trades are charged separately; no offsetting credit is taken "
            "for reduced QQQ/SEMIS trading costs."
        ),
        "qualification_checks": qualification_checks,
        "research_passed": passed,
        "decision": (
            "continue_forward_shadow_only"
            if passed
            else "reject_current_xlv_specification"
        ),
        "limitation": (
            "Target-level marginal screen; it preserves the existing HMM, "
            "cash, gold, GDE, and risk-control path and is not a production "
            "execution simulator."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

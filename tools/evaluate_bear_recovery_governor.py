from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
from tools.evaluate_r10_gde_capital_efficiency import (
    ASSETS,
    simulate_gde_substitution,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
)
from tools.evaluate_r12_volatility_managed_risk import (
    GDE_FRACTION,
    simulate_fixed_r11,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r38_state_dependent_rollout as rollout


OUTPUT = Path("output/bear_recovery_governor")
R38_SHARE = 0.25
GROWTH_ASSETS = ("QQQ", "SEMIS")
SMA_SESSIONS = 200
MARKET_DRAWDOWN_LOOKBACK = 252
RAMP_CAPS = (0.20, 0.35, 0.50)


@dataclass(frozen=True)
class BearRecoveryCandidate:
    name: str
    drawdown_trigger: float = -0.10
    recovery_confirmation_days: int = 20
    defense_growth_cap: float = 0.10
    ramp_stage_sessions: int = 21

    def __post_init__(self) -> None:
        if not -1.0 < self.drawdown_trigger < 0.0:
            raise ValueError("drawdown_trigger must be in (-1, 0)")
        if self.recovery_confirmation_days < 1:
            raise ValueError("recovery_confirmation_days must be positive")
        if not 0.0 <= self.defense_growth_cap <= RAMP_CAPS[0]:
            raise ValueError("defense_growth_cap is outside the declared range")
        if self.ramp_stage_sessions < 1:
            raise ValueError("ramp_stage_sessions must be positive")


CENTRAL = BearRecoveryCandidate("central")
NEIGHBORS = (
    replace(CENTRAL, name="trigger_08", drawdown_trigger=-0.08),
    replace(CENTRAL, name="trigger_12", drawdown_trigger=-0.12),
    replace(CENTRAL, name="confirm_10", recovery_confirmation_days=10),
    replace(CENTRAL, name="confirm_30", recovery_confirmation_days=30),
    replace(CENTRAL, name="cap_00", defense_growth_cap=0.00),
    replace(CENTRAL, name="cap_20", defense_growth_cap=0.20),
    replace(CENTRAL, name="ramp_14", ramp_stage_sessions=14),
    replace(CENTRAL, name="ramp_28", ramp_stage_sessions=28),
)


def causal_bear_signals(
    closes: pd.DataFrame,
    *,
    sma_sessions: int = SMA_SESSIONS,
    drawdown_lookback: int = MARKET_DRAWDOWN_LOOKBACK,
) -> pd.DataFrame:
    missing = [asset for asset in GROWTH_ASSETS if asset not in closes]
    if missing:
        raise ValueError(f"Missing growth prices: {missing}")
    if sma_sessions < 2:
        raise ValueError("sma_sessions must be at least two")
    if drawdown_lookback < sma_sessions:
        raise ValueError("drawdown_lookback cannot be shorter than the SMA")
    prices = closes.loc[:, list(GROWTH_ASSETS)].astype(float)
    returns = prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    growth_index = (1.0 + returns).cumprod()
    rolling_peak = growth_index.rolling(
        drawdown_lookback,
        min_periods=drawdown_lookback,
    ).max()
    drawdown = growth_index / rolling_peak - 1.0
    prior_prices = prices.shift(1)
    prior_sma = prices.rolling(
        sma_sessions,
        min_periods=sma_sessions,
    ).mean().shift(1)
    below = prior_prices.lt(prior_sma).all(axis=1)
    above = prior_prices.gt(prior_sma).all(axis=1)
    return pd.DataFrame(
        {
            "prior_growth_drawdown": drawdown.shift(1).fillna(0.0),
            "both_below_sma": below.fillna(False),
            "both_above_sma": above.fillna(False),
        },
        index=closes.index,
    )


def bear_recovery_state(
    signals: pd.DataFrame,
    candidate: BearRecoveryCandidate,
) -> pd.DataFrame:
    required = {
        "prior_growth_drawdown",
        "both_below_sma",
        "both_above_sma",
    }
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"Bear signals are missing: {sorted(missing)}")
    state = "normal"
    recovery_run = 0
    ramp_day = 0
    rows: list[dict[str, object]] = []
    for date, signal in signals.iterrows():
        entry = bool(
            float(signal["prior_growth_drawdown"])
            <= candidate.drawdown_trigger
            and bool(signal["both_below_sma"])
        )
        previous_state = state
        previous_stage = -1
        if state == "ramp":
            previous_stage = min(
                ramp_day // candidate.ramp_stage_sessions,
                len(RAMP_CAPS) - 1,
            )

        if entry:
            state = "defense"
            recovery_run = 0
            ramp_day = 0
        elif state == "defense":
            if bool(signal["both_above_sma"]):
                recovery_run += 1
            else:
                recovery_run = 0
            if recovery_run >= candidate.recovery_confirmation_days:
                state = "ramp"
                ramp_day = 0
        elif state == "ramp":
            ramp_day += 1
            if ramp_day >= candidate.ramp_stage_sessions * len(RAMP_CAPS):
                state = "normal"
                recovery_run = 0
                ramp_day = 0

        stage = -1
        growth_cap = 1.0
        if state == "defense":
            growth_cap = candidate.defense_growth_cap
        elif state == "ramp":
            stage = min(
                ramp_day // candidate.ramp_stage_sessions,
                len(RAMP_CAPS) - 1,
            )
            growth_cap = RAMP_CAPS[stage]
        rows.append(
            {
                "state": state,
                "entry": entry and previous_state != "defense",
                "state_change": state != previous_state,
                "stage_change": state == "ramp" and stage != previous_stage,
                "recovery_run": recovery_run,
                "ramp_day": ramp_day,
                "growth_cap": growth_cap,
                "r38_incremental_enabled": state == "normal",
            }
        )
    return signals.join(pd.DataFrame(rows, index=signals.index))


def cap_growth_to_cash(
    weights: pd.Series,
    growth_cap: float,
) -> pd.Series:
    if not 0.0 <= growth_cap <= 1.0:
        raise ValueError("growth_cap must be in [0, 1]")
    adjusted = weights.loc[ASSETS].astype(float).copy()
    current_growth = float(adjusted.loc[list(GROWTH_ASSETS)].sum())
    if current_growth <= growth_cap + 1e-12:
        return adjusted
    adjusted.loc[list(GROWTH_ASSETS)] *= growth_cap / current_growth
    adjusted["CASH"] += current_growth - growth_cap
    return adjusted


def apply_governor(
    blended_weights: pd.DataFrame,
    r11_weights: pd.DataFrame,
    blended_daily: pd.DataFrame,
    diagnostics: pd.DataFrame,
    candidate: BearRecoveryCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = (
        blended_weights.index.intersection(r11_weights.index)
        .intersection(blended_daily.index)
        .intersection(diagnostics.index)
    )
    adjusted = blended_weights.reindex(index).loc[:, ASSETS].copy()
    base_r11 = r11_weights.reindex(index).loc[:, ASSETS]
    execution = blended_daily.reindex(index).copy()
    detail = diagnostics.reindex(index).copy()
    implemented_growth: list[float] = []
    changed: list[bool] = []
    for date in index:
        state = str(detail.loc[date, "state"])
        target = adjusted.loc[date]
        if state != "normal":
            target = cap_growth_to_cash(
                base_r11.loc[date],
                float(detail.loc[date, "growth_cap"]),
            )
        changed_today = not np.allclose(
            target.to_numpy(dtype=float),
            adjusted.loc[date].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        )
        adjusted.loc[date] = target
        implemented_growth.append(float(target.loc[list(GROWTH_ASSETS)].sum()))
        changed.append(changed_today)
        if bool(detail.loc[date, "state_change"]) or bool(
            detail.loc[date, "stage_change"]
        ):
            execution.loc[date, "turnover"] = max(
                float(execution.loc[date, "turnover"]),
                1e-12,
            )
    detail["candidate_changed_target"] = changed
    detail["implemented_growth_weight"] = implemented_growth
    detail["implemented_cash_weight"] = adjusted["CASH"]
    non_cash = [asset for asset in ASSETS if asset != "CASH"]
    conservation_error = adjusted[non_cash].sum(axis=1) + adjusted["CASH"] - 1.0
    detail["weight_conservation_error"] = conservation_error
    if float(conservation_error.abs().max()) > 1e-10:
        raise AssertionError("Governor target weights do not sum to one")
    active = detail["state"].ne("normal")
    if (
        detail.loc[active, "implemented_growth_weight"]
        > detail.loc[active, "growth_cap"] + 1e-10
    ).any():
        raise AssertionError("Governor exceeded its active growth cap")
    return adjusted, execution, detail


def base_components(
    settings: dict[str, object],
    scenario: object,
) -> dict[str, pd.DataFrame | pd.Series]:
    _, r11_weights, r11_daily = simulate_fixed_r11(settings, scenario)
    r38_weights, r38_daily, _ = rollout.build_r38_targets(settings, scenario)
    index = (
        r11_weights.index.intersection(r11_daily.index)
        .intersection(r38_weights.index)
        .intersection(r38_daily.index)
    )
    blended_weights = (
        (1.0 - R38_SHARE) * r11_weights.reindex(index)
        + R38_SHARE * r38_weights.reindex(index)
    )
    r11_trade = r11_daily.reindex(index)["turnover"].fillna(0.0).gt(1e-14)
    r38_trade = r38_daily.reindex(index)["turnover"].fillna(0.0).gt(1e-14)
    blended_daily = r11_daily.reindex(index).copy()
    blended_daily["turnover"] = np.where(r11_trade | r38_trade, 1e-12, 0.0)
    blended_slippage = (
        (1.0 - R38_SHARE)
        * r11_daily.reindex(index)["slippage_cost"].fillna(0.0)
        + R38_SHARE
        * r38_daily.reindex(index)["slippage_cost"].fillna(0.0)
    )
    baseline, _ = rollout.simulate_blended_account(
        settings,
        scenario,
        r11_weights,
        r11_daily,
        r38_weights,
        r38_daily,
        pd.Series(R38_SHARE, index=index),
    )
    return {
        "r11_weights": r11_weights.reindex(index),
        "blended_weights": blended_weights,
        "blended_daily": blended_daily,
        "blended_slippage": blended_slippage,
        "r11_slippage": r11_daily.reindex(index)["slippage_cost"].fillna(0.0),
        "baseline": baseline,
    }


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    components: dict[str, pd.DataFrame | pd.Series],
    candidate: BearRecoveryCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes = settings["closes"]
    opens = settings["opens"]
    assert isinstance(closes, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    signals = causal_bear_signals(closes)
    diagnostics = bear_recovery_state(signals, candidate)
    blended_weights = components["blended_weights"]
    r11_weights = components["r11_weights"]
    blended_daily = components["blended_daily"]
    blended_slippage = components["blended_slippage"]
    r11_slippage = components["r11_slippage"]
    assert isinstance(blended_weights, pd.DataFrame)
    assert isinstance(r11_weights, pd.DataFrame)
    assert isinstance(blended_daily, pd.DataFrame)
    assert isinstance(blended_slippage, pd.Series)
    assert isinstance(r11_slippage, pd.Series)
    weights, daily, diagnostics = apply_governor(
        blended_weights,
        r11_weights,
        blended_daily,
        diagnostics,
        candidate,
    )
    active = diagnostics["state"].ne("normal")
    candidate_slippage = blended_slippage.reindex(weights.index).copy()
    candidate_slippage.loc[active] = r11_slippage.reindex(weights.index).loc[
        active
    ]
    trial = simulate_gde_substitution(
        str(settings["directory"]),
        opens,
        closes,
        substitution_fraction=GDE_FRACTION,
        gate_mode="growth_linked",
        gde_return_mode=str(settings["mode"]),
        start_date=str(settings["start"]),
        end_date=None,
        base_one_way_cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,  # type: ignore[attr-defined]
        financing_spread_bps=scenario.financing_spread_bps,  # type: ignore[attr-defined]
        weights_override=weights,
        daily_override=daily,
        extra_slippage=candidate_slippage,
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    return trial, weights, diagnostics.reindex(trial.index)


def period_metrics(
    returns: pd.Series,
    start: str,
    end: str | None,
) -> dict[str, float]:
    selected = returns.loc[start:end].dropna()
    if selected.empty:
        raise ValueError(f"No returns for {start} through {end}")
    return performance_metrics(selected)


def metric_row(
    sample: str,
    period: str,
    bounds: tuple[str, str | None],
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, object]:
    start, end = bounds
    base = period_metrics(baseline, start, end)
    trial = period_metrics(candidate, start, end)
    return {
        "sample": sample,
        "period": period,
        "baseline_cagr": base["cagr"],
        "candidate_cagr": trial["cagr"],
        "cagr_delta": trial["cagr"] - base["cagr"],
        "baseline_max_drawdown": base["max_drawdown"],
        "candidate_max_drawdown": trial["max_drawdown"],
        "max_drawdown_delta": (
            trial["max_drawdown"] - base["max_drawdown"]
        ),
        "baseline_sharpe": base["sharpe"],
        "candidate_sharpe": trial["sharpe"],
        "sharpe_delta": trial["sharpe"] - base["sharpe"],
    }


def pre2008_event_rows(
    candidate_name: str,
    baseline: pd.Series,
    trial: pd.Series,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event, (start, end) in EVENTS.items():
        base = return_metrics(baseline.loc[(baseline.index > start) & (baseline.index <= end)])
        test = return_metrics(trial.loc[(trial.index > start) & (trial.index <= end)])
        rows.append(
            {
                "candidate": candidate_name,
                "event": event,
                "baseline_return": base["return"],
                "candidate_return": test["return"],
                "baseline_max_drawdown": base["max_drawdown"],
                "candidate_max_drawdown": test["max_drawdown"],
                "max_drawdown_delta": (
                    test["max_drawdown"] - base["max_drawdown"]
                ),
                "absolute_20pct_ceiling_pass": test["max_drawdown"] >= -0.20,
            }
        )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    pre_settings, _ = build_pre2008_settings()
    samples = {"pre2008": pre_settings, **r21._build_samples()}
    components = {
        sample: base_components(settings, scenario)
        for sample, settings in samples.items()
    }
    central_paths: dict[str, pd.DataFrame] = {}
    central_diagnostics: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    neighborhood_rows: list[dict[str, object]] = []

    for sample, settings in samples.items():
        trial, weights, diagnostics = simulate_candidate(
            settings,
            scenario,
            components[sample],
            CENTRAL,
        )
        central_paths[sample] = trial
        central_diagnostics[sample] = diagnostics
        trial.to_csv(OUTPUT / f"{sample}_central_daily.csv", index_label="date")
        weights.to_csv(OUTPUT / f"{sample}_central_weights.csv", index_label="date")
        diagnostics.to_csv(
            OUTPUT / f"{sample}_central_diagnostics.csv",
            index_label="date",
        )
        baseline_path = components[sample]["baseline"]
        assert isinstance(baseline_path, pd.DataFrame)
        baseline_returns = baseline_path["net_return"]
        if sample == "pre2008":
            metric_rows.append(
                metric_row(
                    sample,
                    "complete_1931_2007",
                    ("1931-01-02", "2007-12-31"),
                    baseline_returns,
                    trial["net_return"],
                )
            )
            event_rows.extend(
                pre2008_event_rows(
                    CENTRAL.name,
                    baseline_returns,
                    trial["net_return"],
                )
            )
        else:
            periods = settings["periods"]
            assert isinstance(periods, dict)
            for period, bounds in periods.items():
                metric_rows.append(
                    metric_row(
                        sample,
                        period,
                        bounds,
                        baseline_returns,
                        trial["net_return"],
                    )
                )

    for candidate in NEIGHBORS:
        for sample in ("pre2008", "normal_synthetic", "proxy_synthetic"):
            settings = samples[sample]
            trial, _, diagnostics = simulate_candidate(
                settings,
                scenario,
                components[sample],
                candidate,
            )
            baseline_path = components[sample]["baseline"]
            assert isinstance(baseline_path, pd.DataFrame)
            baseline_returns = baseline_path["net_return"]
            if sample == "pre2008":
                complete = metric_row(
                    sample,
                    "complete_1931_2007",
                    ("1931-01-02", "2007-12-31"),
                    baseline_returns,
                    trial["net_return"],
                )
                dotcom = next(
                    row
                    for row in pre2008_event_rows(
                        candidate.name,
                        baseline_returns,
                        trial["net_return"],
                    )
                    if row["event"] == "dotcom_2000_2002"
                )
                period = "complete_1931_2007"
            else:
                periods = settings["periods"]
                assert isinstance(periods, dict)
                period = next(name for name in periods if name.startswith("complete_"))
                complete = metric_row(
                    sample,
                    period,
                    periods[period],
                    baseline_returns,
                    trial["net_return"],
                )
                dotcom = {"candidate_max_drawdown": np.nan}
            neighborhood_rows.append(
                {
                    "candidate": candidate.name,
                    "sample": sample,
                    "period": period,
                    "cagr_delta": complete["cagr_delta"],
                    "candidate_max_drawdown": complete["candidate_max_drawdown"],
                    "max_drawdown_delta": complete["max_drawdown_delta"],
                    "dotcom_max_drawdown": dotcom["candidate_max_drawdown"],
                    "active_fraction": float(
                        diagnostics["state"].ne("normal").mean()
                    ),
                    "defense_entries": int(diagnostics["entry"].sum()),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    neighborhood = pd.DataFrame(neighborhood_rows)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    events.to_csv(OUTPUT / "pre2008_events.csv", index=False)
    neighborhood.to_csv(OUTPUT / "parameter_neighborhood.csv", index=False)

    dotcom = events.loc[events["event"].eq("dotcom_2000_2002")].iloc[0]
    pre_complete = metrics.loc[metrics["sample"].eq("pre2008")].iloc[0]
    normal_complete = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["period"].str.startswith("complete_")
    ].iloc[0]
    proxy_complete = metrics.loc[
        metrics["sample"].eq("proxy_synthetic")
        & metrics["period"].str.startswith("complete_")
    ].iloc[0]
    acceptance = pd.DataFrame(
        [
            {
                "gate": "dotcom_max_drawdown_at_or_above_minus_20pct",
                "passed": bool(dotcom["candidate_max_drawdown"] >= -0.20),
            },
            {
                "gate": "all_named_pre2008_events_at_or_above_minus_20pct",
                "passed": bool(events["absolute_20pct_ceiling_pass"].all()),
            },
            {
                "gate": "complete_pre2008_max_drawdown_at_or_above_minus_25pct",
                "passed": bool(pre_complete["candidate_max_drawdown"] >= -0.25),
            },
            {
                "gate": "normal_complete_cagr_cost_at_most_3pct",
                "passed": bool(normal_complete["cagr_delta"] >= -0.03),
            },
            {
                "gate": "normal_complete_drawdown_not_worse",
                "passed": bool(normal_complete["max_drawdown_delta"] >= 0.0),
            },
            {
                "gate": "proxy_complete_cagr_cost_at_most_3pct",
                "passed": bool(proxy_complete["cagr_delta"] >= -0.03),
            },
            {
                "gate": "proxy_complete_drawdown_not_worse",
                "passed": bool(proxy_complete["max_drawdown_delta"] >= 0.0),
            },
            {
                "gate": "neighbor_dotcom_paths_at_or_above_minus_25pct",
                "passed": bool(
                    neighborhood.loc[
                        neighborhood["sample"].eq("pre2008"),
                        "dotcom_max_drawdown",
                    ].ge(-0.25).all()
                ),
            },
        ]
    )
    research_pass = bool(acceptance["passed"].all())
    acceptance.loc[len(acceptance)] = {
        "gate": "research_pass",
        "passed": research_pass,
    }
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate": CENTRAL.__dict__,
        "research_pass": research_pass,
        "production_eligible": False,
        "production_changed": False,
        "orders_generated": False,
        "rule": (
            "Using prior-close data, enter defense after a 10% equal-weight "
            "QQQ/SEMIS drawdown from the trailing 252-session peak with both "
            "below their 200-day averages; cap "
            "growth at 10%, use R11 rather than R38, require 20 consecutive "
            "sessions above both averages, then ramp through 20/35/50% caps."
        ),
        "dotcom_baseline_max_drawdown": float(dotcom["baseline_max_drawdown"]),
        "dotcom_candidate_max_drawdown": float(dotcom["candidate_max_drawdown"]),
        "pre2008_complete_candidate_max_drawdown": float(
            pre_complete["candidate_max_drawdown"]
        ),
        "normal_complete_cagr_delta": float(normal_complete["cagr_delta"]),
        "proxy_complete_cagr_delta": float(proxy_complete["cagr_delta"]),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))
    print("\nPre-2008 events:")
    print(events.round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

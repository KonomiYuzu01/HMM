from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_bear_recovery_governor import (
    CENTRAL as WEIGHT_CAP_DEFINITION,
    OUTPUT as BEAR_OUTPUT,
    RAMP_CAPS,
    apply_governor,
    base_components,
    metric_row,
    pre2008_event_rows,
)
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r10_gde_capital_efficiency import simulate_gde_substitution
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
)
from tools.evaluate_r12_volatility_managed_risk import GDE_FRACTION
import tools.evaluate_r21_current_engine_industry_momentum as r21


OUTPUT = Path("output/reference_drawdown_governor")
SMA_SESSIONS = 200


@dataclass(frozen=True)
class ReferenceGovernorCandidate:
    name: str
    entry_drawdown: float = -0.10
    release_drawdown: float = -0.05
    recovery_confirmation_days: int = 10
    ramp_stage_sessions: int = 14

    def __post_init__(self) -> None:
        if not -1.0 < self.entry_drawdown < self.release_drawdown < 0.0:
            raise ValueError("drawdown thresholds must provide recovery hysteresis")
        if self.recovery_confirmation_days < 1:
            raise ValueError("recovery_confirmation_days must be positive")
        if self.ramp_stage_sessions < 1:
            raise ValueError("ramp_stage_sessions must be positive")


CENTRAL = ReferenceGovernorCandidate("central")
NEIGHBORS = (
    replace(CENTRAL, name="entry_08", entry_drawdown=-0.08),
    replace(CENTRAL, name="entry_12", entry_drawdown=-0.12),
    replace(CENTRAL, name="release_03", release_drawdown=-0.03),
    replace(CENTRAL, name="release_07", release_drawdown=-0.07),
    replace(CENTRAL, name="confirm_05", recovery_confirmation_days=5),
    replace(CENTRAL, name="confirm_15", recovery_confirmation_days=15),
    replace(CENTRAL, name="ramp_07", ramp_stage_sessions=7),
    replace(CENTRAL, name="ramp_21", ramp_stage_sessions=21),
)


def causal_reference_signals(
    closes: pd.DataFrame,
    reference_path: pd.DataFrame,
    *,
    sma_sessions: int = SMA_SESSIONS,
) -> pd.DataFrame:
    if "drawdown" not in reference_path:
        raise ValueError("Reference path is missing drawdown")
    prices = closes[["QQQ", "SEMIS"]].astype(float)
    growth_return = prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    growth_index = (1.0 + growth_return).cumprod()
    prior_growth = growth_index.shift(1)
    prior_sma = growth_index.rolling(
        sma_sessions,
        min_periods=sma_sessions,
    ).mean().shift(1)
    return pd.DataFrame(
        {
            "prior_reference_drawdown": reference_path["drawdown"].shift(1),
            "growth_above_sma": prior_growth.gt(prior_sma).fillna(False),
        },
        index=closes.index,
    ).fillna({"prior_reference_drawdown": 0.0})


def reference_governor_state(
    signals: pd.DataFrame,
    candidate: ReferenceGovernorCandidate,
) -> pd.DataFrame:
    state = "normal"
    recovery_run = 0
    ramp_day = 0
    rows: list[dict[str, object]] = []
    for _, signal in signals.iterrows():
        entry = float(signal["prior_reference_drawdown"]) <= candidate.entry_drawdown
        release_quality = bool(
            float(signal["prior_reference_drawdown"])
            >= candidate.release_drawdown
            and bool(signal["growth_above_sma"])
        )
        previous_state = state
        previous_stage = (
            min(ramp_day // candidate.ramp_stage_sessions, len(RAMP_CAPS) - 1)
            if state == "ramp"
            else -1
        )
        if entry:
            state = "defense"
            recovery_run = 0
            ramp_day = 0
        elif state == "defense":
            recovery_run = recovery_run + 1 if release_quality else 0
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
            growth_cap = 0.0
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
                "release_quality": release_quality,
                "recovery_run": recovery_run,
                "ramp_day": ramp_day,
                "growth_cap": growth_cap,
                "r38_incremental_enabled": state == "normal",
            }
        )
    return signals.join(pd.DataFrame(rows, index=signals.index))


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    components: dict[str, pd.DataFrame | pd.Series],
    candidate: ReferenceGovernorCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = settings["closes"]
    opens = settings["opens"]
    reference = components["baseline"]
    blended_weights = components["blended_weights"]
    r11_weights = components["r11_weights"]
    blended_daily = components["blended_daily"]
    blended_slippage = components["blended_slippage"]
    r11_slippage = components["r11_slippage"]
    assert isinstance(closes, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(reference, pd.DataFrame)
    assert isinstance(blended_weights, pd.DataFrame)
    assert isinstance(r11_weights, pd.DataFrame)
    assert isinstance(blended_daily, pd.DataFrame)
    assert isinstance(blended_slippage, pd.Series)
    assert isinstance(r11_slippage, pd.Series)
    signals = causal_reference_signals(closes, reference)
    diagnostics = reference_governor_state(signals, candidate)
    weights, daily, diagnostics = apply_governor(
        blended_weights,
        r11_weights,
        blended_daily,
        diagnostics,
        WEIGHT_CAP_DEFINITION,
    )
    active = diagnostics["state"].ne("normal")
    extra_slippage = blended_slippage.reindex(weights.index).copy()
    extra_slippage.loc[active] = r11_slippage.reindex(weights.index).loc[active]
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
        extra_slippage=extra_slippage,
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    return trial, diagnostics.reindex(trial.index)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    pre_settings, _ = build_pre2008_settings()
    samples = {"pre2008": pre_settings, **r21._build_samples()}
    components = {
        sample: base_components(settings, scenario)
        for sample, settings in samples.items()
    }
    metrics: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    neighborhood: list[dict[str, object]] = []
    for sample, settings in samples.items():
        trial, diagnostics = simulate_candidate(
            settings, scenario, components[sample], CENTRAL
        )
        trial.to_csv(OUTPUT / f"{sample}_central_daily.csv", index_label="date")
        diagnostics.to_csv(
            OUTPUT / f"{sample}_central_diagnostics.csv", index_label="date"
        )
        reference = components[sample]["baseline"]
        assert isinstance(reference, pd.DataFrame)
        if sample == "pre2008":
            metrics.append(
                metric_row(
                    sample,
                    "complete_1931_2007",
                    ("1931-01-02", "2007-12-31"),
                    reference["net_return"],
                    trial["net_return"],
                )
            )
            events.extend(
                pre2008_event_rows(
                    CENTRAL.name,
                    reference["net_return"],
                    trial["net_return"],
                )
            )
        else:
            periods = settings["periods"]
            assert isinstance(periods, dict)
            for period, bounds in periods.items():
                metrics.append(
                    metric_row(
                        sample,
                        period,
                        bounds,
                        reference["net_return"],
                        trial["net_return"],
                    )
                )

    for candidate in NEIGHBORS:
        for sample in ("pre2008", "normal_synthetic", "proxy_synthetic"):
            settings = samples[sample]
            trial, diagnostics = simulate_candidate(
                settings, scenario, components[sample], candidate
            )
            reference = components[sample]["baseline"]
            assert isinstance(reference, pd.DataFrame)
            if sample == "pre2008":
                complete = metric_row(
                    sample,
                    "complete_1931_2007",
                    ("1931-01-02", "2007-12-31"),
                    reference["net_return"],
                    trial["net_return"],
                )
                dotcom = next(
                    row
                    for row in pre2008_event_rows(
                        candidate.name,
                        reference["net_return"],
                        trial["net_return"],
                    )
                    if row["event"] == "dotcom_2000_2002"
                )
            else:
                periods = settings["periods"]
                assert isinstance(periods, dict)
                complete_name = next(
                    name for name in periods if name.startswith("complete_")
                )
                complete = metric_row(
                    sample,
                    complete_name,
                    periods[complete_name],
                    reference["net_return"],
                    trial["net_return"],
                )
                dotcom = {"candidate_max_drawdown": np.nan}
            neighborhood.append(
                {
                    "candidate": candidate.name,
                    "sample": sample,
                    "cagr_delta": complete["cagr_delta"],
                    "candidate_max_drawdown": complete["candidate_max_drawdown"],
                    "dotcom_max_drawdown": dotcom["candidate_max_drawdown"],
                    "active_fraction": float(
                        diagnostics["state"].ne("normal").mean()
                    ),
                }
            )

    metric_frame = pd.DataFrame(metrics)
    event_frame = pd.DataFrame(events)
    neighbor_frame = pd.DataFrame(neighborhood)
    metric_frame.to_csv(OUTPUT / "metrics.csv", index=False)
    event_frame.to_csv(OUTPUT / "pre2008_events.csv", index=False)
    neighbor_frame.to_csv(OUTPUT / "parameter_neighborhood.csv", index=False)
    dotcom = event_frame.loc[event_frame["event"].eq("dotcom_2000_2002")].iloc[0]
    pre = metric_frame.loc[metric_frame["sample"].eq("pre2008")].iloc[0]
    normal = metric_frame.loc[
        metric_frame["sample"].eq("normal_synthetic")
        & metric_frame["period"].str.startswith("complete_")
    ].iloc[0]
    proxy = metric_frame.loc[
        metric_frame["sample"].eq("proxy_synthetic")
        & metric_frame["period"].str.startswith("complete_")
    ].iloc[0]
    gates = [
        ("dotcom_mdd_at_or_above_minus_20pct", dotcom["candidate_max_drawdown"] >= -0.20),
        ("all_named_events_at_or_above_minus_20pct", event_frame["absolute_20pct_ceiling_pass"].all()),
        ("complete_pre2008_mdd_at_or_above_minus_25pct", pre["candidate_max_drawdown"] >= -0.25),
        ("normal_cagr_cost_at_most_3pct", normal["cagr_delta"] >= -0.03),
        ("normal_drawdown_not_worse", normal["max_drawdown_delta"] >= 0.0),
        ("proxy_cagr_cost_at_most_3pct", proxy["cagr_delta"] >= -0.03),
        ("proxy_drawdown_not_worse", proxy["max_drawdown_delta"] >= 0.0),
        (
            "neighbor_dotcom_paths_at_or_above_minus_25pct",
            neighbor_frame.loc[
                neighbor_frame["sample"].eq("pre2008"), "dotcom_max_drawdown"
            ].ge(-0.25).all(),
        ),
    ]
    acceptance = pd.DataFrame(gates, columns=["gate", "passed"])
    research_pass = bool(acceptance["passed"].all())
    acceptance.loc[len(acceptance)] = ["research_pass", research_pass]
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate": CENTRAL.__dict__,
        "research_pass": research_pass,
        "production_eligible": False,
        "production_changed": False,
        "orders_generated": False,
        "dotcom_baseline_max_drawdown": float(dotcom["baseline_max_drawdown"]),
        "dotcom_candidate_max_drawdown": float(dotcom["candidate_max_drawdown"]),
        "pre2008_complete_candidate_max_drawdown": float(pre["candidate_max_drawdown"]),
        "normal_complete_cagr_delta": float(normal["cagr_delta"]),
        "proxy_complete_cagr_delta": float(proxy["cagr_delta"]),
        "prior_candidate_output": str(BEAR_OUTPUT),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))
    print("\nPre-2008 events:")
    print(event_frame.round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

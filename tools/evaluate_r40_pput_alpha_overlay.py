from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import tools.evaluate_bear_recovery_governor as bear
from tools.evaluate_bear_recovery_governor import base_components, metric_row
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r10_gde_capital_efficiency import simulate_gde_substitution
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
)
from tools.evaluate_r12_volatility_managed_risk import GDE_FRACTION
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r40_price_shock_multiplier import causal_price_stress_signals
from tools.evaluate_r40_trend_conditioned_cap import (
    TrendCapCandidate,
    TrendConditionedCapPolicy,
)


OUTPUT = Path("output/r40_pput_alpha_overlay")
PPUT_HISTORY = Path("data/reference/cboe_pput_history.csv")
OVERLAY_NOTIONALS = (0.08, 0.085, 0.09, 0.095, 0.10)
ANNUAL_IMPLEMENTATION_DRAG = 0.005
BASE_CANDIDATE = TrendCapCandidate(
    "trend_cap_base",
    normal_non_cash_cap=1.20,
    floor_drawdown=-0.19,
    bull_multiplier=100.0,
    bear_multiplier=20.0,
    tier_size=0.05,
    five_day_stress_threshold=-0.07,
    shock_bear_multiplier=20.0,
    extension_drawdown_gate=-1.0,
)


@dataclass(frozen=True)
class PputOverlayCandidate:
    name: str
    overlay_notional: float
    annual_implementation_drag: float = ANNUAL_IMPLEMENTATION_DRAG

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlay_notional <= 0.10:
            raise ValueError("overlay_notional must be in [0, 0.10]")
        if self.annual_implementation_drag < 0.0:
            raise ValueError("annual_implementation_drag cannot be negative")


def load_pput_history(path: Path = PPUT_HISTORY) -> pd.Series:
    frame = pd.read_csv(path)
    required = {"DATE", "PPUT"}
    if not required.issubset(frame.columns):
        raise ValueError(f"PPUT history missing columns: {sorted(required - set(frame))}")
    dates = pd.to_datetime(frame["DATE"], format="%m/%d/%Y")
    result = pd.Series(
        frame["PPUT"].astype(float).to_numpy(),
        index=dates,
        name="PPUT",
    ).sort_index()
    if result.index.has_duplicates:
        raise ValueError("PPUT history contains duplicate dates")
    return result


def pput_alpha_overlay_return(
    closes: pd.DataFrame,
    pput: pd.Series,
    candidate: PputOverlayCandidate,
) -> tuple[pd.Series, pd.Series]:
    if "SPX" not in closes:
        raise ValueError("SPX prices are required for PPUT alpha")
    pput_return = pput.pct_change(fill_method=None).reindex(closes.index)
    spx_return = closes["SPX"].pct_change(fill_method=None)
    available = pput_return.notna() & spx_return.notna()
    alpha = (pput_return - spx_return).where(available, 0.0)
    implementation_drag = (
        candidate.overlay_notional
        * candidate.annual_implementation_drag
        / 252.0
    )
    overlay = (
        candidate.overlay_notional * alpha
        - available.astype(float) * implementation_drag
    )
    overlay.name = "pput_alpha_overlay_return"
    available.name = "pput_data_available"
    return overlay, available


def simulate_candidate(
    settings,
    scenario,
    components,
    candidate,
    pput,
    base_candidate=BASE_CANDIDATE,
):
    closes = settings["closes"]
    opens = settings["opens"]
    weights = components["blended_weights"]
    r11_weights = components["r11_weights"]
    daily = components["blended_daily"]
    slippage = components["blended_slippage"]
    assert isinstance(closes, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(r11_weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(slippage, pd.Series)
    trend_signals = bear.causal_bear_signals(closes).join(
        causal_price_stress_signals(
            closes,
            five_day_stress_threshold=base_candidate.five_day_stress_threshold,
        )[["stress"]]
    )
    policy = TrendConditionedCapPolicy(
        base_candidate,
        trend_signals,
        r11_weights,
    )
    overlay, available = pput_alpha_overlay_return(closes, pput, candidate)
    adjusted_slippage = slippage.reindex(weights.index).fillna(0.0) - overlay.reindex(
        weights.index
    ).fillna(0.0)
    trial = simulate_gde_substitution(
        str(settings["directory"]),
        opens,
        closes,
        substitution_fraction=GDE_FRACTION,
        gate_mode="growth_linked",
        gde_return_mode=str(settings["mode"]),
        start_date=str(settings["start"]),
        end_date=None,
        base_one_way_cost_bps=scenario.base_one_way_cost_bps,
        gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
        financing_spread_bps=scenario.financing_spread_bps,
        weights_override=weights,
        daily_override=daily,
        extra_slippage=adjusted_slippage,
        gde_no_trade_band=GDE_NO_TRADE_BAND,
        target_policy=policy,
    )
    trial["pput_alpha_overlay_return"] = overlay.reindex(trial.index).fillna(0.0)
    trial["pput_data_available"] = available.reindex(trial.index).fillna(False)
    return trial


def complete_bounds(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(
        (name for name in periods if name.startswith("complete_")),
        next(iter(periods)),
    )
    return name, periods[name]


def evaluate(
    sample,
    settings,
    scenario,
    components,
    candidate,
    pput,
    base_candidate=BASE_CANDIDATE,
):
    trial = simulate_candidate(
        settings,
        scenario,
        components,
        candidate,
        pput,
        base_candidate,
    )
    baseline = components["baseline"]
    assert isinstance(baseline, pd.DataFrame)
    period, bounds = complete_bounds(settings)
    return trial, {
        "candidate": candidate.name,
        "sample": sample,
        "overlay_notional": candidate.overlay_notional,
        "annual_implementation_drag": candidate.annual_implementation_drag,
        "pput_available_fraction": float(trial["pput_data_available"].mean()),
        "mean_daily_overlay_return": float(
            trial["pput_alpha_overlay_return"].mean()
        ),
        **metric_row(
            sample,
            period,
            bounds,
            baseline["net_return"],
            trial["net_return"],
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not PPUT_HISTORY.exists():
        raise FileNotFoundError(
            "Official Cboe PPUT history is required at " f"{PPUT_HISTORY}"
        )
    pput = load_pput_history()
    scenario = COST_SCENARIOS[0]
    samples = r21._build_samples()
    settings_by_sample = {
        sample: samples[sample]
        for sample in ("normal_synthetic", "proxy_synthetic")
    }
    components_by_sample = {
        sample: base_components(settings, scenario)
        for sample, settings in settings_by_sample.items()
    }
    candidates = tuple(
        PputOverlayCandidate(f"pput_alpha_{int(weight * 1000):03d}", weight)
        for weight in OVERLAY_NOTIONALS
    )
    rows: list[dict[str, object]] = []
    modern_paths: dict[str, pd.DataFrame] = {}
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            trial, row = evaluate(
                sample,
                settings,
                scenario,
                components_by_sample[sample],
                candidate,
                pput,
            )
            rows.append(row)
            if sample == "normal_synthetic":
                modern_paths[candidate.name] = trial
    screen = pd.DataFrame(rows)
    screen.to_csv(OUTPUT / "screen_metrics.csv", index=False)
    modern = screen.loc[screen["sample"].eq("normal_synthetic")]
    proxy = screen.loc[screen["sample"].eq("proxy_synthetic")]
    frontier = modern.merge(
        proxy[["candidate", "candidate_cagr", "candidate_max_drawdown"]],
        on="candidate",
        suffixes=("_modern", "_proxy"),
    )
    selected = set(
        frontier.loc[
            frontier["candidate_cagr_modern"].between(0.24, 0.25)
            & frontier["candidate_max_drawdown_modern"].ge(-0.20)
            & frontier["candidate_cagr_proxy"].ge(0.14)
            & frontier["candidate_max_drawdown_proxy"].ge(-0.20),
            "candidate",
        ]
    )
    pre_rows: list[dict[str, object]] = []
    pre_paths: dict[str, pd.DataFrame] = {}
    if selected:
        pre_settings, _ = build_pre2008_settings()
        pre_settings = {
            **pre_settings,
            "periods": {
                "complete_1931_2007": ("1931-01-02", "2007-12-31")
            },
        }
        pre_components = base_components(pre_settings, scenario)
        for candidate in candidates:
            if candidate.name not in selected:
                continue
            print(f"crash replay {candidate.name}", flush=True)
            trial, row = evaluate(
                "pre2008",
                pre_settings,
                scenario,
                pre_components,
                candidate,
                pput,
            )
            pre_rows.append(row)
            pre_paths[candidate.name] = trial
    pre = pd.DataFrame(pre_rows)
    pre.to_csv(OUTPUT / "pre2008_metrics.csv", index=False)
    if pre.empty:
        frontier["pre2008_max_drawdown"] = float("nan")
    else:
        frontier = frontier.merge(
            pre[["candidate", "candidate_max_drawdown"]],
            on="candidate",
            how="left",
        ).rename(columns={"candidate_max_drawdown": "pre2008_max_drawdown"})
    frontier["target_pass"] = (
        frontier["candidate"].isin(selected)
        & frontier["pre2008_max_drawdown"].ge(-0.20)
    )
    frontier.to_csv(OUTPUT / "frontier.csv", index=False)
    for name in frontier.loc[frontier["target_pass"], "candidate"]:
        modern_paths[name].to_csv(OUTPUT / f"normal_synthetic_{name}_daily.csv")
        pre_paths[name].to_csv(OUTPUT / f"pre2008_{name}_daily.csv")
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "pput_source": "Cboe PPUT official daily history",
        "pput_first_date": str(pput.index.min().date()),
        "pre_pput_overlay_return": 0.0,
        "candidate_count": len(candidates),
        "pre2008_replay_count": len(pre),
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

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_bear_recovery_governor import base_components
from tools.evaluate_pre2008_crash_replay import build_settings as build_pre2008_settings
from tools.evaluate_r10_gde_capital_efficiency import ASSETS, simulate_gde_substitution
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
)
from tools.evaluate_r12_volatility_managed_risk import GDE_FRACTION
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r40_term_trend_capacity as term_trend
from tools.evaluate_recursive_cushion_budget import (
    RecursiveCushionPolicy,
    simulate_cushion_candidate,
)
from tools.evaluate_recursive_trend_cushion_production_candidate import (
    CENTRAL as R40_CENTRAL,
)


OUTPUT = Path("output/r40_highwater_satellite")
MODERN_TERM_PRICES = Path("data/prices_vix_hedge.csv")
PRE2008_TERM_PRICES = Path("data/prices_pre2008_crash_proxy.csv")
SATELLITE_WEIGHTS = (0.15, 0.20, 0.25)
ACCOUNT_DRAWDOWN_GATES = (-0.10,)
SATELLITE_BASKETS = (
    ("qqq_gold", 0.00, 0.00, 0.50),
    ("spx_gold", 0.00, 0.50, 0.50),
    ("growth_gold", 0.20, 0.00, 0.40),
)


@dataclass(frozen=True)
class SatelliteCandidate:
    name: str
    satellite_weight: float
    account_drawdown_gate: float
    semis_share: float
    spx_share: float = 0.0
    gold_share: float = 0.0
    term_premium_threshold: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.satellite_weight <= 0.25:
            raise ValueError("satellite_weight must be in [0, 0.25]")
        if not -0.10 <= self.account_drawdown_gate <= 0.0:
            raise ValueError("account_drawdown_gate must be in [-0.10, 0]")
        if not 0.0 <= self.semis_share <= 1.0:
            raise ValueError("semis_share must be in [0, 1]")
        if not 0.0 <= self.spx_share <= 1.0:
            raise ValueError("spx_share must be in [0, 1]")
        if not 0.0 <= self.gold_share <= 1.0:
            raise ValueError("gold_share must be in [0, 1]")
        if self.semis_share + self.spx_share + self.gold_share > 1.0 + 1e-12:
            raise ValueError("satellite asset shares cannot exceed one")
        if self.term_premium_threshold < 1.0:
            raise ValueError("term_premium_threshold must be at least one")


class HighWaterSatellitePolicy:
    def __init__(
        self,
        base_policy: RecursiveCushionPolicy,
        permission: pd.Series,
        candidate: SatelliteCandidate,
    ) -> None:
        self.base_policy = base_policy
        self.permission = permission.astype(bool)
        self.candidate = candidate
        self.enabled = False

    def __call__(
        self,
        date: pd.Timestamp,
        blended_target: pd.Series,
        prior_equity: float,
        prior_peak: float,
    ) -> tuple[pd.Series, bool, dict[str, float | int | bool | str]]:
        target, base_force_trade, metadata = self.base_policy(
            date,
            blended_target,
            prior_equity,
            prior_peak,
        )
        prior_drawdown = prior_equity / prior_peak - 1.0
        permitted = bool(
            self.permission.get(date, False)
            and metadata["state"] == "normal"
            and prior_drawdown >= self.candidate.account_drawdown_gate
        )
        extra = self.candidate.satellite_weight if permitted else 0.0
        if extra > 0.0:
            semis_extra = extra * self.candidate.semis_share
            spx_extra = extra * self.candidate.spx_share
            gold_extra = extra * self.candidate.gold_share
            qqq_extra = extra - semis_extra - spx_extra - gold_extra
            target.loc["QQQ"] += qqq_extra
            target.loc["SEMIS"] += semis_extra
            target.loc["SPX"] += spx_extra
            target.loc["GOLD"] += gold_extra
            target.loc["CASH"] -= extra
        if abs(float(target.sum()) - 1.0) > 1e-10:
            raise AssertionError("satellite target weights do not sum to one")
        state_changed = permitted != self.enabled
        self.enabled = permitted
        result_metadata = dict(metadata)
        result_metadata.update(
            {
                "satellite_enabled": permitted,
                "satellite_permission": bool(self.permission.get(date, False)),
                "satellite_weight": extra,
                "satellite_qqq_weight": extra
                * (
                    1.0
                    - self.candidate.semis_share
                    - self.candidate.spx_share
                    - self.candidate.gold_share
                ),
                "satellite_semis_weight": extra * self.candidate.semis_share,
                "satellite_spx_weight": extra * self.candidate.spx_share,
                "satellite_gold_weight": extra * self.candidate.gold_share,
                "satellite_account_gate": self.candidate.account_drawdown_gate,
            }
        )
        return target, bool(base_force_trade or state_changed), result_metadata


def attach_term_prices(
    settings: dict[str, object],
    term_prices: pd.DataFrame,
) -> dict[str, object]:
    closes = settings["closes"]
    assert isinstance(closes, pd.DataFrame)
    term_closes = closes.loc[:, ["QQQ", "SEMIS"]].join(
        term_prices.loc[:, ["VIX", "VIX3M"]],
        how="left",
    )
    return {**settings, "term_closes": term_closes}


def simulate_satellite_candidate(settings, scenario, components, candidate):
    closes = settings["closes"]
    opens = settings["opens"]
    term_closes = settings["term_closes"]
    weights = components["blended_weights"]
    r11_weights = components["r11_weights"]
    daily = components["blended_daily"]
    slippage = components["blended_slippage"]
    assert isinstance(closes, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(term_closes, pd.DataFrame)
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(r11_weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(slippage, pd.Series)
    permission = term_trend.causal_term_trend_permission(
        term_closes,
        term_premium_threshold=candidate.term_premium_threshold,
    )
    base_policy = RecursiveCushionPolicy(
        R40_CENTRAL,
        term_trend.bear.causal_bear_signals(closes),
        r11_weights,
    )
    policy = HighWaterSatellitePolicy(base_policy, permission, candidate)
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
        extra_slippage=slippage,
        gde_no_trade_band=GDE_NO_TRADE_BAND,
        target_policy=policy,
    )
    return trial


def complete_period(settings: dict[str, object]) -> tuple[str, str | None]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(name for name in periods if name.startswith("complete_"))
    return periods[name]


def metric_row(
    sample: str,
    settings: dict[str, object],
    baseline: pd.Series,
    trial: pd.DataFrame,
    candidate: SatelliteCandidate,
) -> dict[str, object]:
    start, end = complete_period(settings)
    base = performance_metrics(baseline.loc[start:end])
    tested = performance_metrics(trial.loc[start:end, "net_return"])
    return {
        "candidate": candidate.name,
        "sample": sample,
        "satellite_weight": candidate.satellite_weight,
        "account_drawdown_gate": candidate.account_drawdown_gate,
        "semis_share": candidate.semis_share,
        "spx_share": candidate.spx_share,
        "gold_share": candidate.gold_share,
        "enabled_fraction": float(trial["satellite_enabled"].mean()),
        "baseline_cagr": base["cagr"],
        "candidate_cagr": tested["cagr"],
        "cagr_delta": tested["cagr"] - base["cagr"],
        "baseline_max_drawdown": base["max_drawdown"],
        "candidate_max_drawdown": tested["max_drawdown"],
        "baseline_sharpe": base["sharpe"],
        "candidate_sharpe": tested["sharpe"],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scenario = COST_SCENARIOS[0]
    samples = r21._build_samples()
    modern_term = pd.read_csv(
        MODERN_TERM_PRICES,
        index_col="date",
        parse_dates=True,
    )
    settings_by_sample = {
        sample: attach_term_prices(samples[sample], modern_term)
        for sample in ("normal_synthetic", "proxy_synthetic")
    }
    components = {
        sample: base_components(settings, scenario)
        for sample, settings in settings_by_sample.items()
    }
    baselines = {
        sample: simulate_cushion_candidate(
            settings,
            scenario,
            components[sample],
            R40_CENTRAL,
        )[0]
        for sample, settings in settings_by_sample.items()
    }
    candidates = tuple(
        SatelliteCandidate(
            f"sat{int(weight * 1000):03d}_dd{int(abs(gate) * 1000):03d}_{basket}",
            weight,
            gate,
            semis_share,
            spx_share,
            gold_share,
        )
        for weight in SATELLITE_WEIGHTS
        for gate in ACCOUNT_DRAWDOWN_GATES
        for basket, semis_share, spx_share, gold_share in SATELLITE_BASKETS
    )
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"screening {candidate.name}", flush=True)
        for sample, settings in settings_by_sample.items():
            trial = simulate_satellite_candidate(
                settings,
                scenario,
                components[sample],
                candidate,
            )
            rows.append(
                metric_row(
                    sample,
                    settings,
                    baselines[sample]["net_return"],
                    trial,
                    candidate,
                )
            )
    screen = pd.DataFrame(rows)
    screen.to_csv(OUTPUT / "screen_metrics.csv", index=False)
    modern = screen.loc[screen["sample"].eq("normal_synthetic")]
    proxy = screen.loc[screen["sample"].eq("proxy_synthetic")]
    frontier = modern.merge(
        proxy[
            [
                "candidate",
                "candidate_cagr",
                "candidate_max_drawdown",
                "enabled_fraction",
            ]
        ],
        on="candidate",
        suffixes=("_modern", "_proxy"),
    )
    selected = set(
        frontier.loc[
            frontier["candidate_cagr_modern"].between(0.24, 0.25)
            & frontier["candidate_max_drawdown_modern"].ge(-0.20)
            & frontier["candidate_max_drawdown_proxy"].ge(-0.20)
            & frontier["candidate_cagr_proxy"].ge(
                frontier["baseline_cagr"] - 0.02
            ),
            "candidate",
        ]
    )
    pre_rows: list[dict[str, object]] = []
    if selected:
        pre_settings, _ = build_pre2008_settings()
        pre_term = pd.read_csv(
            PRE2008_TERM_PRICES,
            index_col="date",
            parse_dates=True,
        )
        pre_settings = attach_term_prices(pre_settings, pre_term)
        pre_components = base_components(pre_settings, scenario)
        baseline_pre, _ = simulate_cushion_candidate(
            pre_settings,
            scenario,
            pre_components,
            R40_CENTRAL,
        )
        for candidate in candidates:
            if candidate.name not in selected:
                continue
            print(f"crash replay {candidate.name}", flush=True)
            trial = simulate_satellite_candidate(
                pre_settings,
                scenario,
                pre_components,
                candidate,
            )
            pre_rows.append(
                metric_row(
                    "pre2008",
                    {
                        **pre_settings,
                        "periods": {
                            "complete_1931_2007": ("1931-01-02", "2007-12-31")
                        },
                    },
                    baseline_pre["net_return"],
                    trial,
                    candidate,
                )
            )
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
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
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

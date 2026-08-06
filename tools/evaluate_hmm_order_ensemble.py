from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import (
    load_strategy_inputs,
    metric_delta,
)
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as r39


OUTPUT = Path("output/hmm_order_ensemble_validation")
CANDIDATE_METADATA: dict[str, object] = {"ensemble_orders": [2, 3, 4]}
DEFINITIONS = {
    "normal": {
        "sample": "normal_synthetic",
        "baseline": Path("output/research_hmm_gaussian_diagnostics"),
        "candidate": Path("output/research_hmm_order_ensemble_234"),
        "periods": {
            "development_2015_2021": ("2015-01-01", "2021-12-31"),
            "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
            "complete_2015_2025": ("2015-01-01", "2025-12-31"),
            "post_holdout_2026": ("2026-01-01", "2026-12-31"),
        },
    },
    "proxy": {
        "sample": "proxy_synthetic",
        "baseline": Path(
            "output/research_hmm_gaussian_diagnostics_20y_proxy"
        ),
        "candidate": Path(
            "output/research_hmm_order_ensemble_234_20y_proxy"
        ),
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_2025": ("2015-01-01", "2025-12-31"),
            "complete_2006_2025": ("2006-08-01", "2025-12-31"),
            "post_holdout_2026": ("2026-01-01", "2026-12-31"),
        },
    },
}


def with_strategy_inputs(
    settings: dict[str, object], directory: Path
) -> dict[str, object]:
    weights, daily = load_strategy_inputs(str(directory.relative_to("output")))
    updated = deepcopy(settings)
    updated["weights"] = weights
    updated["daily"] = daily
    updated["directory"] = directory.name
    return updated


def doubled_trading_cost(frame: pd.DataFrame) -> pd.Series:
    """Double trading cost while leaving financing and returns unchanged."""
    return frame["net_return"] - frame["trading_cost"]


def state_switch_rows(
    sample: str,
    baseline_directory: Path,
    candidate_directory: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for baseline_path in sorted(
        (baseline_directory / "members").glob("seed_*/regimes.csv")
    ):
        candidate_path = (
            candidate_directory
            / "members"
            / baseline_path.parent.name
            / "regimes.csv"
        )
        baseline = pd.read_csv(baseline_path, index_col=0, parse_dates=True)
        candidate = pd.read_csv(candidate_path, index_col=0, parse_dates=True)
        common = baseline.index.intersection(candidate.index)

        def switches(frame: pd.DataFrame) -> int:
            values = frame.loc[common, "paper_risk_on_candidate"].astype(int)
            return int(values.ne(values.shift(1)).iloc[1:].sum())

        rows.append(
            {
                "sample": sample,
                "member": baseline_path.parent.name,
                "selected_order_switches": switches(baseline),
                "candidate_switches": switches(candidate),
            }
        )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    metric_rows: list[dict[str, object]] = []
    switch_rows: list[dict[str, object]] = []
    for label, definition in DEFINITIONS.items():
        sample_name = str(definition["sample"])
        baseline_directory = definition["baseline"]
        candidate_directory = definition["candidate"]
        periods = definition["periods"]
        assert isinstance(baseline_directory, Path)
        assert isinstance(candidate_directory, Path)
        assert isinstance(periods, dict)
        baseline_settings = with_strategy_inputs(
            samples[sample_name], baseline_directory
        )
        candidate_settings = with_strategy_inputs(
            samples[sample_name], candidate_directory
        )
        for scenario in COST_SCENARIOS:
            baseline, _ = r39.simulate_candidate(baseline_settings, scenario)
            candidate, _ = r39.simulate_candidate(candidate_settings, scenario)
            common = baseline.index.intersection(candidate.index)
            for period, (start, end) in periods.items():
                selected = common[(common >= start) & (common <= end)]
                for cost_view, transform in (
                    ("reported", lambda frame: frame["net_return"]),
                    ("double_trading_cost", doubled_trading_cost),
                ):
                    metric_rows.append(
                        {
                            "sample": label,
                            "scenario": scenario.name,
                            "cost_view": cost_view,
                            "period": period,
                            **metric_delta(
                                transform(baseline.loc[selected]),
                                transform(candidate.loc[selected]),
                            ),
                        }
                    )
            if scenario.name == "current_liquidity":
                baseline.to_csv(OUTPUT / f"{label}_baseline_r39_daily.csv")
                candidate.to_csv(OUTPUT / f"{label}_candidate_r39_daily.csv")
        switch_rows.extend(
            state_switch_rows(
                label, baseline_directory, candidate_directory
            )
        )

    metrics = pd.DataFrame(metric_rows)
    switches = pd.DataFrame(switch_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    switches.to_csv(OUTPUT / "state_switches.csv", index=False)

    def delta(
        sample: str,
        period: str,
        field: str,
        *,
        scenario: str = "current_liquidity",
        cost_view: str = "reported",
    ) -> float:
        return float(
            metrics.loc[
                metrics["sample"].eq(sample)
                & metrics["period"].eq(period)
                & metrics["scenario"].eq(scenario)
                & metrics["cost_view"].eq(cost_view),
                field,
            ].iloc[0]
        )

    gates = {
        "development_cagr_nonnegative": delta(
            "normal", "development_2015_2021", "cagr_delta"
        )
        >= 0.0,
        "holdout_cagr_nonnegative": delta(
            "normal", "holdout_2022_2025", "cagr_delta"
        )
        >= 0.0,
        "normal_drawdown_within_50bp": delta(
            "normal", "complete_2015_2025", "max_drawdown_delta"
        )
        >= -0.005,
        "proxy_early_cagr_nonnegative": delta(
            "proxy", "early_2006_2014", "cagr_delta"
        )
        >= 0.0,
        "proxy_late_cagr_nonnegative": delta(
            "proxy", "late_2015_2025", "cagr_delta"
        )
        >= 0.0,
        "proxy_complete_cagr_nonnegative": delta(
            "proxy", "complete_2006_2025", "cagr_delta"
        )
        >= 0.0,
        "double_cost_complete_cagr_nonnegative": delta(
            "normal",
            "complete_2015_2025",
            "cagr_delta",
            cost_view="double_trading_cost",
        )
        >= 0.0,
        "state_switches_not_increased": bool(
            switches["candidate_switches"].sum()
            <= switches["selected_order_switches"].sum()
        ),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        **CANDIDATE_METADATA,
        "gates": gates,
        "research_pass": bool(all(gates.values())),
        "decision": (
            "continue_forward_shadow"
            if all(gates.values())
            else "reject_current_specification"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

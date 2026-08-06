from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_hmm_order_ensemble import DEFINITIONS, with_strategy_inputs
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as r39


OUTPUT = Path("output/hmm_eighth_wave_robustness")
CANDIDATES = {
    "robust_scaler": {
        "normal": Path("output/research_hmm_robust_scaler"),
        "proxy": Path("output/research_hmm_robust_scaler_20y_proxy"),
    },
    "empirical_bayes_means": {
        "normal": Path("output/research_hmm_empirical_bayes_means"),
        "proxy": Path("output/research_hmm_empirical_bayes_means_20y_proxy"),
    },
}
EVENTS = {
    "gfc": (pd.Timestamp("2007-10-09"), pd.Timestamp("2009-03-09")),
    "euro_stress": (pd.Timestamp("2011-04-29"), pd.Timestamp("2011-10-03")),
    "q4_2018": (pd.Timestamp("2018-09-20"), pd.Timestamp("2018-12-24")),
    "covid": (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
    "tightening_2022": (pd.Timestamp("2022-01-03"), pd.Timestamp("2022-10-14")),
}


def _delta_row(
    candidate: str,
    sample: str,
    period: str,
    baseline: pd.DataFrame,
    trial: pd.DataFrame,
    selected: pd.Index,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "sample": sample,
        "period": period,
        "start": selected[0].date().isoformat(),
        "end": selected[-1].date().isoformat(),
        "observations": len(selected),
        **metric_delta(
            baseline.loc[selected, "net_return"],
            trial.loc[selected, "net_return"],
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    scenario = next(item for item in COST_SCENARIOS if item.name == "current_liquidity")
    paths: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for sample, definition in DEFINITIONS.items():
        sample_name = str(definition["sample"])
        baseline_directory = definition["baseline"]
        assert isinstance(baseline_directory, Path)
        baseline_settings = with_strategy_inputs(samples[sample_name], baseline_directory)
        baseline, _ = r39.simulate_candidate(baseline_settings, scenario)
        for candidate, directories in CANDIDATES.items():
            settings = with_strategy_inputs(samples[sample_name], directories[sample])
            trial, _ = r39.simulate_candidate(settings, scenario)
            paths[(candidate, sample)] = (baseline, trial)

    annual_rows: list[dict[str, object]] = []
    rolling_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for (candidate, sample), (baseline, trial) in paths.items():
        common = baseline.index.intersection(trial.index)
        for year in sorted(set(common.year)):
            selected = common[common.year == year]
            if len(selected) >= 100:
                annual_rows.append(
                    _delta_row(candidate, sample, str(year), baseline, trial, selected)
                )
        if len(common) >= 756:
            for end_position in range(755, len(common), 21):
                selected = common[end_position - 755 : end_position + 1]
                rolling_rows.append(
                    _delta_row(
                        candidate,
                        sample,
                        f"rolling_{selected[-1].date().isoformat()}",
                        baseline,
                        trial,
                        selected,
                    )
                )
        for event, (start, end) in EVENTS.items():
            selected = common[(common >= start) & (common <= end)]
            if len(selected) >= 20:
                event_rows.append(
                    _delta_row(candidate, sample, event, baseline, trial, selected)
                )

    annual = pd.DataFrame(annual_rows)
    rolling = pd.DataFrame(rolling_rows)
    events = pd.DataFrame(event_rows)
    annual.to_csv(OUTPUT / "annual_deltas.csv", index=False)
    rolling.to_csv(OUTPUT / "rolling_756_deltas.csv", index=False)
    events.to_csv(OUTPUT / "event_deltas.csv", index=False)

    summaries: dict[str, dict[str, float | bool]] = {}
    for candidate in CANDIDATES:
        annual_selected = annual.loc[
            annual["candidate"].eq(candidate) & annual["sample"].eq("proxy")
        ]
        rolling_selected = rolling.loc[
            rolling["candidate"].eq(candidate) & rolling["sample"].eq("proxy")
        ]
        event_selected = events.loc[
            events["candidate"].eq(candidate) & events["sample"].eq("proxy")
        ]
        proxy_baseline, proxy_trial = paths[(candidate, "proxy")]
        common = proxy_baseline.index.intersection(proxy_trial.index)
        proxy_metrics = metric_delta(
            proxy_baseline.loc[common, "net_return"],
            proxy_trial.loc[common, "net_return"],
        )
        values: dict[str, float | bool] = {
            "proxy_max_drawdown_delta": float(proxy_metrics["max_drawdown_delta"]),
            "annual_median_cagr_delta": float(annual_selected["cagr_delta"].median()),
            "annual_nonnegative_fraction": float(
                annual_selected["cagr_delta"].ge(0.0).mean()
            ),
            "rolling_median_cagr_delta": float(
                rolling_selected["cagr_delta"].median()
            ),
            "rolling_nonnegative_fraction": float(
                rolling_selected["cagr_delta"].ge(0.0).mean()
            ),
            "rolling_worst_cagr_delta": float(
                rolling_selected["cagr_delta"].min()
            ),
            "event_median_cagr_delta": float(event_selected["cagr_delta"].median()),
            "event_worst_max_drawdown_delta": float(
                event_selected["max_drawdown_delta"].min()
            ),
        }
        gates = {
            "proxy_drawdown_within_50bp": values["proxy_max_drawdown_delta"] >= -0.005,
            "annual_median_nonnegative": values["annual_median_cagr_delta"] >= 0.0,
            "annual_majority_nonnegative": values["annual_nonnegative_fraction"] >= 0.5,
            "rolling_median_nonnegative": values["rolling_median_cagr_delta"] >= 0.0,
            "rolling_majority_nonnegative": values["rolling_nonnegative_fraction"] >= 0.5,
            "all_events_drawdown_within_50bp": values[
                "event_worst_max_drawdown_delta"
            ]
            >= -0.005,
            "event_median_cagr_nonnegative": values["event_median_cagr_delta"] >= 0.0,
        }
        values.update(gates)
        values["extended_pass"] = bool(all(gates.values()))
        summaries[candidate] = values

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        "candidates": summaries,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

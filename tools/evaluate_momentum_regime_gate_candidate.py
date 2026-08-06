from __future__ import annotations

import pandas as pd

import evaluate_lowvol_rebound_candidate as evaluation
from evaluate_open_execution import load_open_close, simulate_open_execution
from regime_strategy.report import performance_metrics


CANDIDATE = (
    "paper_core_growth_gold20_lowvol_momentum_floor_guard_netted_ensemble"
)
CANDIDATE_KEY = "momentum_regime_gate"
DESTINATION_NAME = "momentum_regime_gate_candidate_validation"
REPORT_TITLE = "High-volatility relative-momentum gate"


def write_open_execution_metrics() -> None:
    opens, closes = load_open_close(refresh=False)
    rows = []
    strategies = {
        "production": evaluation.BASELINE,
        "lowvol_rebound": (
            "paper_core_growth_gold20_lowvol_rebound_floor_guard_"
            "netted_ensemble"
        ),
        "momentum_regime_gate": CANDIDATE,
    }
    for cost_bps in (7.5, 15.0):
        for strategy, directory in strategies.items():
            simulated = simulate_open_execution(
                directory,
                opens,
                closes,
                cost_bps=cost_bps,
            )
            rows.append(
                {
                    "cost_bps": cost_bps,
                    "strategy": strategy,
                    **performance_metrics(simulated["net_return"]),
                    "average_open_turnover": float(
                        simulated["open_turnover"].mean()
                    ),
                }
            )
    pd.DataFrame(rows).set_index(
        ["cost_bps", "strategy"]
    ).to_csv(evaluation.DESTINATION / "open_execution_metrics.csv")


def main() -> None:
    evaluation.DESTINATION = (
        evaluation.OUTPUT / DESTINATION_NAME
    )
    evaluation.CANDIDATE = CANDIDATE
    evaluation.CANDIDATE_COST15 = f"{CANDIDATE}_cost15"
    evaluation.CANDIDATE_2012 = f"{CANDIDATE}_2012"
    evaluation.CANDIDATE_KEY = CANDIDATE_KEY
    evaluation.REPORT_TITLE = REPORT_TITLE
    evaluation.RECOVERY_FAMILY = {
        **evaluation.RECOVERY_FAMILY,
        "panic_veto": (
            "paper_core_growth_gold20_panic_veto_floor_guard_"
            "netted_ensemble"
        ),
        "crisis_veto10": (
            "paper_core_growth_gold20_crisis_veto10_floor_guard_"
            "netted_ensemble"
        ),
        "downside_semivol_combined": (
            "paper_core_growth_gold20_downside_semivol_floor_guard_"
            "netted_ensemble"
        ),
        "downside_semivol_rebound": (
            "paper_core_growth_gold20_downside_semivol_rebound_floor_guard_"
            "netted_ensemble"
        ),
        "downside_semivol_low_rebound": (
            "paper_core_growth_gold20_downside_semivol_low_rebound_"
            "floor_guard_netted_ensemble"
        ),
        "momentum_regime_gate": (
            "paper_core_growth_gold20_lowvol_momentum_floor_guard_"
            "netted_ensemble"
        ),
        "inverse_momentum_regime_gate": (
            "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
            "netted_ensemble"
        ),
        "zero_bridge_qqq20": (
            "paper_core_growth_gold20_zero_bridge_qqq20_momentum_gate_"
            "netted_ensemble"
        ),
        "dual_reentry": (
            "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_fast10": (
            "paper_core_growth_gold20_dual_reentry_fast10_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_fast40": (
            "paper_core_growth_gold20_dual_reentry_fast40_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_q85": (
            "paper_core_growth_gold20_dual_reentry_q85_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_q95": (
            "paper_core_growth_gold20_dual_reentry_q95_inverse_momentum_"
            "netted_ensemble"
        ),
        "dual_equal_bridge": (
            "paper_core_growth_gold20_dual_reentry_equal_bridge_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_hysteresis19": (
            "paper_core_growth_gold20_dual_reentry_hysteresis19_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_goodvol": (
            "paper_core_growth_gold20_dual_reentry_goodvol_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_goodvol_bear20": (
            "paper_core_growth_gold20_dual_reentry_goodvol_bear20_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_goodvol_bear20_contango": (
            "paper_core_growth_gold20_dual_reentry_goodvol_bear20_"
            "contango_inverse_momentum_netted_ensemble"
        ),
        "dual_parent_correction_veto": (
            "paper_core_growth_gold20_dual_reentry_parent_correction_veto_"
            "inverse_momentum_netted_ensemble"
        ),
        "dual_plain_momentum": (
            "paper_core_growth_gold20_dual_reentry_plain_momentum_"
            "netted_ensemble"
        ),
        CANDIDATE_KEY: CANDIDATE,
    }
    evaluation.main()
    write_open_execution_metrics()


if __name__ == "__main__":
    main()

from __future__ import annotations

import evaluate_lowvol_rebound_candidate as evaluation


CANDIDATE = (
    "paper_core_growth_gold20_panic_veto_floor_guard_netted_ensemble"
)


def main() -> None:
    evaluation.DESTINATION = (
        evaluation.OUTPUT / "panic_veto_candidate_validation"
    )
    evaluation.CANDIDATE = CANDIDATE
    evaluation.CANDIDATE_COST15 = f"{CANDIDATE}_cost15"
    evaluation.CANDIDATE_2012 = f"{CANDIDATE}_2012"
    evaluation.CANDIDATE_KEY = "panic_veto"
    evaluation.REPORT_TITLE = "High-volatility panic veto floor guard"
    evaluation.RECOVERY_FAMILY = {
        **evaluation.RECOVERY_FAMILY,
        "lowvol_rebound": (
            "paper_core_growth_gold20_lowvol_rebound_floor_guard_"
            "netted_ensemble"
        ),
        "panic_veto": CANDIDATE,
    }
    evaluation.main()


if __name__ == "__main__":
    main()

from __future__ import annotations

import evaluate_momentum_regime_gate_candidate as evaluation


def main() -> None:
    evaluation.CANDIDATE = (
        "paper_core_growth_gold20_dual_reentry_inverse_momentum_"
        "netted_ensemble"
    )
    evaluation.CANDIDATE_KEY = "dual_reentry"
    evaluation.DESTINATION_NAME = "dual_reentry_candidate_validation"
    evaluation.REPORT_TITLE = (
        "Dual-stage zero-entry and established-position re-entry"
    )
    evaluation.main()


if __name__ == "__main__":
    main()

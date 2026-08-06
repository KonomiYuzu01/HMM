from __future__ import annotations

import evaluate_momentum_regime_gate_candidate as evaluation


def main() -> None:
    evaluation.CANDIDATE = (
        "paper_core_growth_gold20_lowvol_inverse_momentum_floor_guard_"
        "netted_ensemble"
    )
    evaluation.CANDIDATE_KEY = "inverse_momentum_regime_gate"
    evaluation.DESTINATION_NAME = (
        "inverse_momentum_regime_gate_candidate_validation"
    )
    evaluation.REPORT_TITLE = (
        "High-volatility inverse-risk relative-momentum gate"
    )
    evaluation.main()


if __name__ == "__main__":
    main()

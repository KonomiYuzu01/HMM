from __future__ import annotations

import evaluate_floor45_dual_candidate as evaluation


CANDIDATE = (
    "paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
    "inverse_momentum_netted_ensemble"
)


def main() -> None:
    tested_family = evaluation.candidate_family()
    evaluation.CANDIDATE = CANDIDATE
    evaluation.CANDIDATE_PROXY = f"{CANDIDATE}_20y_proxy"
    evaluation.CANDIDATE_KEY = "r8_floor45_goodvol_bear20"
    evaluation.ARTIFACT_PREFIX = "r8"
    evaluation.candidate_family = lambda: {
        **tested_family,
        evaluation.CANDIDATE_KEY: CANDIDATE,
    }
    evaluation.main()


if __name__ == "__main__":
    main()

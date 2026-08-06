from __future__ import annotations

from pathlib import Path

import tools.evaluate_hmm_third_wave as evaluation


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_fourth_wave_validation")
    evaluation.CANDIDATES = {
        "adaptive_refit": {
            "normal": Path("output/research_hmm_adaptive_refit"),
            "proxy": Path("output/research_hmm_adaptive_refit_20y_proxy"),
        },
        "exclude_smh": {
            "normal": Path("output/research_hmm_exclude_smh"),
            "proxy": Path("output/research_hmm_exclude_smh_20y_proxy"),
        },
    }
    evaluation.main()


if __name__ == "__main__":
    main()

from copy import deepcopy
from pathlib import Path

import evaluate_hmm_third_wave as evaluation


def main() -> None:
    definitions = deepcopy(evaluation.DEFINITIONS)
    definitions["normal"]["baseline"] = Path(
        "output/research_hmm_standard_restarts20"
    )
    definitions["proxy"]["baseline"] = Path(
        "output/research_hmm_standard_restarts20_20y_proxy"
    )
    evaluation.DEFINITIONS = definitions
    evaluation.OUTPUT = Path("output/hmm_robust_restarts20_validation")
    evaluation.CANDIDATES = {
        "robust_restarts20": {
            "normal": Path("output/research_hmm_robust_restarts20"),
            "proxy": Path("output/research_hmm_robust_restarts20_20y_proxy"),
        }
    }
    evaluation.main()


if __name__ == "__main__":
    main()

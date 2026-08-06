from copy import deepcopy
from pathlib import Path

import evaluate_hmm_third_wave as evaluation


def main() -> None:
    definitions = deepcopy(evaluation.DEFINITIONS)
    definitions["normal"]["baseline"] = Path(
        "output/research_hmm_standard_restarts5"
    )
    definitions["proxy"]["baseline"] = Path(
        "output/research_hmm_standard_restarts5_20y_proxy"
    )
    evaluation.DEFINITIONS = definitions
    evaluation.OUTPUT = Path("output/hmm_fourteenth_wave_validation")
    evaluation.CANDIDATES = {
        "pca90": {
            "normal": Path("output/research_hmm_pca90"),
            "proxy": Path("output/research_hmm_pca90_20y_proxy"),
        }
    }
    evaluation.main()


if __name__ == "__main__":
    main()

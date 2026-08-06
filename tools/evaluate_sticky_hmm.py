from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import tools.evaluate_hmm_order_ensemble as evaluation


def main() -> None:
    definitions = deepcopy(evaluation.DEFINITIONS)
    definitions["normal"]["candidate"] = Path(
        "output/research_hmm_sticky10"
    )
    definitions["proxy"]["candidate"] = Path(
        "output/research_hmm_sticky10_20y_proxy"
    )
    evaluation.OUTPUT = Path("output/sticky_hmm_validation")
    evaluation.DEFINITIONS = definitions
    evaluation.CANDIDATE_METADATA = {"sticky_transition_prior": 10.0}
    evaluation.main()


if __name__ == "__main__":
    main()

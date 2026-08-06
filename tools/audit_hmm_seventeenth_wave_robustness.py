from copy import deepcopy
from pathlib import Path

import audit_hmm_eighth_wave_robustness as audit


def main() -> None:
    definitions = deepcopy(audit.DEFINITIONS)
    definitions["normal"]["baseline"] = Path(
        "output/research_hmm_standard_restarts5"
    )
    definitions["proxy"]["baseline"] = Path(
        "output/research_hmm_standard_restarts5_20y_proxy"
    )
    audit.DEFINITIONS = definitions
    audit.OUTPUT = Path("output/hmm_seventeenth_wave_robustness")
    audit.CANDIDATES = {
        "without_momentum": {
            "normal": Path("output/research_hmm_without_momentum_restarts5"),
            "proxy": Path(
                "output/research_hmm_without_momentum_restarts5_20y_proxy"
            ),
        }
    }
    audit.main()


if __name__ == "__main__":
    main()

from pathlib import Path

import evaluate_hmm_third_wave as evaluation
from evaluate_hmm_ninth_wave import _average_member_switch_rows


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_tenth_wave_validation")
    evaluation.state_switch_rows = _average_member_switch_rows
    evaluation.CANDIDATES = {
        "multi_lookback_disagreement": {
            "normal": Path("output/research_hmm_multi_lookback_disagreement"),
            "proxy": Path(
                "output/research_hmm_multi_lookback_disagreement_20y_proxy"
            ),
        }
    }
    evaluation.main()


if __name__ == "__main__":
    main()

from pathlib import Path

import evaluate_hmm_third_wave as evaluation


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_fifth_wave_validation")
    evaluation.CANDIDATES = {
        "transition_refit": {
            "normal": Path("output/research_hmm_transition_refit"),
            "proxy": Path("output/research_hmm_transition_refit_20y_proxy"),
        },
        "order_select252": {
            "normal": Path("output/research_hmm_order_select252"),
            "proxy": Path("output/research_hmm_order_select252_20y_proxy"),
        },
    }
    evaluation.main()


if __name__ == "__main__":
    main()

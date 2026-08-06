from pathlib import Path

import evaluate_hmm_third_wave as evaluation


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_seventh_wave_validation")
    evaluation.CANDIDATES = {
        "filtered_return_moments": {
            "normal": Path("output/research_hmm_filtered_return_moments"),
            "proxy": Path("output/research_hmm_filtered_return_moments_20y_proxy"),
        },
        "conditioned_order_validation": {
            "normal": Path("output/research_hmm_conditioned_order_validation"),
            "proxy": Path(
                "output/research_hmm_conditioned_order_validation_20y_proxy"
            ),
        },
    }
    evaluation.main()


if __name__ == "__main__":
    main()

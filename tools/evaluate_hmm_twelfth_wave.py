from pathlib import Path

import evaluate_hmm_third_wave as evaluation


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_twelfth_wave_validation")
    evaluation.CANDIDATES = {
        "tied_covariance": {
            "normal": Path("output/research_hmm_tied_covariance"),
            "proxy": Path("output/research_hmm_tied_covariance_20y_proxy"),
        }
    }
    evaluation.main()


if __name__ == "__main__":
    main()

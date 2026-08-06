from pathlib import Path

import evaluate_hmm_third_wave as evaluation


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_eighth_wave_validation")
    evaluation.CANDIDATES = {
        "robust_scaler": {
            "normal": Path("output/research_hmm_robust_scaler"),
            "proxy": Path("output/research_hmm_robust_scaler_20y_proxy"),
        },
        "empirical_bayes_means": {
            "normal": Path("output/research_hmm_empirical_bayes_means"),
            "proxy": Path("output/research_hmm_empirical_bayes_means_20y_proxy"),
        },
    }
    evaluation.main()


if __name__ == "__main__":
    main()

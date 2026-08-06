from pathlib import Path

import evaluate_hmm_third_wave as evaluation


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_sixth_wave_validation")
    evaluation.CANDIDATES = {
        "template_refit_only": {
            "normal": Path("output/research_hmm_template_refit_only"),
            "proxy": Path("output/research_hmm_template_refit_only_20y_proxy"),
        },
        "wasserstein_barycenter": {
            "normal": Path("output/research_hmm_wasserstein_barycenter"),
            "proxy": Path("output/research_hmm_wasserstein_barycenter_20y_proxy"),
        },
    }
    evaluation.main()


if __name__ == "__main__":
    main()

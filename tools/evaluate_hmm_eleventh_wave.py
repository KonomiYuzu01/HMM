from pathlib import Path

import evaluate_hmm_third_wave as evaluation


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_eleventh_wave_validation")
    evaluation.CANDIDATES = {
        "huber_return_location": {
            "normal": Path("output/research_hmm_huber_return_location"),
            "proxy": Path("output/research_hmm_huber_return_location_20y_proxy"),
        }
    }
    evaluation.main()


if __name__ == "__main__":
    main()

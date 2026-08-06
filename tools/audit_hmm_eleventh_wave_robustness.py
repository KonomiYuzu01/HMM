from pathlib import Path

import audit_hmm_eighth_wave_robustness as audit


def main() -> None:
    audit.OUTPUT = Path("output/hmm_eleventh_wave_robustness")
    audit.CANDIDATES = {
        "huber_return_location": {
            "normal": Path("output/research_hmm_huber_return_location"),
            "proxy": Path("output/research_hmm_huber_return_location_20y_proxy"),
        }
    }
    audit.main()


if __name__ == "__main__":
    main()

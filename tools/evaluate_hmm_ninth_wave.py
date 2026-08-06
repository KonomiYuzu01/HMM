from pathlib import Path

import evaluate_hmm_third_wave as evaluation
import pandas as pd


def _average_member_switch_rows(
    sample: str,
    baseline_directory: Path,
    candidate_directory: Path,
) -> list[dict[str, object]]:
    def average_switches(directory: Path) -> float:
        counts: list[int] = []
        for path in sorted((directory / "members").glob("*/regimes.csv")):
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
            values = frame["paper_risk_on_candidate"].astype(int)
            counts.append(int(values.ne(values.shift(1)).iloc[1:].sum()))
        if not counts:
            raise ValueError(f"No member regime histories found in {directory}")
        return float(sum(counts) / len(counts))

    return [
        {
            "sample": sample,
            "member": "per_member_average",
            "selected_order_switches": average_switches(baseline_directory),
            "candidate_switches": average_switches(candidate_directory),
        }
    ]


def main() -> None:
    evaluation.OUTPUT = Path("output/hmm_ninth_wave_validation")
    evaluation.state_switch_rows = _average_member_switch_rows
    evaluation.CANDIDATES = {
        "multi_lookback_756_1008_1512": {
            "normal": Path("output/research_hmm_multi_lookback_756_1008_1512"),
            "proxy": Path(
                "output/research_hmm_multi_lookback_756_1008_1512_20y_proxy"
            ),
        }
    }
    evaluation.main()


if __name__ == "__main__":
    main()

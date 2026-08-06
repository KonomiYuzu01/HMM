from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_drawdown_uncertainty import paired_circular_block_bootstrap


ROOT = Path("output")
DESTINATION = ROOT / "diversifier_drawdown_uncertainty"
REFERENCES = {
    "complete_2015_2025": "paper_core_zero_entry_growth_reallocation_ensemble",
    "extended_2012_2025": "paper_core_zero_entry_growth_reallocation_ensemble_2012",
}
CANDIDATES = {
    "gold_15": {
        "complete_2015_2025": "paper_core_zero_entry_growth_gold15_ensemble",
        "extended_2012_2025": "paper_core_zero_entry_growth_gold15_ensemble_2012",
    },
    "gold_20": {
        "complete_2015_2025": "paper_core_zero_entry_growth_gold20_ensemble",
        "extended_2012_2025": "paper_core_zero_entry_growth_gold20_ensemble_2012",
    },
    "gold_20_lev110": {
        "complete_2015_2025": "paper_core_zero_entry_growth_gold20_lev110_ensemble",
        "extended_2012_2025": "paper_core_zero_entry_growth_gold20_lev110_ensemble_2012",
    },
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample, reference_directory in REFERENCES.items():
        reference = pd.read_csv(
            ROOT / reference_directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        ).loc[:"2025", "net_return"]
        if sample == "complete_2015_2025":
            reference = reference.loc["2015":]
        for label, directories in CANDIDATES.items():
            candidate = pd.read_csv(
                ROOT / directories[sample] / "daily_returns.csv",
                index_col=0,
                parse_dates=True,
            ).loc[:"2025", "net_return"]
            if sample == "complete_2015_2025":
                candidate = candidate.loc["2015":]
            aligned = pd.concat([candidate, reference], axis=1, join="inner").dropna()
            for block_length in (21, 63, 126):
                simulations = paired_circular_block_bootstrap(
                    aligned.iloc[:, 0].to_numpy(),
                    aligned.iloc[:, 1].to_numpy(),
                    block_length,
                    simulations=5_000,
                    seed=20260723 + block_length,
                )
                candidate_drawdown = simulations["candidate_max_drawdown"]
                reference_drawdown = simulations["baseline_max_drawdown"]
                rows.append(
                    {
                        "strategy": label,
                        "sample": sample,
                        "block_length": block_length,
                        "median_cagr_delta": float(
                            simulations["cagr_delta"].median()
                        ),
                        "probability_cagr_higher": float(
                            (simulations["cagr_delta"] > 0.0).mean()
                        ),
                        "median_max_drawdown_delta": float(
                            simulations["max_drawdown_delta"].median()
                        ),
                        "probability_drawdown_better": float(
                            (simulations["max_drawdown_delta"] > 0.0).mean()
                        ),
                        "reference_probability_breach_18pct": float(
                            (reference_drawdown < -0.18).mean()
                        ),
                        "candidate_probability_breach_18pct": float(
                            (candidate_drawdown < -0.18).mean()
                        ),
                        "reference_median_max_drawdown": float(
                            reference_drawdown.median()
                        ),
                        "candidate_median_max_drawdown": float(
                            candidate_drawdown.median()
                        ),
                    }
                )
    summary = pd.DataFrame(rows).set_index(
        ["strategy", "sample", "block_length"]
    )
    summary.to_csv(DESTINATION / "paired_summary.csv")
    print(summary.round(6).to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()

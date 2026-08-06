from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASELINE = Path(
    "output/paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
)
DEFAULT_RESEARCH = Path("output/research_hmm_gaussian_diagnostics")
DEFAULT_OUTPUT = Path("output/hmm_whitebox_diagnostics")
MEMBER_COLUMNS = [
    "state_entropy",
    "state_probability_margin",
    "one_step_predictive_log_likelihood",
    "template_expected_nearest_distance",
    "template_expected_distance_margin",
    "paper_risk_on_candidate",
    "dominant_template",
]


def run_lengths(values: pd.Series) -> pd.Series:
    clean = values.astype(int)
    groups = clean.ne(clean.shift(1)).cumsum()
    return groups.groupby(groups).cumcount().add(1).astype(int)


def causal_percentile(values: pd.Series, minimum_history: int = 52) -> pd.Series:
    clean = values.astype(float)
    result = pd.Series(np.nan, index=clean.index, dtype=float)
    for position in range(minimum_history, len(clean)):
        history = clean.iloc[:position].dropna()
        current = clean.iloc[position]
        if len(history) >= minimum_history and np.isfinite(current):
            result.iloc[position] = float((history <= current).mean())
    return result


def load_member_regimes(directory: Path) -> dict[str, pd.DataFrame]:
    members: dict[str, pd.DataFrame] = {}
    for path in sorted((directory / "members").glob("seed_*/regimes.csv")):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        missing = sorted(set(MEMBER_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing diagnostics: {missing}")
        selected = frame[MEMBER_COLUMNS].copy()
        selected["state_duration_evaluations"] = run_lengths(
            selected["paper_risk_on_candidate"]
        )
        members[path.parent.name] = selected
    if not members:
        raise ValueError(f"No member regime files found in {directory}")
    return members


def ensemble_diagnostics(members: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common = next(iter(members.values())).index
    for frame in members.values():
        common = common.intersection(frame.index)
    rows: list[dict[str, object]] = []
    for date in common:
        current = pd.DataFrame(
            {name: frame.loc[date] for name, frame in members.items()}
        ).T
        votes = current["paper_risk_on_candidate"].astype(float)
        vote_fraction = float(votes.mean())
        vote_margin = float(abs(2.0 * vote_fraction - 1.0))
        probability_margin_mean = float(
            current["state_probability_margin"].mean()
        )
        template_counts = current["dominant_template"].astype(int).value_counts()
        rows.append(
            {
                "date": date,
                "state_entropy_mean": float(current["state_entropy"].mean()),
                "state_entropy_max": float(current["state_entropy"].max()),
                "state_probability_margin_mean": probability_margin_mean,
                "risk_on_vote_fraction": vote_fraction,
                "risk_on_vote_margin": vote_margin,
                "risk_on_unanimous": int(vote_margin > 1.0 - 1e-12),
                "uncertainty_capacity_confidence": (
                    probability_margin_mean * vote_margin
                ),
                "dominant_template_agreement": float(
                    template_counts.iloc[0] / len(current)
                ),
                "one_step_predictive_log_likelihood_mean": float(
                    current["one_step_predictive_log_likelihood"].mean()
                ),
                "template_expected_nearest_distance_mean": float(
                    current["template_expected_nearest_distance"].mean()
                ),
                "template_expected_distance_margin_mean": float(
                    current["template_expected_distance_margin"].mean()
                ),
                "state_duration_evaluations_mean": float(
                    current["state_duration_evaluations"].mean()
                ),
                "state_duration_evaluations_max": int(
                    current["state_duration_evaluations"].max()
                ),
            }
        )
    result = pd.DataFrame(rows).set_index("date")
    result["template_distance_causal_percentile"] = causal_percentile(
        result["template_expected_nearest_distance_mean"]
    )
    result["negative_log_likelihood_causal_percentile"] = causal_percentile(
        -result["one_step_predictive_log_likelihood_mean"]
    )
    return result


def maximum_path_difference(
    baseline: Path,
    research: Path,
    filename: str,
) -> float:
    left = pd.read_csv(baseline / filename, index_col=0, parse_dates=True)
    right = pd.read_csv(research / filename, index_col=0, parse_dates=True)
    common_index = left.index.intersection(right.index)
    common_columns = left.columns.intersection(right.columns)
    return float(
        left.loc[common_index, common_columns]
        .sub(right.loc[common_index, common_columns])
        .abs()
        .max()
        .max()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-seed HMM white-box diagnostics"
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--research", type=Path, default=DEFAULT_RESEARCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    members = load_member_regimes(args.research)
    ensemble = ensemble_diagnostics(members)
    daily_difference = maximum_path_difference(
        args.baseline, args.research, "daily_returns.csv"
    )
    weight_difference = maximum_path_difference(
        args.baseline, args.research, "weights.csv"
    )
    latest = ensemble.iloc[-1]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        "member_count": len(members),
        "observations": len(ensemble),
        "latest_date": ensemble.index[-1].date().isoformat(),
        "baseline_identity": {
            "maximum_daily_return_path_difference": daily_difference,
            "maximum_weight_path_difference": weight_difference,
            "passed": bool(max(daily_difference, weight_difference) <= 1e-10),
        },
        "diagnostics": {
            "mean_state_entropy": float(ensemble["state_entropy_mean"].mean()),
            "mean_state_probability_margin": float(
                ensemble["state_probability_margin_mean"].mean()
            ),
            "unanimous_vote_fraction": float(
                ensemble["risk_on_unanimous"].mean()
            ),
            "mean_capacity_confidence": float(
                ensemble["uncertainty_capacity_confidence"].mean()
            ),
            "latest_capacity_confidence": float(
                latest["uncertainty_capacity_confidence"]
            ),
            "latest_template_distance_percentile": float(
                latest["template_distance_causal_percentile"]
            ),
            "latest_negative_log_likelihood_percentile": float(
                latest["negative_log_likelihood_causal_percentile"]
            ),
        },
        "decision": (
            "diagnostics_qualified_for_research_use"
            if max(daily_difference, weight_difference) <= 1e-10
            else "baseline_identity_failed"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(members, names=["member", "date"]).to_csv(
        args.output / "member_diagnostics.csv"
    )
    ensemble.to_csv(args.output / "ensemble_diagnostics.csv")
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

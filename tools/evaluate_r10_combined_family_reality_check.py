from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_causal_delayed_grid import variants
from evaluate_smh_causal_family_reality_check import (
    circular_family_reality_check,
)
from evaluate_smh_dynamic_guard import (
    load_inputs as load_normal_inputs,
    simulate,
)
from evaluate_smh_long_proxy_validation import (
    load_inputs as load_proxy_inputs,
)
from evaluate_r10_gde_capital_efficiency import (
    NORMAL_DIRECTORY,
    PROXY_DIRECTORY,
    simulate_gde_substitution,
)
from evaluate_r10_combined_tail_capital import selected_guard


OUTPUT = Path("output/r10_combined_family_reality_check")
COMBINED_OUTPUT = Path("output/r10_combined_tail_capital")
GDE_OUTPUT = Path("output/r10_gde_capital_efficiency")
CASH_OUTPUT = Path("output/r10_conditional_cash_completion")
PROXY_ROOT = Path("output/experiment_r9_broad50_stage35_d10_20y_proxy")
def digest_path(relative_log_return: np.ndarray) -> str:
    return sha256(
        np.round(relative_log_return, 14).tobytes()
    ).hexdigest()


def load_daily_return(path: Path) -> pd.Series:
    return pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )["net_return"]


def relative_log_path(
    candidate: pd.Series,
    baseline: pd.Series,
) -> np.ndarray:
    aligned = pd.concat(
        [
            baseline.rename("baseline"),
            candidate.rename("candidate"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) != len(baseline):
        raise ValueError("Candidate does not cover the full baseline path")
    return (
        np.log1p(aligned["candidate"].to_numpy(dtype=float))
        - np.log1p(aligned["baseline"].to_numpy(dtype=float))
    )


def register_path(
    arrays: dict[str, np.ndarray],
    names: dict[str, list[str]],
    name: str,
    relative_log_return: np.ndarray,
) -> str:
    digest = digest_path(relative_log_return)
    arrays.setdefault(digest, relative_log_return)
    names.setdefault(digest, []).append(name)
    return digest


def sample_inputs(
    sample: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    if sample == "normal":
        weights, daily, opens, closes = load_normal_inputs()
        return weights, daily, opens, closes, "2015-01-01"
    if sample == "proxy":
        weights, daily, opens, closes = load_proxy_inputs(PROXY_ROOT)
        return weights, daily, opens, closes, "2006-08-01"
    raise ValueError(f"Unknown sample: {sample}")


def available_extra_paths(sample_prefix: str) -> dict[str, Path]:
    return {
        "combined_selected_fraction50": (
            COMBINED_OUTPUT
            / f"{sample_prefix}_primary_fraction50_daily.csv"
        ),
        "combined_selected_fraction75": (
            COMBINED_OUTPUT
            / f"{sample_prefix}_primary_fraction75_daily.csv"
        ),
        "gde_fraction50_without_guard": (
            GDE_OUTPUT / f"{sample_prefix}_primary_daily.csv"
        ),
        "gde_calm_broad_bull": (
            GDE_OUTPUT / f"{sample_prefix}_calm_broad_bull_daily.csv"
        ),
        "conditional_cash_completion": (
            CASH_OUTPUT / f"{sample_prefix}_primary_daily.csv"
        ),
        "r11_risk1p025_gde40": (
            Path("output/r11_levered_diversified_strategy")
            / f"{sample_prefix}_risk1.025_gde40_daily.csv"
        ),
        "r11_risk1p065_gde10": (
            Path("output/r11_confirmatory_financing_mix")
            / f"{sample_prefix}_risk1.065_gde10_daily.csv"
        ),
    }


def evaluate(
    sample: str,
    selected_cap: float,
    selected_relative_filter: float | None,
    substitution_fraction: float,
    gde_cost_bps: float,
    gde_no_trade_band: float,
    selected_extra_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights, base_daily, opens, closes, start_date = sample_inputs(sample)
    sample_prefix = (
        "normal_synthetic" if sample == "normal" else "proxy_synthetic"
    )
    guards = variants()
    baseline_daily, _ = simulate(
        weights,
        base_daily,
        opens,
        closes,
        guards[0],
        start_date=start_date,
    )
    baseline = baseline_daily["net_return"]
    gde_baseline = load_daily_return(
        COMBINED_OUTPUT / f"{sample_prefix}_baseline_daily.csv"
    )
    gde_candidate = simulate_gde_substitution(
        NORMAL_DIRECTORY if sample == "normal" else PROXY_DIRECTORY,
        opens,
        closes,
        substitution_fraction=substitution_fraction,
        gate_mode="growth_linked",
        gde_return_mode="synthetic",
        start_date=start_date,
        end_date=None,
        gde_one_way_cost_bps=gde_cost_bps,
        gde_no_trade_band=gde_no_trade_band,
    )
    cap_label = int(round(selected_cap * 100))
    if selected_relative_filter is None:
        selected_guard_name = (
            f"causal_gap250bp_min60_cap{cap_label}_confirm2_hold4"
        )
        exact_selected_path = (
            Path("output/r10_guard_cap_neighborhood")
            / f"{sample_prefix}_cap{cap_label}_daily.csv"
        )
    else:
        selected_configuration = selected_guard(
            20.0,
            post_trigger_cap=selected_cap,
            relative_gap_trigger=selected_relative_filter,
        )
        selected_guard_name = selected_configuration.name
        relative_label = int(
            round(abs(selected_relative_filter) * 10_000)
        )
        band_label = int(round(gde_no_trade_band * 10_000))
        fraction_label = int(round(substitution_fraction * 100))
        frontier_path = (
            Path("output/r10_high_fraction_band_frontier")
            / (
                f"{sample_prefix}_fraction{fraction_label}"
                f"_band{band_label}bp_daily.csv"
            )
        )
        legacy_path = (
            Path("output/r10_gde_no_trade_band")
            / f"{sample_prefix}_band{band_label}bp_daily.csv"
        )
        exact_selected_path = (
            frontier_path if frontier_path.exists() else legacy_path
        )
        guards.append(selected_configuration)
    exact_selected = load_daily_return(exact_selected_path)
    common = baseline.index.intersection(gde_baseline.index)
    if len(common) != len(baseline):
        raise ValueError("GDE baseline does not cover the guard baseline")
    baseline_reconstruction_error = float(
        (baseline.loc[common] - gde_baseline.loc[common]).abs().max()
    )
    gde_increment = (
        gde_candidate.loc[common, "net_return"]
        - gde_baseline.loc[common]
    )

    arrays: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}
    selected_digest: str | None = None
    approximation_error: float | None = None
    for guard in guards:
        guard_daily, _ = simulate(
            weights,
            base_daily,
            opens,
            closes,
            guard,
            start_date=start_date,
        )
        approximate = (
            guard_daily.loc[common, "net_return"] + gde_increment
        )
        if guard.name == selected_guard_name:
            approximation_error = float(
                (approximate - exact_selected.loc[common]).abs().max()
            )
            relative = relative_log_path(
                exact_selected.loc[common],
                baseline.loc[common],
            )
            path_name = (
                f"{guard.name}+gde{substitution_fraction:.0%}"
                f"_band{gde_no_trade_band:.2%}_exact"
            )
            selected_digest = register_path(
                arrays,
                names,
                path_name,
                relative,
            )
        else:
            relative = relative_log_path(
                approximate,
                baseline.loc[common],
            )
            register_path(
                arrays,
                names,
                (
                    f"{guard.name}+gde{substitution_fraction:.0%}"
                    f"_band{gde_no_trade_band:.2%}_approximate"
                ),
                relative,
            )

    for name, path in available_extra_paths(sample_prefix).items():
        if path.exists():
            extra_digest = register_path(
                arrays,
                names,
                name,
                relative_log_path(
                    load_daily_return(path).loc[common],
                    baseline.loc[common],
                ),
            )
            if name == selected_extra_name:
                selected_digest = extra_digest

    if selected_digest is None or approximation_error is None:
        raise RuntimeError(
            f"Selected guard was not found: {selected_guard_name}"
        )
    digests = list(arrays)
    matrix = np.column_stack([arrays[digest] for digest in digests])
    selected_index = digests.index(selected_digest)
    observed = matrix.mean(axis=0) * 252.0
    ranking = (
        pd.DataFrame(
            {
                "stream_id": np.arange(len(digests)),
                "representative_candidate": [
                    names[digest][0] for digest in digests
                ],
                "candidate_name_count": [
                    len(names[digest]) for digest in digests
                ],
                "annualized_relative_log_return": observed,
                "includes_selected": [
                    int(digest == selected_digest) for digest in digests
                ],
            }
        )
        .sort_values(
            "annualized_relative_log_return",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    rows: list[dict[str, float | int | str]] = []
    selected_rank = int(
        ranking.index[ranking["includes_selected"].eq(1)][0] + 1
    )
    for block_days in (21, 63, 126):
        rows.append(
            {
                "sample": sample,
                "selected_post_trigger_cap": selected_cap,
                "selected_relative_gap_filter": (
                    selected_relative_filter
                ),
                "substitution_fraction": substitution_fraction,
                "gde_cost_bps": gde_cost_bps,
                "gde_no_trade_band": gde_no_trade_band,
                "selected_extra_name": selected_extra_name or "",
                **circular_family_reality_check(
                    matrix,
                    selected_index,
                    block_days,
                ),
                "guard_parameterizations": len(guards),
                "registered_candidate_names": int(
                    sum(len(value) for value in names.values())
                ),
                "selected_rank": selected_rank,
                "baseline_reconstruction_error": (
                    baseline_reconstruction_error
                ),
                "selected_approximation_max_daily_error": (
                    approximation_error
                ),
            }
        )
    return pd.DataFrame(rows), ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        choices=("normal", "proxy"),
        required=True,
    )
    parser.add_argument(
        "--selected-cap",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--selected-relative-filter",
        type=float,
        default=-0.005,
    )
    parser.add_argument(
        "--substitution-fraction",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--gde-cost-bps",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--gde-no-trade-band",
        type=float,
        default=0.0025,
    )
    parser.add_argument(
        "--selected-extra-name",
        choices=tuple(available_extra_paths("sample").keys()),
    )
    arguments = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result, ranking = evaluate(
        arguments.sample,
        arguments.selected_cap,
        arguments.selected_relative_filter,
        arguments.substitution_fraction,
        arguments.gde_cost_bps,
        arguments.gde_no_trade_band,
        arguments.selected_extra_name,
    )
    cap_label = int(round(arguments.selected_cap * 100))
    relative_label = int(
        round(abs(arguments.selected_relative_filter) * 10_000)
    )
    fraction_label = int(round(arguments.substitution_fraction * 100))
    band_label = int(round(arguments.gde_no_trade_band * 10_000))
    output_stem = (
        f"{arguments.sample}_cap{cap_label}_rel{relative_label}bp"
        f"_gde{fraction_label}_band{band_label}bp"
    )
    if arguments.selected_extra_name:
        output_stem += f"_selected_{arguments.selected_extra_name}"
    result.to_csv(
        OUTPUT / f"{output_stem}_family_reality_check.csv",
        index=False,
    )
    ranking.to_csv(
        OUTPUT / f"{output_stem}_candidate_ranking.csv",
        index=False,
    )
    print(result.round(6).to_string(index=False))
    print("\nTop candidates:")
    print(ranking.head(15).round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

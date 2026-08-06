from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_r10_gde_capital_efficiency import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    load_adjusted_open_close,
)
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    load_strategy_inputs,
    metric_delta,
    relative_log_return,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
)
from tools.evaluate_r14_incremental_trend_permission import (
    TrendCandidate,
    simulate_candidate,
)
from tools.evaluate_r16_portable_managed_futures import (
    NORMAL_R11,
    NORMAL_R14,
    PRICE_CACHE,
    PROXY_R11,
    PROXY_R14,
    adjusted_returns,
    data_audit,
    leave_one_out_rows,
    load_prices,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "r17_causal_managed_futures_permission"
ACADEMIC_CSV = ROOT / "data" / "aqr_tsmom_factors_monthly.csv"
ACADEMIC_XLSX = ROOT / "data" / "aqr_tsmom_factors_monthly.xlsx"
ACADEMIC_METADATA = (
    ROOT / "data" / "aqr_tsmom_factors_monthly.metadata.json"
)
TRADING_DAYS = 252
OVERLAY_WEIGHT = 0.15
CUMULATIVE_TRIALS = 64
SWITCH_COST_BPS = 10.0


def causal_monthly_permission(prices: pd.DataFrame) -> pd.Series:
    common = prices[["AQMIX", "BIL"]].dropna()
    if common.empty:
        raise ValueError("AQMIX/BIL prices have no common observations")
    relative_price = common["AQMIX"] / common["BIL"]
    trailing = relative_price / relative_price.shift(TRADING_DAYS) - 1.0
    month_end_signal = trailing.groupby(trailing.index.to_period("M")).last()
    permission_by_month = month_end_signal.gt(0.0).shift(1)
    daily_period = common.index.to_period("M")
    permission = pd.Series(
        daily_period.map(permission_by_month).fillna(False).to_numpy(
            dtype=bool
        ),
        index=common.index,
        name="permission",
    )
    return permission


def gated_stack(
    base_returns: pd.Series,
    fund_returns: pd.Series,
    cash_returns: pd.Series,
    permission: pd.Series,
    overlay_weight: float,
    *,
    financing_spread_bps: float,
    one_way_cost_bps: float = SWITCH_COST_BPS,
) -> pd.DataFrame:
    if overlay_weight < 0.0:
        raise ValueError("overlay_weight must be non-negative")
    aligned = pd.concat(
        [
            base_returns.rename("base_return"),
            fund_returns.rename("fund_return"),
            cash_returns.rename("cash_return"),
            permission.rename("permission"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    aligned["permission"] = aligned["permission"].astype(bool)
    aligned["implemented_overlay_weight"] = (
        overlay_weight * aligned["permission"].astype(float)
    )
    spread = financing_spread_bps / 10_000.0 / TRADING_DAYS
    aligned["overlay_excess_return"] = (
        aligned["fund_return"] - aligned["cash_return"] - spread
    )
    prior_weight = (
        aligned["implemented_overlay_weight"].shift(1).fillna(0.0)
    )
    aligned["switching_cost"] = (
        (
            aligned["implemented_overlay_weight"] - prior_weight
        ).abs()
        * one_way_cost_bps
        / 10_000.0
    )
    aligned["candidate_return"] = (
        aligned["base_return"]
        + aligned["implemented_overlay_weight"]
        * aligned["overlay_excess_return"]
        - aligned["switching_cost"]
    )
    if aligned["candidate_return"].le(-1.0).any():
        raise ValueError("Gated portable-alpha path lost all capital")
    return aligned


def academic_permission(factor_returns: pd.Series) -> pd.Series:
    trailing = (
        (1.0 + factor_returns)
        .rolling(12, min_periods=12)
        .apply(np.prod, raw=True)
        - 1.0
    )
    return trailing.shift(1).gt(0.0).rename("permission")


def academic_gate_metrics(
    factors: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    periods = {
        "old_1985_2009": ("1985-01-01", "2009-12-31"),
        "later_2010_2026": ("2010-01-01", "2026-12-31"),
    }
    for column in factors.columns:
        permission = academic_permission(factors[column])
        gated = factors[column].where(permission, 0.0)
        valid = (
            factors[column]
            .rolling(12, min_periods=12)
            .count()
            .shift(1)
            .eq(12)
        )
        for period, (start, end) in periods.items():
            selected = factors.index[
                (factors.index >= start)
                & (factors.index <= end)
                & valid
            ]
            always = factors.loc[selected, column]
            trial = gated.loc[selected]
            always_metrics = performance_metrics(
                always,
                annualization=12,
            )
            trial_metrics = performance_metrics(
                trial,
                annualization=12,
            )
            rows.append(
                {
                    "factor": column,
                    "period": period,
                    "observations": len(selected),
                    "permission_share": float(
                        permission.loc[selected].mean()
                    ),
                    "always_annualized_arithmetic_return": float(
                        always.mean() * 12.0
                    ),
                    "gated_annualized_arithmetic_return": float(
                        trial.mean() * 12.0
                    ),
                    "annualized_arithmetic_delta": float(
                        (trial.mean() - always.mean()) * 12.0
                    ),
                    "always_cagr": float(always_metrics["cagr"]),
                    "gated_cagr": float(trial_metrics["cagr"]),
                    "always_max_drawdown": float(
                        always_metrics["max_drawdown"]
                    ),
                    "gated_max_drawdown": float(
                        trial_metrics["max_drawdown"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def academic_data_audit() -> pd.DataFrame:
    metadata = json.loads(
        ACADEMIC_METADATA.read_text(encoding="utf-8")
    )
    csv_digest = hashlib.sha256(ACADEMIC_CSV.read_bytes()).hexdigest()
    xlsx_digest = hashlib.sha256(ACADEMIC_XLSX.read_bytes()).hexdigest()
    factors = pd.read_csv(
        ACADEMIC_CSV,
        index_col="date",
        parse_dates=True,
    )
    expected_columns = {
        "TSMOM",
        "TSMOM_CM",
        "TSMOM_EQ",
        "TSMOM_FI",
        "TSMOM_FX",
    }
    return pd.DataFrame(
        [
            {
                "check": "csv_sha256",
                "observed": csv_digest,
                "expected": metadata["csv_sha256"],
                "pass": csv_digest == metadata["csv_sha256"],
            },
            {
                "check": "xlsx_sha256",
                "observed": xlsx_digest,
                "expected": metadata["xlsx_sha256"],
                "pass": xlsx_digest == metadata["xlsx_sha256"],
            },
            {
                "check": "valid_row_count",
                "observed": len(factors),
                "expected": 497,
                "pass": len(factors) == 497,
            },
            {
                "check": "first_date",
                "observed": factors.index.min().date().isoformat(),
                "expected": "1985-01-31",
                "pass": factors.index.min()
                == pd.Timestamp("1985-01-31"),
            },
            {
                "check": "last_date",
                "observed": factors.index.max().date().isoformat(),
                "expected": "2026-05-29",
                "pass": factors.index.max()
                == pd.Timestamp("2026-05-29"),
            },
            {
                "check": "factor_columns",
                "observed": ",".join(factors.columns),
                "expected": ",".join(sorted(expected_columns)),
                "pass": set(factors.columns) == expected_columns,
            },
            {
                "check": "finite_returns",
                "observed": bool(
                    np.isfinite(factors.to_numpy(dtype=float)).all()
                ),
                "expected": True,
                "pass": bool(
                    np.isfinite(factors.to_numpy(dtype=float)).all()
                ),
            },
        ]
    )


def _load_return(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col="date", parse_dates=True)
    return frame["net_return"].astype(float)


def active118_paths() -> dict[str, pd.Series]:
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(
        PROXY_OPEN_CLOSE
    )
    normal_weights, normal_daily = load_strategy_inputs(
        NORMAL_DIRECTORY
    )
    proxy_weights, proxy_daily = load_strategy_inputs(
        PROXY_DIRECTORY
    )
    samples = {
        "normal_actual": {
            "directory": NORMAL_DIRECTORY,
            "weights": normal_weights,
            "daily": normal_daily,
            "opens": normal_opens,
            "closes": normal_closes,
            "mode": "synthetic",
            "start": "2015-01-01",
        },
        "expanded_actual": {
            "directory": PROXY_DIRECTORY,
            "weights": proxy_weights,
            "daily": proxy_daily,
            "opens": proxy_opens,
            "closes": proxy_closes,
            "mode": "synthetic",
            "start": "2006-08-01",
        },
    }
    result: dict[str, pd.Series] = {}
    definition = TrendCandidate("both_sma200", "both_sma200")
    for sample, settings in samples.items():
        trial, _ = simulate_candidate(
            settings,
            COST_SCENARIOS[0],
            definition,
            active_multiplier=1.18,
            cash_floor=-0.25,
        )
        result[sample] = trial["net_return"]
    return result


def _period_row(
    sample: str,
    scenario: str,
    candidate: str,
    period: str,
    comparator: str,
    comparator_returns: pd.Series,
    candidate_returns: pd.Series,
    start: str,
    end: str,
) -> dict[str, object]:
    aligned = pd.concat(
        [
            comparator_returns.rename("baseline"),
            candidate_returns.rename("candidate"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    selected = aligned.loc[start:end]
    relative = (
        np.log1p(selected["candidate"])
        - np.log1p(selected["baseline"])
    )
    return {
        "sample": sample,
        "scenario": scenario,
        "candidate": candidate,
        "period": period,
        "comparator": comparator,
        "observations": len(selected),
        "annualized_relative_log_return": float(
            relative.mean() * TRADING_DAYS
        ),
        **metric_delta(selected["baseline"], selected["candidate"]),
    }


def _academic_mechanism_pass(
    academic_metrics: pd.DataFrame,
) -> bool:
    for period in ("old_1985_2009", "later_2010_2026"):
        composite = academic_metrics.loc[
            (academic_metrics["period"] == period)
            & (academic_metrics["factor"] == "TSMOM")
        ].iloc[0]
        asset_classes = academic_metrics.loc[
            (academic_metrics["period"] == period)
            & (academic_metrics["factor"] != "TSMOM")
        ]
        if not (
            composite["gated_annualized_arithmetic_return"] > 0.0
            and composite["annualized_arithmetic_delta"] >= 0.0
            and composite["gated_max_drawdown"]
            >= composite["always_max_drawdown"]
            and asset_classes[
                "gated_annualized_arithmetic_return"
            ].gt(0.0).sum()
            >= 3
        ):
            return False
    return True


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    price_audit = data_audit(prices)
    academic_audit = academic_data_audit()
    audit = pd.concat(
        [
            price_audit.assign(source="AQMIX_BIL"),
            academic_audit.assign(source="AQR_TSMOM"),
        ],
        ignore_index=True,
    )
    audit.to_csv(OUTPUT / "data_audit.csv", index=False)

    academic = pd.read_csv(
        ACADEMIC_CSV,
        index_col="date",
        parse_dates=True,
    )
    academic_metrics = academic_gate_metrics(academic)
    academic_metrics.to_csv(
        OUTPUT / "academic_gate_metrics.csv",
        index=False,
    )
    academic_pass = _academic_mechanism_pass(academic_metrics)

    actual_returns = adjusted_returns(prices)
    fund = actual_returns["fund_return"]
    cash = actual_returns["cash_return"]
    permission = causal_monthly_permission(prices)
    permission.to_csv(
        OUTPUT / "aqmix_monthly_permission.csv",
        index_label="date",
    )

    active118 = active118_paths()
    bases = {
        "active115_gate12m_overlay15": {
            "normal_actual": _load_return(NORMAL_R14),
            "expanded_actual": _load_return(PROXY_R14),
        },
        "active118_gate12m_overlay15": {
            "normal_actual": active118["normal_actual"],
            "expanded_actual": active118["expanded_actual"],
        },
    }
    r11 = {
        "normal_actual": _load_return(NORMAL_R11),
        "expanded_actual": _load_return(PROXY_R11),
    }
    periods = {
        "normal_actual": {
            "development_2015_2021": (
                "2015-01-02",
                "2021-12-31",
            ),
            "holdout_2022_2025": (
                "2022-01-03",
                "2025-12-31",
            ),
            "recent_2026": ("2026-01-02", "2026-12-31"),
            "complete_2015_2026": (
                "2015-01-02",
                "2026-12-31",
            ),
        },
        "expanded_actual": {
            "early_2010_2017": (
                "2010-01-05",
                "2017-12-31",
            ),
            "late_2018_2026": (
                "2018-01-01",
                "2026-12-31",
            ),
            "complete_2010_2026": (
                "2010-01-05",
                "2026-12-31",
            ),
        },
    }
    metric_rows: list[dict[str, object]] = []
    paths: dict[str, dict[str, pd.Series]] = {
        "normal_actual": {},
        "expanded_actual": {},
    }
    for name, sample_bases in bases.items():
        for sample, base in sample_bases.items():
            for scenario, spread in (
                ("current_100bps", 100.0),
                ("stress_150bps", 150.0),
            ):
                path = gated_stack(
                    base,
                    fund,
                    cash,
                    permission,
                    OVERLAY_WEIGHT,
                    financing_spread_bps=spread,
                )
                candidate_returns = path["candidate_return"]
                if scenario == "current_100bps":
                    paths[sample][name] = candidate_returns
                    path.to_csv(
                        OUTPUT / f"{sample}_{name}_daily.csv",
                        index_label="date",
                    )
                for period, (start, end) in periods[sample].items():
                    for comparator, comparator_returns in (
                        ("r11", r11[sample]),
                        ("r14_base", base),
                    ):
                        metric_rows.append(
                            _period_row(
                                sample,
                                scenario,
                                name,
                                period,
                                comparator,
                                comparator_returns,
                                candidate_returns,
                                start,
                                end,
                            )
                        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    family_rows: list[dict[str, object]] = []
    names = list(bases)
    for sample in ("normal_actual", "expanded_actual"):
        matrix = np.column_stack(
            [
                relative_log_return(
                    bases[name][sample],
                    paths[sample][name],
                )
                for name in names
            ]
        )
        for selected_index, name in enumerate(names):
            for block_days in (21, 63, 126):
                check = circular_family_reality_check(
                    matrix,
                    selected_index,
                    block_days,
                )
                family_p = float(
                    check["familywise_reality_check_p_value"]
                )
                family_rows.append(
                    {
                        "sample": sample,
                        "candidate": name,
                        "declared_family_size": len(names),
                        "cumulative_trials": CUMULATIVE_TRIALS,
                        **check,
                        "cumulative_trial_adjusted_p_value": min(
                            1.0,
                            family_p
                            * CUMULATIVE_TRIALS
                            / len(names),
                        ),
                    }
                )
    family = pd.DataFrame(family_rows)
    family.to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    year_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for name in names:
        years, events = leave_one_out_rows(
            name,
            r11["normal_actual"],
            bases[name]["normal_actual"],
            paths["normal_actual"][name],
        )
        year_rows.extend(years)
        event_rows.extend(events)
    years = pd.DataFrame(year_rows)
    events = pd.DataFrame(event_rows)
    years.to_csv(OUTPUT / "leave_one_year_out.csv", index=False)
    events.to_csv(
        OUTPUT / "leave_one_r11_drawdown_event_out.csv",
        index=False,
    )

    neighborhood_rows: list[dict[str, object]] = []
    for name in names:
        for weight in (0.125, 0.175):
            path = gated_stack(
                bases[name]["normal_actual"],
                fund,
                cash,
                permission,
                weight,
                financing_spread_bps=100.0,
            )
            aligned = pd.concat(
                [
                    r11["normal_actual"].rename("r11"),
                    path["candidate_return"].rename("candidate"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            trial_metrics = performance_metrics(aligned["candidate"])
            r11_metrics = performance_metrics(aligned["r11"])
            neighborhood_rows.append(
                {
                    "candidate": name,
                    "neighbor_weight": weight,
                    "candidate_cagr": trial_metrics["cagr"],
                    "candidate_max_drawdown": (
                        trial_metrics["max_drawdown"]
                    ),
                    "r11_max_drawdown": r11_metrics["max_drawdown"],
                    "ridge_pass": bool(
                        trial_metrics["cagr"] >= 0.25
                        and trial_metrics["max_drawdown"]
                        >= r11_metrics["max_drawdown"]
                    ),
                }
            )
    neighborhoods = pd.DataFrame(neighborhood_rows)
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    acceptance_rows: list[dict[str, object]] = []
    for name in names:
        central = metrics.loc[
            (metrics["candidate"] == name)
            & (metrics["scenario"] == "current_100bps")
        ]
        stress = metrics.loc[
            (metrics["candidate"] == name)
            & (metrics["scenario"] == "stress_150bps")
        ]

        def select(
            frame: pd.DataFrame,
            sample: str,
            period: str,
            comparator: str,
        ) -> pd.Series:
            selected = frame.loc[
                (frame["sample"] == sample)
                & (frame["period"] == period)
                & (frame["comparator"] == comparator)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one {name}/{sample}/{period}/"
                    f"{comparator} row"
                )
            return selected.iloc[0]

        complete = select(
            central,
            "normal_actual",
            "complete_2015_2026",
            "r11",
        )
        development_r11 = select(
            central,
            "normal_actual",
            "development_2015_2021",
            "r11",
        )
        development_base = select(
            central,
            "normal_actual",
            "development_2015_2021",
            "r14_base",
        )
        holdout_r11 = select(
            central,
            "normal_actual",
            "holdout_2022_2025",
            "r11",
        )
        holdout_base = select(
            central,
            "normal_actual",
            "holdout_2022_2025",
            "r14_base",
        )
        stress_complete = select(
            stress,
            "normal_actual",
            "complete_2015_2026",
            "r11",
        )
        expanded_complete = select(
            central,
            "expanded_actual",
            "complete_2010_2026",
            "r11",
        )
        expanded_early = select(
            central,
            "expanded_actual",
            "early_2010_2017",
            "r14_base",
        )
        expanded_late = select(
            central,
            "expanded_actual",
            "late_2018_2026",
            "r14_base",
        )
        family_pass = bool(
            family.loc[
                (family["sample"] == "normal_actual")
                & (family["candidate"] == name),
                "cumulative_trial_adjusted_p_value",
            ].le(0.05).all()
        )
        leave_one_out_pass = bool(
            years.loc[
                years["candidate"] == name,
                "annualized_relative_log_return",
            ].gt(0.0).all()
            and events.loc[
                events["candidate"] == name,
                "annualized_relative_log_return",
            ].gt(0.0).all()
        )
        ridge_pass = bool(
            neighborhoods.loc[
                neighborhoods["candidate"] == name,
                "ridge_pass",
            ].any()
        )
        data_pass = bool(audit["pass"].all())
        economic_statistical_pass = bool(
            complete["candidate_cagr"] >= 0.25
            and complete["candidate_max_drawdown"]
            >= complete["baseline_max_drawdown"]
            and development_r11["annualized_relative_log_return"] > 0.0
            and development_base["annualized_relative_log_return"] > 0.0
            and holdout_r11["annualized_relative_log_return"] > 0.0
            and holdout_base["annualized_relative_log_return"] > 0.0
            and stress_complete["candidate_cagr"] >= 0.25
            and stress_complete["candidate_max_drawdown"]
            >= stress_complete["baseline_max_drawdown"]
            and expanded_complete["candidate_max_drawdown"]
            >= expanded_complete["baseline_max_drawdown"]
            and expanded_early["annualized_relative_log_return"] > 0.0
            and expanded_late["annualized_relative_log_return"] > 0.0
            and family_pass
            and leave_one_out_pass
            and ridge_pass
            and academic_pass
        )
        product_access_pass = False
        acceptance_rows.append(
            {
                "candidate": name,
                "complete_cagr": complete["candidate_cagr"],
                "complete_max_drawdown": (
                    complete["candidate_max_drawdown"]
                ),
                "r11_max_drawdown": (
                    complete["baseline_max_drawdown"]
                ),
                "development_relative_to_base": (
                    development_base[
                        "annualized_relative_log_return"
                    ]
                ),
                "holdout_relative_to_base": (
                    holdout_base["annualized_relative_log_return"]
                ),
                "stress_cagr": stress_complete["candidate_cagr"],
                "stress_max_drawdown": (
                    stress_complete["candidate_max_drawdown"]
                ),
                "family_multiple_testing_pass": family_pass,
                "leave_one_out_pass": leave_one_out_pass,
                "parameter_ridge_pass": ridge_pass,
                "academic_mechanism_pass": academic_pass,
                "data_audit_pass": data_pass,
                "product_access_and_financing_confirmed": (
                    product_access_pass
                ),
                "statistical_economic_pass": (
                    economic_statistical_pass
                ),
                "production_pass": bool(
                    economic_statistical_pass
                    and data_pass
                    and product_access_pass
                ),
            }
        )
    acceptance = pd.DataFrame(acceptance_rows)
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    print(
        acceptance[
            [
                "candidate",
                "complete_cagr",
                "complete_max_drawdown",
                "r11_max_drawdown",
                "academic_mechanism_pass",
                "family_multiple_testing_pass",
                "statistical_economic_pass",
                "production_pass",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

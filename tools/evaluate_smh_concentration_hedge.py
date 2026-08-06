from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "smh_concentration_hedge"
ASSETS = [
    "SPX",
    "QQQ",
    "SEMIS",
    "BOND",
    "GOLD",
    "OIL",
    "USD",
    "CASH",
    "VIX_HEDGE",
]
SAMPLES = {
    "normal": {
        "strategy": (
            "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
        ),
        "prices": "data/prices_vix_hedge.csv",
        "periods": {
            "development_2015_2021": ("2015-01-01", "2021-12-31"),
            "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
            "complete_2015_present": ("2015-01-01", None),
            "recent_2024_present": ("2024-01-01", None),
        },
    },
    "proxy": {
        "strategy": (
            "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
            "_20y_proxy"
        ),
        "prices": "data/prices_20y_proxy.csv",
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_present": ("2015-01-01", None),
            "complete_proxy": ("2006-08-01", None),
        },
    },
}


@dataclass(frozen=True)
class HedgeRule:
    name: str
    entry_semis_weight: float | None = None
    exit_semis_weight: float | None = None
    minimum_vix_hedge_weight: float = 0.0
    funding_asset: str = "SEMIS"
    maximum_entry_term_ratio: float | None = None
    maximum_exit_term_ratio: float | None = None


def variants() -> list[HedgeRule]:
    candidates = [HedgeRule("baseline")]
    for entry_weight in (0.40, 0.50, 0.60):
        entry_label = int(round(entry_weight * 100))
        for hedge_weight in (0.01, 0.02, 0.03, 0.04):
            hedge_label = int(round(hedge_weight * 100))
            for funding_asset in ("SEMIS", "CASH"):
                candidates.append(
                    HedgeRule(
                        name=(
                            f"entry{entry_label}_hedge{hedge_label}"
                            f"_from{funding_asset}"
                        ),
                        entry_semis_weight=entry_weight,
                        exit_semis_weight=entry_weight - 0.10,
                        minimum_vix_hedge_weight=hedge_weight,
                        funding_asset=funding_asset,
                    )
                )
    for entry_weight in (0.50, 0.60):
        entry_label = int(round(entry_weight * 100))
        for maximum_term_ratio in (0.85, 0.90, 0.95):
            ratio_label = int(round(maximum_term_ratio * 100))
            for hedge_weight in (0.01, 0.02, 0.03):
                hedge_label = int(round(hedge_weight * 100))
                candidates.append(
                    HedgeRule(
                        name=(
                            f"entry{entry_label}_term{ratio_label}"
                            f"_hedge{hedge_label}_fromSEMIS"
                        ),
                        entry_semis_weight=entry_weight,
                        exit_semis_weight=entry_weight - 0.10,
                        minimum_vix_hedge_weight=hedge_weight,
                        funding_asset="SEMIS",
                        maximum_entry_term_ratio=maximum_term_ratio,
                        maximum_exit_term_ratio=maximum_term_ratio + 0.05,
                    )
                )
    return candidates


def apply_minimum_hedge(
    weights: np.ndarray,
    hedge_weight: float,
    funding_asset: str,
) -> np.ndarray:
    adjusted = weights.copy()
    hedge_index = ASSETS.index("VIX_HEDGE")
    funding_index = ASSETS.index(funding_asset)
    required = max(hedge_weight - float(adjusted[hedge_index]), 0.0)
    available = max(float(adjusted[funding_index]), 0.0)
    transfer = min(required, available)
    adjusted[hedge_index] += transfer
    adjusted[funding_index] -= transfer
    return adjusted


def remove_overlay_hedge(
    weights: np.ndarray,
    baseline_hedge_weight: float,
    funding_asset: str,
) -> np.ndarray:
    adjusted = weights.copy()
    hedge_index = ASSETS.index("VIX_HEDGE")
    funding_index = ASSETS.index(funding_asset)
    overlay = max(
        float(adjusted[hedge_index]) - baseline_hedge_weight,
        0.0,
    )
    adjusted[hedge_index] -= overlay
    adjusted[funding_index] += overlay
    return adjusted


def load_sample(
    sample: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    settings = SAMPLES[sample]
    root = OUTPUT / str(settings["strategy"])
    weights = pd.read_csv(
        root / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        root / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    prices = pd.read_csv(
        str(settings["prices"]),
        index_col=0,
        parse_dates=True,
    )
    returns = prices.loc[:, ASSETS].pct_change(fill_method=None)
    term_ratio = (prices["VIX"] / prices["VIX3M"]).shift(1)
    return weights, daily, returns, term_ratio


def simulate(
    base_weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    asset_returns: pd.DataFrame,
    term_ratio: pd.Series,
    rule: HedgeRule,
    cost_bps: float = 7.5,
    no_trade_turnover: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = base_weights.index
    base_weight_values = base_weights.loc[dates, ASSETS].to_numpy(dtype=float)
    base_trade_values = (
        base_daily.loc[dates, "turnover"].to_numpy(dtype=float) > 0.0
    )
    financing_cost_values = base_daily.loc[
        dates,
        "financing_cost",
    ].to_numpy(dtype=float)
    return_values = (
        asset_returns.reindex(dates)
        .loc[:, ASSETS]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    term_ratio_values = term_ratio.reindex(dates).to_numpy(dtype=float)

    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    active = False
    rows: list[dict[str, float | int | bool]] = []
    weight_rows: list[np.ndarray] = []
    for position, date in enumerate(dates):
        baseline_target = base_weight_values[position]
        reference_semis_weight = float(
            baseline_target[ASSETS.index("SEMIS")]
        )
        current_term_ratio = float(term_ratio_values[position])
        activated = False
        deactivated = False
        if (
            not active
            and rule.entry_semis_weight is not None
            and reference_semis_weight >= rule.entry_semis_weight
            and (
                rule.maximum_entry_term_ratio is None
                or current_term_ratio <= rule.maximum_entry_term_ratio
            )
        ):
            active = True
            activated = True
        elif (
            active
            and rule.exit_semis_weight is not None
            and (
                reference_semis_weight <= rule.exit_semis_weight
                or (
                    rule.maximum_exit_term_ratio is not None
                    and current_term_ratio > rule.maximum_exit_term_ratio
                )
            )
        ):
            active = False
            deactivated = True

        base_trade = bool(base_trade_values[position])
        desired: np.ndarray | None = None
        if base_trade:
            desired = baseline_target.copy()
        elif activated or deactivated:
            desired = current.copy()

        if active and desired is not None:
            desired = apply_minimum_hedge(
                desired,
                rule.minimum_vix_hedge_weight,
                rule.funding_asset,
            )
        elif deactivated and desired is not None:
            desired = remove_overlay_hedge(
                desired,
                float(baseline_target[ASSETS.index("VIX_HEDGE")]),
                rule.funding_asset,
            )

        proposed_turnover = (
            0.0
            if desired is None
            else 0.5 * float(np.abs(desired - current).sum())
        )
        executed = proposed_turnover >= no_trade_turnover
        turnover = proposed_turnover if executed else 0.0
        trading_cost = (
            2.0 * turnover * cost_bps / 10_000.0
            if executed
            else 0.0
        )
        if executed and desired is not None:
            current = desired

        day_returns = return_values[position]
        gross_return = float(current @ day_returns)
        financing_cost = float(financing_cost_values[position])
        net_return = gross_return - financing_cost - trading_cost
        rows.append(
            {
                "net_return": net_return,
                "gross_return": gross_return,
                "trading_cost": trading_cost,
                "financing_cost": financing_cost,
                "turnover": turnover,
                "active": active,
                "activated": activated,
                "deactivated": deactivated,
                "base_trade": base_trade,
                "vix_term_ratio": current_term_ratio,
            }
        )
        weight_rows.append(current.copy())

        denominator = 1.0 + gross_return
        if denominator <= 0.0:
            raise ValueError("Hedged portfolio lost all capital in one day")
        current = current * (1.0 + day_returns) / denominator

    return (
        pd.DataFrame(rows, index=dates),
        pd.DataFrame(weight_rows, index=dates, columns=ASSETS),
    )


def metric_rows(
    sample: str,
    rule: HedgeRule,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    settings = SAMPLES[sample]
    for period, (start, end) in settings["periods"].items():
        selected = daily.loc[start:end]
        selected_weights = weights.loc[selected.index]
        values = performance_metrics(selected["net_return"])
        rows.append(
            {
                "sample": sample,
                "variant": rule.name,
                "period": period,
                **values,
                "worst_day": float(selected["net_return"].min()),
                "maximum_semis_weight": float(
                    selected_weights["SEMIS"].max()
                ),
                "average_vix_hedge_weight": float(
                    selected_weights["VIX_HEDGE"].mean()
                ),
                "maximum_vix_hedge_weight": float(
                    selected_weights["VIX_HEDGE"].max()
                ),
                "active_days": int(selected["active"].sum()),
                "activations": int(selected["activated"].sum()),
                "annualized_turnover": float(
                    selected["turnover"].mean() * 252.0
                ),
                "annualized_trading_cost": float(
                    selected["trading_cost"].mean() * 252.0
                ),
            }
        )
    return rows


def add_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index(["sample", "variant", "period"])
    for column in ("cagr", "max_drawdown", "worst_day"):
        baseline = indexed.xs("baseline", level="variant")[column]
        indexed[f"{column}_delta_vs_baseline"] = [
            float(row[column] - baseline.loc[(sample, period)])
            for (sample, _, period), row in indexed.iterrows()
        ]
    return indexed.reset_index()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, float | str]] = []
    for sample in SAMPLES:
        base_weights, base_daily, asset_returns, term_ratio = load_sample(sample)
        for rule in variants():
            daily, weights = simulate(
                base_weights,
                base_daily,
                asset_returns,
                term_ratio,
                rule,
            )
            rows.extend(metric_rows(sample, rule, daily, weights))
            if rule.name == "baseline":
                reconstruction_rows.append(
                    {
                        "sample": sample,
                        "maximum_return_difference": float(
                            (
                                daily["net_return"]
                                - base_daily["net_return"]
                            )
                            .abs()
                            .max()
                        ),
                        "maximum_weight_difference": float(
                            (
                                weights - base_weights.loc[:, ASSETS]
                            )
                            .abs()
                            .to_numpy()
                            .max()
                        ),
                    }
                )

    metrics = add_deltas(pd.DataFrame(rows))
    reconstruction = pd.DataFrame(reconstruction_rows)
    metrics.to_csv(DESTINATION / "metrics.csv", index=False)
    reconstruction.to_csv(
        DESTINATION / "reconstruction_check.csv",
        index=False,
    )
    complete = metrics.loc[
        metrics["period"].isin(
            ["complete_2015_present", "complete_proxy"]
        )
    ]
    normal = complete.loc[
        complete["period"] == "complete_2015_present"
    ].set_index("variant")
    proxy = complete.loc[
        complete["period"] == "complete_proxy"
    ].set_index("variant")
    summary = normal[
        [
            "cagr_delta_vs_baseline",
            "max_drawdown_delta_vs_baseline",
            "worst_day_delta_vs_baseline",
            "average_vix_hedge_weight",
            "active_days",
            "activations",
        ]
    ].join(
        proxy[
            [
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
            ]
        ].rename(
            columns={
                "cagr_delta_vs_baseline": "proxy_cagr_delta",
                "max_drawdown_delta_vs_baseline": "proxy_mdd_delta",
            }
        )
    )
    print("Reconstruction:")
    print(reconstruction.to_string(index=False))
    print("\nComplete-period summary:")
    print(
        summary.sort_values(
            ["cagr_delta_vs_baseline", "max_drawdown_delta_vs_baseline"],
            ascending=False,
        )
        .round(6)
        .to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()

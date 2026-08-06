from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.evaluate_r10_gde_capital_efficiency import (
    load_adjusted_open_close,
    paired_block_bootstrap,
)


OUTPUT = Path("output/r10_trade_to_boundary")
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
GROWTH_ASSETS = ["SPX", "QQQ", "SEMIS"]
SAMPLES = {
    "normal": {
        "directory": (
            "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
        ),
        "open_close": Path("data/adjusted_open_close_2011_present.csv"),
        "start": "2015-01-01",
        "periods": {
            "development_2015_2021": ("2015-01-01", "2021-12-31"),
            "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
            "recent_2026": ("2026-01-01", "2026-12-31"),
            "complete_2015_2026": ("2015-01-01", "2026-12-31"),
        },
    },
    "proxy": {
        "directory": "experiment_r9_broad50_stage35_d10_20y_proxy",
        "open_close": Path("data/adjusted_open_close_20y_proxy.csv"),
        "start": "2006-08-01",
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_2026": ("2015-01-01", "2026-12-31"),
            "complete_2006_2026": ("2006-08-01", "2026-12-31"),
        },
    },
}


def boundary_target(
    current: np.ndarray,
    desired: np.ndarray,
    boundary_turnover: float,
) -> tuple[np.ndarray, float]:
    proposed = 0.5 * float(np.abs(desired - current).sum())
    if proposed <= boundary_turnover:
        return current.copy(), 0.0
    fraction = 1.0 - boundary_turnover / proposed
    target = current + fraction * (desired - current)
    executed = 0.5 * float(np.abs(target - current).sum())
    return target, executed


def material_risk_change(
    current: np.ndarray,
    desired: np.ndarray,
    *,
    threshold: float = 0.01,
) -> bool:
    indices = {asset: ASSETS.index(asset) for asset in ASSETS}
    growth_indices = [indices[asset] for asset in GROWTH_ASSETS]
    risky_indices = [
        index for index, asset in enumerate(ASSETS) if asset != "CASH"
    ]
    growth_change = abs(
        float(desired[growth_indices].sum())
        - float(current[growth_indices].sum())
    )
    gross_risk_change = abs(
        float(np.abs(desired[risky_indices]).sum())
        - float(np.abs(current[risky_indices]).sum())
    )
    hedge_change = abs(
        float(
            desired[indices["VIX_HEDGE"]]
            - current[indices["VIX_HEDGE"]]
        )
    )
    return max(growth_change, gross_risk_change, hedge_change) >= threshold


def simulate_trade_to_boundary(
    strategy_directory: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    start_date: str,
    boundary_turnover: float,
    mode: str,
    cost_bps: float,
    risk_change_threshold: float = 0.01,
    financing_spread_bps: float = 100.0,
) -> pd.DataFrame:
    source = Path("output") / strategy_directory
    weights = pd.read_csv(
        source / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        source / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    dates = (
        weights.index.intersection(daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    dates = dates[dates >= start_date]
    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    previous_date: pd.Timestamp | None = None
    rows: list[dict[str, float | int]] = []
    for date in dates:
        if previous_date is None:
            overnight_return = 0.0
        else:
            overnight_asset_returns = (
                opens.loc[date, ASSETS].to_numpy(dtype=float)
                / closes.loc[previous_date, ASSETS].to_numpy(dtype=float)
                - 1.0
            )
            overnight_return = float(current @ overnight_asset_returns)
            current = current * (1.0 + overnight_asset_returns) / (
                1.0 + overnight_return
            )
        model_trade = float(daily.loc[date, "turnover"]) > 1e-14
        turnover = 0.0
        full_risk_trade = False
        if model_trade:
            desired = weights.loc[date, ASSETS].to_numpy(dtype=float)
            proposed = 0.5 * float(np.abs(desired - current).sum())
            if mode == "full_target":
                target = desired
                turnover = proposed
            elif mode == "all_boundary":
                target, turnover = boundary_target(
                    current,
                    desired,
                    boundary_turnover,
                )
            elif mode == "risk_change_full":
                full_risk_trade = material_risk_change(
                    current,
                    desired,
                    threshold=risk_change_threshold,
                )
                if full_risk_trade:
                    target = desired
                    turnover = proposed
                else:
                    target, turnover = boundary_target(
                        current,
                        desired,
                        boundary_turnover,
                    )
            else:
                raise ValueError(f"Unknown boundary mode: {mode}")
            current = target
        intraday_asset_returns = (
            closes.loc[date, ASSETS].to_numpy(dtype=float)
            / opens.loc[date, ASSETS].to_numpy(dtype=float)
            - 1.0
        )
        intraday_return = float(current @ intraday_asset_returns)
        trading_cost = 2.0 * turnover * cost_bps / 10_000.0
        financing_cost = (
            max(-float(current[ASSETS.index("CASH")]), 0.0)
            * financing_spread_bps
            / 10_000.0
            / 252.0
        )
        net_return = (
            (1.0 + overnight_return) * (1.0 + intraday_return)
            - 1.0
            - trading_cost
            - financing_cost
        )
        gross_intraday_growth = 1.0 + intraday_return
        if gross_intraday_growth <= 0.0:
            raise RuntimeError("Non-positive intraday portfolio value")
        current = (
            current
            * (1.0 + intraday_asset_returns)
            / gross_intraday_growth
        )
        rows.append(
            {
                "date": date,
                "net_return": net_return,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "model_trade": int(model_trade),
                "executed_trade": int(turnover > 1e-14),
                "full_risk_trade": int(full_risk_trade),
            }
        )
        previous_date = date
    frame = pd.DataFrame(rows).set_index("date")
    frame["equity"] = (1.0 + frame["net_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    variants = [
        ("full_target", 0.00),
        *[
            (mode, boundary)
            for mode in ("risk_change_full", "all_boundary")
            for boundary in (0.01, 0.02, 0.03)
        ],
    ]
    rows: list[dict[str, float | int | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    for sample, settings in SAMPLES.items():
        opens, closes = load_adjusted_open_close(settings["open_close"])
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for cost_bps in (7.5, 15.0, 30.0):
            paths = {
                (mode, boundary): simulate_trade_to_boundary(
                    str(settings["directory"]),
                    opens,
                    closes,
                    start_date=str(settings["start"]),
                    boundary_turnover=boundary,
                    mode=mode,
                    cost_bps=cost_bps,
                )
                for mode, boundary in variants
            }
            baseline = paths[("full_target", 0.00)]
            for (mode, boundary), candidate in paths.items():
                common = baseline.index.intersection(candidate.index)
                for period, (start, end) in periods.items():
                    selected = common[
                        (common >= start) & (common <= end)
                    ]
                    base_metrics = performance_metrics(
                        baseline.loc[selected, "net_return"]
                    )
                    metrics = performance_metrics(
                        candidate.loc[selected, "net_return"]
                    )
                    rows.append(
                        {
                            "sample": sample,
                            "cost_bps": cost_bps,
                            "mode": mode,
                            "boundary_turnover": boundary,
                            "period": period,
                            **metrics,
                            "cagr_delta": (
                                metrics["cagr"] - base_metrics["cagr"]
                            ),
                            "sharpe_delta": (
                                metrics["sharpe"] - base_metrics["sharpe"]
                            ),
                            "max_drawdown_delta": (
                                metrics["max_drawdown"]
                                - base_metrics["max_drawdown"]
                            ),
                            "annualized_one_way_turnover": float(
                                candidate.loc[
                                    selected,
                                    "turnover",
                                ].mean()
                                * 252.0
                            ),
                        }
                    )
                if (
                    cost_bps == 7.5
                    and (mode, boundary) != ("full_target", 0.00)
                ):
                    for block_days in (21, 63, 126):
                        bootstrap_rows.append(
                            {
                                "sample": sample,
                                "mode": mode,
                                "boundary_turnover": boundary,
                                **paired_block_bootstrap(
                                    baseline.loc[common, "net_return"],
                                    candidate.loc[common, "net_return"],
                                    block_days=block_days,
                                ),
                            }
                        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT / "paired_bootstrap.csv",
        index=False,
    )
    full_periods = {
        "normal": "complete_2015_2026",
        "proxy": "complete_2006_2026",
    }
    summary = metrics.loc[
        metrics.apply(
            lambda row: row["period"] == full_periods[row["sample"]],
            axis=1,
        )
    ]
    columns = [
        "sample",
        "cost_bps",
        "mode",
        "boundary_turnover",
        "cagr_delta",
        "sharpe_delta",
        "max_drawdown",
        "annualized_one_way_turnover",
    ]
    print(summary[columns].round(6).to_string(index=False))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

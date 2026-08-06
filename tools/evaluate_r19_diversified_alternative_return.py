from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    relative_log_return,
)
from tools.evaluate_r12_volatility_managed_risk import major_drawdown_events
from tools.evaluate_r16_portable_managed_futures import (
    FinancingScenario,
    _load_strategy_return,
    _metrics,
    _period_metrics,
    calendar_cagr,
    leave_one_out_rows,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "r19_diversified_alternative_return"
PRICE_CACHE = (
    ROOT / "data" / "r19_aqmix_qspix_bil_adjusted_close.csv"
)
PRICE_METADATA = PRICE_CACHE.with_suffix(".csv.metadata.json")

NORMAL_R14 = (
    ROOT
    / "output"
    / "r14_incremental_trend_permission"
    / "normal_synthetic_both_sma200_daily.csv"
)
NORMAL_R11 = (
    ROOT
    / "output"
    / "r14_incremental_trend_permission"
    / "normal_synthetic_r11_baseline_daily.csv"
)
PROXY_R14 = (
    ROOT
    / "output"
    / "r14_incremental_trend_permission"
    / "proxy_synthetic_both_sma200_daily.csv"
)
PROXY_R11 = (
    ROOT
    / "output"
    / "r14_incremental_trend_permission"
    / "proxy_synthetic_r11_baseline_daily.csv"
)

TRADING_DAYS = 252
FUND_WEIGHTS = {"AQMIX": 0.50, "QSPIX": 0.50}
CANDIDATES = {"blend20": 0.20, "blend30": 0.30}
CUMULATIVE_TRIALS = 68
SCENARIOS = (
    FinancingScenario("current_100bps", 100.0),
    FinancingScenario("stress_150bps", 150.0),
)
OFFICIAL_CHECKS = {
    "AQMIX": {
        "as_of": pd.Timestamp("2026-06-30"),
        "inception": pd.Timestamp("2010-01-05"),
        "since_inception_cagr": 0.0407,
        "ytd": 0.0920,
    },
    "QSPIX": {
        "as_of": pd.Timestamp("2026-03-31"),
        "inception": pd.Timestamp("2013-10-30"),
        "since_inception_cagr": 0.0757,
        "ytd": 0.0983,
    },
}
OFFICIAL_TOLERANCE = 0.0030


def _adjusted_close_frame(
    raw: pd.DataFrame,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no R19 price data")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise ValueError("Yahoo response lacks adjusted-close prices")
        adjusted = raw["Adj Close"].copy()
    else:
        if len(tickers) != 1 or "Adj Close" not in raw:
            raise ValueError("Unexpected Yahoo adjusted-close schema")
        adjusted = raw[["Adj Close"]].rename(
            columns={"Adj Close": tickers[0]}
        )
    adjusted = adjusted.reindex(columns=list(tickers))
    adjusted.index = pd.to_datetime(adjusted.index).tz_localize(None)
    adjusted.index.name = "date"
    return adjusted.astype(float).sort_index()


def download_prices() -> pd.DataFrame:
    tickers = ("AQMIX", "QSPIX", "BIL")
    raw = yf.download(
        list(tickers),
        start="2010-01-01",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    prices = _adjusted_close_frame(raw, tickers).dropna(how="all")
    for ticker in tickers:
        if prices[ticker].first_valid_index() is None:
            raise ValueError(f"{ticker} has no valid adjusted-close data")
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICE_CACHE, index_label="date")
    digest = hashlib.sha256(PRICE_CACHE.read_bytes()).hexdigest()
    metadata = {
        "cache_sha256": digest,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": (
            "tools/evaluate_r19_diversified_alternative_return.py"
            "::download_prices"
        ),
        "market_data_source": "Yahoo Finance via yfinance",
        "market_data_adjustment": (
            "auto_adjust=False; Adj Close field selected"
        ),
        "tickers": list(tickers),
        "first_valid_date": {
            ticker: prices[ticker]
            .dropna()
            .index.min()
            .date()
            .isoformat()
            for ticker in tickers
        },
        "last_valid_date": {
            ticker: prices[ticker]
            .dropna()
            .index.max()
            .date()
            .isoformat()
            for ticker in tickers
        },
        "row_count": len(prices),
        "purpose": (
            "R19 actual diversified alternative-return validation; "
            "research total-return data, not an execution quote"
        ),
    }
    PRICE_METADATA.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return prices


def load_prices(*, refresh: bool = False) -> pd.DataFrame:
    if refresh or not PRICE_CACHE.exists():
        return download_prices()
    prices = pd.read_csv(
        PRICE_CACHE,
        index_col="date",
        parse_dates=True,
    )
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.astype(float).sort_index()


def _official_return_checks(
    prices: pd.DataFrame,
    ticker: str,
) -> list[dict[str, object]]:
    expected = OFFICIAL_CHECKS[ticker]
    as_of = expected["as_of"]
    inception = expected["inception"]
    assert isinstance(as_of, pd.Timestamp)
    assert isinstance(inception, pd.Timestamp)
    official = prices[ticker].loc[:as_of].dropna()
    prior_year = official.loc[: "2025-12-31"]
    current_year = official.loc["2026-01-01":]
    observed_cagr = calendar_cagr(official)
    observed_ytd = float(
        current_year.iloc[-1] / prior_year.iloc[-1] - 1.0
    )
    observed_inception = official.index.min()
    return [
        {
            "check": f"{ticker.lower()}_first_valid_date",
            "observed": observed_inception.date().isoformat(),
            "expected": inception.date().isoformat(),
            "pass": observed_inception == inception,
        },
        {
            "check": f"{ticker.lower()}_official_since_inception_cagr",
            "observed": observed_cagr,
            "expected": expected["since_inception_cagr"],
            "pass": (
                abs(observed_cagr - expected["since_inception_cagr"])
                <= OFFICIAL_TOLERANCE
            ),
        },
        {
            "check": f"{ticker.lower()}_official_2026_ytd",
            "observed": observed_ytd,
            "expected": expected["ytd"],
            "pass": (
                abs(observed_ytd - expected["ytd"])
                <= OFFICIAL_TOLERANCE
            ),
        },
    ]


def data_audit(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"AQMIX", "QSPIX", "BIL"}
    if not required.issubset(prices.columns):
        raise ValueError("Price cache must contain AQMIX, QSPIX, and BIL")
    common = prices[list(FUND_WEIGHTS) + ["BIL"]].dropna()
    if common.empty:
        raise ValueError("R19 instruments have no common observations")
    digest_matches = False
    if PRICE_METADATA.exists() and PRICE_CACHE.exists():
        metadata = json.loads(PRICE_METADATA.read_text(encoding="utf-8"))
        digest_matches = (
            metadata.get("cache_sha256")
            == hashlib.sha256(PRICE_CACHE.read_bytes()).hexdigest()
        )
    returns = common.pct_change(fill_method=None).dropna()
    rows: list[dict[str, object]] = [
        {
            "check": "strictly_increasing_unique_dates",
            "observed": bool(
                prices.index.is_monotonic_increasing
                and not prices.index.has_duplicates
            ),
            "expected": True,
            "pass": bool(
                prices.index.is_monotonic_increasing
                and not prices.index.has_duplicates
            ),
        },
        {
            "check": "cache_sha256_matches_metadata",
            "observed": digest_matches,
            "expected": True,
            "pass": digest_matches,
        },
        {
            "check": "common_observation_count",
            "observed": len(common),
            "expected": ">= 3000",
            "pass": len(common) >= 3000,
        },
        {
            "check": "plausible_daily_returns",
            "observed": float(returns.abs().max().max()),
            "expected": "<= 0.50",
            "pass": bool(returns.abs().max().max() <= 0.50),
        },
    ]
    for ticker in FUND_WEIGHTS:
        rows.extend(_official_return_checks(prices, ticker))
    return pd.DataFrame(rows)


def adjusted_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices[["AQMIX", "QSPIX", "BIL"]].pct_change(
        fill_method=None
    ).rename(columns={"BIL": "cash_return"})


def constant_notional_blend(
    base_returns: pd.Series,
    fund_returns: pd.DataFrame,
    cash_returns: pd.Series,
    overlay_weight: float,
    *,
    financing_spread_bps: float,
) -> pd.DataFrame:
    if overlay_weight < 0.0:
        raise ValueError("overlay_weight must be non-negative")
    aligned = pd.concat(
        [
            base_returns.rename("base_return"),
            fund_returns[list(FUND_WEIGHTS)],
            cash_returns.rename("cash_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    spread = financing_spread_bps / 10_000.0 / TRADING_DAYS
    for ticker, fraction in FUND_WEIGHTS.items():
        aligned[f"{ticker.lower()}_net_contribution"] = (
            overlay_weight
            * fraction
            * (aligned[ticker] - aligned["cash_return"] - spread)
        )
    contributions = [
        f"{ticker.lower()}_net_contribution"
        for ticker in FUND_WEIGHTS
    ]
    aligned["overlay_excess_return"] = aligned[contributions].sum(
        axis=1
    )
    aligned["candidate_return"] = (
        aligned["base_return"] + aligned["overlay_excess_return"]
    )
    if aligned["candidate_return"].le(-1.0).any():
        raise ValueError("R19 path produced a total loss day")
    aligned["overlay_weight"] = overlay_weight
    aligned["financing_spread_bps"] = financing_spread_bps
    return aligned


def monthly_reset_blend(
    base_returns: pd.Series,
    fund_returns: pd.DataFrame,
    cash_returns: pd.Series,
    overlay_weight: float,
    *,
    financing_spread_bps: float,
    one_way_cost_bps: float,
) -> pd.DataFrame:
    if overlay_weight < 0.0:
        raise ValueError("overlay_weight must be non-negative")
    aligned = pd.concat(
        [
            base_returns.rename("base_return"),
            fund_returns[list(FUND_WEIGHTS)],
            cash_returns.rename("cash_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    equity = 1.0
    fund_values = {ticker: 0.0 for ticker in FUND_WEIGHTS}
    loan_value = 0.0
    prior_month: pd.Period | None = None
    spread = financing_spread_bps / 10_000.0 / TRADING_DAYS
    cost_rate = one_way_cost_bps / 10_000.0
    rows: list[dict[str, float | bool]] = []
    for date, row in aligned.iterrows():
        start_equity = equity
        month = date.to_period("M")
        reset = prior_month is None or month != prior_month
        trading_cost = 0.0
        if reset:
            desired_values = {
                ticker: overlay_weight * fraction * equity
                for ticker, fraction in FUND_WEIGHTS.items()
            }
            trading_cost = sum(
                abs(desired_values[ticker] - fund_values[ticker])
                for ticker in FUND_WEIGHTS
            ) * cost_rate
            equity -= trading_cost
            fund_values = desired_values
            loan_value = overlay_weight * equity
        base_profit = equity * float(row["base_return"])
        fund_profits = {
            ticker: fund_values[ticker] * float(row[ticker])
            for ticker in FUND_WEIGHTS
        }
        loan_return = float(row["cash_return"]) + spread
        financing_cost = loan_value * loan_return
        equity += base_profit + sum(fund_profits.values()) - financing_cost
        for ticker in FUND_WEIGHTS:
            fund_values[ticker] *= 1.0 + float(row[ticker])
        loan_value *= 1.0 + loan_return
        candidate_return = equity / start_equity - 1.0
        if candidate_return <= -1.0:
            raise ValueError("R19 monthly path lost all capital")
        output_row: dict[str, float | bool] = {
            "candidate_return": candidate_return,
            "base_return": float(row["base_return"]),
            "cash_return": float(row["cash_return"]),
            "trading_cost": trading_cost / start_equity,
            "financing_cost": financing_cost / start_equity,
            "reset": reset,
            "effective_total_fund_notional": (
                sum(fund_values.values()) / equity
            ),
            "effective_loan_notional": loan_value / equity,
        }
        for ticker in FUND_WEIGHTS:
            output_row[f"{ticker.lower()}_return"] = float(row[ticker])
            output_row[f"effective_{ticker.lower()}_notional"] = (
                fund_values[ticker] / equity
            )
        rows.append(output_row)
        prior_month = month
    return pd.DataFrame(rows, index=aligned.index)


def shapley_contributions(
    path: pd.DataFrame,
) -> pd.DataFrame:
    base = path["base_return"]
    aqmix = path["aqmix_net_contribution"]
    qspix = path["qspix_net_contribution"]
    full = base + aqmix + qspix
    aqmix_only = base + aqmix
    qspix_only = base + qspix
    result = pd.DataFrame(index=path.index)
    result["aqmix_log_contribution"] = 0.5 * (
        np.log1p(aqmix_only)
        - np.log1p(base)
        + np.log1p(full)
        - np.log1p(qspix_only)
    )
    result["qspix_log_contribution"] = 0.5 * (
        np.log1p(qspix_only)
        - np.log1p(base)
        + np.log1p(full)
        - np.log1p(aqmix_only)
    )
    result["total_relative_log_return"] = (
        np.log1p(full) - np.log1p(base)
    )
    if not np.allclose(
        result["aqmix_log_contribution"]
        + result["qspix_log_contribution"],
        result["total_relative_log_return"],
        atol=1e-12,
    ):
        raise AssertionError("Shapley contributions do not reconcile")
    return result


def contribution_diagnostics(
    candidate: str,
    path: pd.DataFrame,
    r11_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contributions = shapley_contributions(path)
    overall = {
        column: float(contributions[column].mean() * TRADING_DAYS)
        for column in contributions
    }
    positive_total = (
        max(0.0, overall["aqmix_log_contribution"])
        + max(0.0, overall["qspix_log_contribution"])
    )
    dominant_share = (
        max(
            overall["aqmix_log_contribution"],
            overall["qspix_log_contribution"],
        )
        / positive_total
        if positive_total > 0.0
        else float("nan")
    )
    summary = pd.DataFrame(
        [
            {
                "candidate": candidate,
                **overall,
                "dominant_positive_contribution_share": dominant_share,
                "concentration_pass": bool(dominant_share <= 0.90),
            }
        ]
    )
    annual = (
        contributions.groupby(contributions.index.year)
        .mean()
        .mul(TRADING_DAYS)
        .reset_index(names="year")
    )
    annual.insert(0, "candidate", candidate)
    aligned = pd.concat(
        [r11_returns.rename("r11"), contributions],
        axis=1,
        join="inner",
    ).dropna()
    events: list[dict[str, object]] = []
    for number, (start, end) in enumerate(
        major_drawdown_events(aligned["r11"]),
        start=1,
    ):
        selected = aligned.loc[start:end]
        events.append(
            {
                "candidate": candidate,
                "event": number,
                "start": start,
                "end": end,
                "aqmix_log_contribution": float(
                    selected["aqmix_log_contribution"].sum()
                ),
                "qspix_log_contribution": float(
                    selected["qspix_log_contribution"].sum()
                ),
                "total_relative_log_return": float(
                    selected["total_relative_log_return"].sum()
                ),
            }
        )
    return summary, annual, pd.DataFrame(events)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    audit = data_audit(prices)
    audit.to_csv(OUTPUT / "data_audit.csv", index=False)
    returns = adjusted_returns(prices)
    funds = returns[list(FUND_WEIGHTS)]
    cash = returns["cash_return"]

    normal_r14 = _load_strategy_return(NORMAL_R14)
    normal_r11 = _load_strategy_return(NORMAL_R11)
    proxy_r14 = _load_strategy_return(PROXY_R14)
    proxy_r11 = _load_strategy_return(PROXY_R11)
    samples = {
        "normal_actual": {
            "r14": normal_r14,
            "r11": normal_r11,
            "periods": {
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
        },
        "expanded_actual": {
            "r14": proxy_r14,
            "r11": proxy_r11,
            "periods": {
                "early_2013_2019": (
                    "2013-10-30",
                    "2019-12-31",
                ),
                "late_2020_2026": (
                    "2020-01-01",
                    "2026-12-31",
                ),
                "complete_2013_2026": (
                    "2013-10-30",
                    "2026-12-31",
                ),
            },
        },
    }

    metric_rows: list[dict[str, object]] = []
    paths: dict[str, dict[str, pd.DataFrame]] = {}
    for sample, settings in samples.items():
        r14 = settings["r14"]
        r11 = settings["r11"]
        periods = settings["periods"]
        assert isinstance(r14, pd.Series)
        assert isinstance(r11, pd.Series)
        assert isinstance(periods, dict)
        sample_paths: dict[str, pd.DataFrame] = {}
        for scenario in SCENARIOS:
            for name, weight in CANDIDATES.items():
                path = constant_notional_blend(
                    r14,
                    funds,
                    cash,
                    weight,
                    financing_spread_bps=scenario.spread_bps,
                )
                if scenario.name == "current_100bps":
                    sample_paths[name] = path
                    path.to_csv(
                        OUTPUT / f"{sample}_{name}_daily.csv",
                        index_label="date",
                    )
                for period, (start, end) in periods.items():
                    for comparator, comparator_returns in (
                        ("r11", r11),
                        ("r14", r14),
                    ):
                        metric_rows.append(
                            _period_metrics(
                                sample,
                                scenario.name,
                                name,
                                period,
                                comparator,
                                comparator_returns,
                                path["candidate_return"],
                                start,
                                end,
                            )
                        )
        paths[sample] = sample_paths
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    names = list(CANDIDATES)
    family_rows: list[dict[str, object]] = []
    for sample in samples:
        r14 = samples[sample]["r14"]
        assert isinstance(r14, pd.Series)
        matrix = np.column_stack(
            [
                relative_log_return(
                    r14,
                    paths[sample][name]["candidate_return"],
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
    contribution_summaries: list[pd.DataFrame] = []
    contribution_years: list[pd.DataFrame] = []
    contribution_events: list[pd.DataFrame] = []
    for name in names:
        years, events = leave_one_out_rows(
            name,
            normal_r11,
            normal_r14,
            paths["normal_actual"][name]["candidate_return"],
        )
        year_rows.extend(years)
        event_rows.extend(events)
        summary, annual, event = contribution_diagnostics(
            name,
            paths["normal_actual"][name],
            normal_r11,
        )
        contribution_summaries.append(summary)
        contribution_years.append(annual)
        contribution_events.append(event)
    years = pd.DataFrame(year_rows)
    events = pd.DataFrame(event_rows)
    years.to_csv(OUTPUT / "leave_one_year_out.csv", index=False)
    events.to_csv(
        OUTPUT / "leave_one_r11_drawdown_event_out.csv",
        index=False,
    )
    contribution_summary = pd.concat(
        contribution_summaries, ignore_index=True
    )
    contribution_summary.to_csv(
        OUTPUT / "contribution_concentration.csv",
        index=False,
    )
    pd.concat(contribution_years, ignore_index=True).to_csv(
        OUTPUT / "contribution_by_year.csv",
        index=False,
    )
    pd.concat(contribution_events, ignore_index=True).to_csv(
        OUTPUT / "contribution_by_r11_drawdown_event.csv",
        index=False,
    )

    neighborhood_rows: list[dict[str, object]] = []
    for name, weight in CANDIDATES.items():
        for neighbor in (weight - 0.025, weight + 0.025):
            path = constant_notional_blend(
                normal_r14,
                funds,
                cash,
                neighbor,
                financing_spread_bps=100.0,
            )
            aligned = pd.concat(
                [
                    normal_r11.rename("r11"),
                    path["candidate_return"].rename("candidate"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            candidate_metrics = _metrics(aligned["candidate"])
            r11_metrics = _metrics(aligned["r11"])
            neighborhood_rows.append(
                {
                    "candidate": name,
                    "center_weight": weight,
                    "neighbor_weight": neighbor,
                    **{
                        f"candidate_{key}": value
                        for key, value in candidate_metrics.items()
                    },
                    "r11_max_drawdown": r11_metrics["max_drawdown"],
                    "ridge_pass": bool(
                        candidate_metrics["cagr"] >= 0.25
                        and candidate_metrics["max_drawdown"]
                        >= r11_metrics["max_drawdown"]
                    ),
                }
            )
    neighborhoods = pd.DataFrame(neighborhood_rows)
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    monthly_rows: list[dict[str, object]] = []
    for name, weight in CANDIDATES.items():
        path = monthly_reset_blend(
            normal_r14,
            funds,
            cash,
            weight,
            financing_spread_bps=100.0,
            one_way_cost_bps=10.0,
        )
        path.to_csv(
            OUTPUT / f"normal_actual_{name}_monthly_reset_daily.csv",
            index_label="date",
        )
        aligned = pd.concat(
            [
                normal_r11.rename("r11"),
                path["candidate_return"].rename("candidate"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        candidate_metrics = _metrics(aligned["candidate"])
        r11_metrics = _metrics(aligned["r11"])
        monthly_rows.append(
            {
                "candidate": name,
                "overlay_weight": weight,
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_metrics.items()
                },
                "r11_max_drawdown": r11_metrics["max_drawdown"],
                "implementation_pass": bool(
                    candidate_metrics["cagr"] >= 0.25
                    and candidate_metrics["max_drawdown"]
                    >= r11_metrics["max_drawdown"]
                ),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(
        OUTPUT / "monthly_reset_cost_stress.csv",
        index=False,
    )

    worst_rows: list[pd.DataFrame] = []
    for name in names:
        path = paths["normal_actual"][name].copy()
        path["candidate"] = name
        path["relative_to_r14"] = (
            path["candidate_return"] - path["base_return"]
        )
        worst_rows.append(
            path.nsmallest(20, "candidate_return").reset_index()
        )
    pd.concat(worst_rows, ignore_index=True).to_csv(
        OUTPUT / "worst_20_days.csv",
        index=False,
    )

    acceptance_rows: list[dict[str, object]] = []
    for name in names:
        central = metrics.loc[
            (metrics["sample"] == "normal_actual")
            & (metrics["scenario"] == "current_100bps")
            & (metrics["candidate"] == name)
        ]
        stress = metrics.loc[
            (metrics["sample"] == "normal_actual")
            & (metrics["scenario"] == "stress_150bps")
            & (metrics["candidate"] == name)
        ]
        expanded = metrics.loc[
            (metrics["sample"] == "expanded_actual")
            & (metrics["scenario"] == "current_100bps")
            & (metrics["candidate"] == name)
        ]

        def row(
            frame: pd.DataFrame,
            period: str,
            comparator: str,
        ) -> pd.Series:
            selected = frame.loc[
                (frame["period"] == period)
                & (frame["comparator"] == comparator)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one {name}/{period}/{comparator} row"
                )
            return selected.iloc[0]

        complete_r11 = row(central, "complete_2015_2026", "r11")
        complete_r14 = row(central, "complete_2015_2026", "r14")
        development_r11 = row(
            central, "development_2015_2021", "r11"
        )
        development_r14 = row(
            central, "development_2015_2021", "r14"
        )
        holdout_r11 = row(central, "holdout_2022_2025", "r11")
        holdout_r14 = row(central, "holdout_2022_2025", "r14")
        stress_r11 = row(stress, "complete_2015_2026", "r11")
        stress_r14 = row(stress, "complete_2015_2026", "r14")
        expanded_complete_r11 = row(
            expanded, "complete_2013_2026", "r11"
        )
        expanded_complete_r14 = row(
            expanded, "complete_2013_2026", "r14"
        )
        expanded_early_r14 = row(
            expanded, "early_2013_2019", "r14"
        )
        expanded_late_r14 = row(
            expanded, "late_2020_2026", "r14"
        )
        candidate_family = family.loc[
            (family["sample"] == "normal_actual")
            & (family["candidate"] == name)
        ]
        family_pass = bool(
            candidate_family[
                "cumulative_trial_adjusted_p_value"
            ].le(0.05).all()
        )
        robustness_pass = bool(
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
        monthly_row = monthly.loc[
            monthly["candidate"] == name
        ].iloc[0]
        concentration_row = contribution_summary.loc[
            contribution_summary["candidate"] == name
        ].iloc[0]
        expanded_pass = bool(
            expanded_complete_r11[
                "annualized_relative_log_return"
            ]
            > 0.0
            and expanded_complete_r14[
                "annualized_relative_log_return"
            ]
            > 0.0
            and expanded_complete_r11["candidate_max_drawdown"]
            >= expanded_complete_r11["baseline_max_drawdown"]
            and expanded_early_r14[
                "annualized_relative_log_return"
            ]
            > 0.0
            and expanded_late_r14[
                "annualized_relative_log_return"
            ]
            > 0.0
        )
        statistical_economic_pass = bool(
            complete_r11["candidate_cagr"] >= 0.25
            and complete_r11["candidate_max_drawdown"]
            >= complete_r11["baseline_max_drawdown"]
            and development_r11["annualized_relative_log_return"] > 0.0
            and development_r14["annualized_relative_log_return"] > 0.0
            and holdout_r11["annualized_relative_log_return"] > 0.0
            and holdout_r14["annualized_relative_log_return"] > 0.0
            and stress_r14["annualized_relative_log_return"] > 0.0
            and stress_r11["candidate_max_drawdown"]
            >= stress_r11["baseline_max_drawdown"]
            and family_pass
            and robustness_pass
            and ridge_pass
            and bool(monthly_row["implementation_pass"])
            and expanded_pass
            and bool(concentration_row["concentration_pass"])
        )
        data_pass = bool(audit["pass"].all())
        long_mechanism_evidence_pass = False
        product_access_pass = False
        financing_and_risk_limit_pass = False
        acceptance_rows.append(
            {
                "candidate": name,
                "overlay_weight": CANDIDATES[name],
                "complete_cagr": complete_r11["candidate_cagr"],
                "complete_max_drawdown": (
                    complete_r11["candidate_max_drawdown"]
                ),
                "r11_max_drawdown": (
                    complete_r11["baseline_max_drawdown"]
                ),
                "complete_relative_to_r14": (
                    complete_r14["annualized_relative_log_return"]
                ),
                "development_relative_to_r14": (
                    development_r14["annualized_relative_log_return"]
                ),
                "holdout_relative_to_r14": (
                    holdout_r14["annualized_relative_log_return"]
                ),
                "stress_relative_to_r14": (
                    stress_r14["annualized_relative_log_return"]
                ),
                "family_multiple_testing_pass": family_pass,
                "leave_one_out_pass": robustness_pass,
                "parameter_ridge_pass": ridge_pass,
                "monthly_implementation_pass": bool(
                    monthly_row["implementation_pass"]
                ),
                "expanded_actual_pass": expanded_pass,
                "contribution_concentration_pass": bool(
                    concentration_row["concentration_pass"]
                ),
                "data_audit_pass": data_pass,
                "long_mechanism_evidence_pass": (
                    long_mechanism_evidence_pass
                ),
                "product_access_confirmed": product_access_pass,
                "financing_and_risk_limit_pass": (
                    financing_and_risk_limit_pass
                ),
                "statistical_economic_pass": (
                    statistical_economic_pass
                ),
                "production_pass": bool(
                    statistical_economic_pass
                    and data_pass
                    and long_mechanism_evidence_pass
                    and product_access_pass
                    and financing_and_risk_limit_pass
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

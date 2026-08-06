from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from regime_strategy.report import performance_metrics
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    metric_delta,
    relative_log_return,
)
from tools.evaluate_r12_volatility_managed_risk import (
    major_drawdown_events,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "r16_portable_managed_futures"
PRICE_CACHE = ROOT / "data" / "r16_aqmix_bil_adjusted_close.csv"
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
CANDIDATES = {
    "overlay10": 0.10,
    "overlay15": 0.15,
    "overlay20": 0.20,
}
CUMULATIVE_TRIALS = 62
OFFICIAL_AS_OF = pd.Timestamp("2026-06-30")
OFFICIAL_SINCE_INCEPTION_CAGR = 0.0407
OFFICIAL_2026_YTD = 0.0920
OFFICIAL_TOLERANCE = 0.0030


@dataclass(frozen=True)
class FinancingScenario:
    name: str
    spread_bps: float


SCENARIOS = (
    FinancingScenario("current_100bps", 100.0),
    FinancingScenario("stress_150bps", 150.0),
)


def _adjusted_close_frame(
    raw: pd.DataFrame,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no AQMIX/BIL data")
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
    tickers = ("AQMIX", "BIL")
    raw = yf.download(
        list(tickers),
        start="2010-01-01",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    prices = _adjusted_close_frame(raw, tickers).dropna(how="all")
    if prices["AQMIX"].first_valid_index() is None:
        raise ValueError("AQMIX has no valid adjusted-close observations")
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICE_CACHE, index_label="date")
    digest = hashlib.sha256(PRICE_CACHE.read_bytes()).hexdigest()
    metadata = {
        "cache_sha256": digest,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": (
            "tools/evaluate_r16_portable_managed_futures.py"
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
            "R16 actual managed-futures portable-alpha validation; "
            "research data, not an execution quote"
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


def calendar_cagr(prices: pd.Series) -> float:
    clean = prices.dropna()
    if len(clean) < 2:
        return float("nan")
    elapsed_days = (clean.index[-1] - clean.index[0]).days
    if elapsed_days <= 0:
        return float("nan")
    return float(
        (clean.iloc[-1] / clean.iloc[0])
        ** (365.2425 / elapsed_days)
        - 1.0
    )


def data_audit(prices: pd.DataFrame) -> pd.DataFrame:
    if not {"AQMIX", "BIL"}.issubset(prices.columns):
        raise ValueError("Price cache must contain AQMIX and BIL")
    common = prices[["AQMIX", "BIL"]].dropna()
    if common.empty:
        raise ValueError("AQMIX and BIL have no common observations")
    official = prices["AQMIX"].loc[:OFFICIAL_AS_OF].dropna()
    prior_year = official.loc[: "2025-12-31"]
    current_year = official.loc["2026-01-01":]
    since_inception = calendar_cagr(official)
    ytd = float(
        current_year.iloc[-1] / prior_year.iloc[-1] - 1.0
    )
    returns = common.pct_change(fill_method=None).dropna()
    digest_matches = False
    if PRICE_METADATA.exists() and PRICE_CACHE.exists():
        metadata = json.loads(PRICE_METADATA.read_text(encoding="utf-8"))
        digest_matches = (
            metadata.get("cache_sha256")
            == hashlib.sha256(PRICE_CACHE.read_bytes()).hexdigest()
        )
    rows = [
        {
            "check": "strictly_increasing_unique_dates",
            "observed": bool(
                common.index.is_monotonic_increasing
                and not common.index.has_duplicates
            ),
            "expected": True,
            "pass": bool(
                common.index.is_monotonic_increasing
                and not common.index.has_duplicates
            ),
        },
        {
            "check": "cache_sha256_matches_metadata",
            "observed": digest_matches,
            "expected": True,
            "pass": digest_matches,
        },
        {
            "check": "aqmix_first_valid_date",
            "observed": common.index.min().date().isoformat(),
            "expected": "2010-01-05",
            "pass": common.index.min() == pd.Timestamp("2010-01-05"),
        },
        {
            "check": "aqmix_official_since_inception_cagr",
            "observed": since_inception,
            "expected": OFFICIAL_SINCE_INCEPTION_CAGR,
            "pass": (
                abs(
                    since_inception
                    - OFFICIAL_SINCE_INCEPTION_CAGR
                )
                <= OFFICIAL_TOLERANCE
            ),
        },
        {
            "check": "aqmix_official_2026_ytd",
            "observed": ytd,
            "expected": OFFICIAL_2026_YTD,
            "pass": abs(ytd - OFFICIAL_2026_YTD) <= OFFICIAL_TOLERANCE,
        },
        {
            "check": "plausible_daily_returns",
            "observed": float(returns.abs().max().max()),
            "expected": "<= 0.50",
            "pass": bool(returns.abs().max().max() <= 0.50),
        },
    ]
    return pd.DataFrame(rows)


def adjusted_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices[["AQMIX", "BIL"]].pct_change(
        fill_method=None
    )
    return returns.rename(
        columns={"AQMIX": "fund_return", "BIL": "cash_return"}
    )


def constant_notional_stack(
    base_returns: pd.Series,
    fund_returns: pd.Series,
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
            fund_returns.rename("fund_return"),
            cash_returns.rename("cash_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    spread = financing_spread_bps / 10_000.0 / TRADING_DAYS
    aligned["overlay_excess_return"] = (
        aligned["fund_return"] - aligned["cash_return"] - spread
    )
    aligned["candidate_return"] = (
        aligned["base_return"]
        + overlay_weight * aligned["overlay_excess_return"]
    )
    if aligned["candidate_return"].le(-1.0).any():
        raise ValueError("Portable-alpha path produced a total loss day")
    aligned["overlay_weight"] = overlay_weight
    aligned["financing_spread_bps"] = financing_spread_bps
    return aligned


def monthly_reset_stack(
    base_returns: pd.Series,
    fund_returns: pd.Series,
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
            fund_returns.rename("fund_return"),
            cash_returns.rename("cash_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    equity = 1.0
    fund_value = 0.0
    loan_value = 0.0
    prior_month: pd.Period | None = None
    rows: list[dict[str, float | bool]] = []
    spread = financing_spread_bps / 10_000.0 / TRADING_DAYS
    cost_rate = one_way_cost_bps / 10_000.0
    for date, row in aligned.iterrows():
        start_equity = equity
        month = date.to_period("M")
        reset = prior_month is None or month != prior_month
        trading_cost = 0.0
        if reset:
            desired = overlay_weight * equity
            trading_cost = abs(desired - fund_value) * cost_rate
            equity -= trading_cost
            fund_value = desired
            loan_value = desired
        base_profit = equity * float(row["base_return"])
        fund_profit = fund_value * float(row["fund_return"])
        loan_return = float(row["cash_return"]) + spread
        financing_cost = loan_value * loan_return
        equity += base_profit + fund_profit - financing_cost
        fund_value *= 1.0 + float(row["fund_return"])
        loan_value *= 1.0 + loan_return
        candidate_return = equity / start_equity - 1.0
        if candidate_return <= -1.0:
            raise ValueError("Monthly portable-alpha path lost all capital")
        rows.append(
            {
                "candidate_return": candidate_return,
                "base_return": float(row["base_return"]),
                "fund_return": float(row["fund_return"]),
                "cash_return": float(row["cash_return"]),
                "trading_cost": trading_cost / start_equity,
                "financing_cost": financing_cost / start_equity,
                "reset": reset,
                "effective_fund_notional": fund_value / equity,
                "effective_loan_notional": loan_value / equity,
            }
        )
        prior_month = month
    return pd.DataFrame(rows, index=aligned.index)


def _load_strategy_return(path: Path) -> pd.Series:
    daily = pd.read_csv(path, index_col="date", parse_dates=True)
    return daily["net_return"].astype(float).rename(path.stem)


def _period_metrics(
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
    return {
        "sample": sample,
        "scenario": scenario,
        "candidate": candidate,
        "period": period,
        "comparator": comparator,
        "observations": len(selected),
        "annualized_relative_log_return": float(
            (
                np.log1p(selected["candidate"])
                - np.log1p(selected["baseline"])
            ).mean()
            * TRADING_DAYS
        ),
        **metric_delta(selected["baseline"], selected["candidate"]),
    }


def leave_one_out_rows(
    candidate: str,
    r11_returns: pd.Series,
    r14_returns: pd.Series,
    candidate_returns: pd.Series,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    aligned = pd.concat(
        [
            r11_returns.rename("r11"),
            r14_returns.rename("r14"),
            candidate_returns.rename("candidate"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    relative = (
        np.log1p(aligned["candidate"]) - np.log1p(aligned["r14"])
    )
    year_rows: list[dict[str, object]] = []
    for year in sorted(aligned.index.year.unique()):
        kept = relative.loc[relative.index.year != year]
        year_rows.append(
            {
                "candidate": candidate,
                "removed_year": int(year),
                "annualized_relative_log_return": float(
                    kept.mean() * TRADING_DAYS
                ),
            }
        )
    event_rows: list[dict[str, object]] = []
    for number, (start, end) in enumerate(
        major_drawdown_events(aligned["r11"]),
        start=1,
    ):
        kept = relative.loc[
            (relative.index < start) | (relative.index > end)
        ]
        event_rows.append(
            {
                "candidate": candidate,
                "event": number,
                "removed_start": start,
                "removed_end": end,
                "annualized_relative_log_return": float(
                    kept.mean() * TRADING_DAYS
                ),
            }
        )
    return year_rows, event_rows


def _metrics(returns: pd.Series) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in performance_metrics(returns).items()
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    audit = data_audit(prices)
    audit.to_csv(OUTPUT / "data_audit.csv", index=False)
    returns = adjusted_returns(prices)
    fund = returns["fund_return"]
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
        },
    }

    metric_rows: list[dict[str, object]] = []
    paths: dict[str, dict[str, pd.Series]] = {}
    for sample, settings in samples.items():
        r14 = settings["r14"]
        r11 = settings["r11"]
        periods = settings["periods"]
        assert isinstance(r14, pd.Series)
        assert isinstance(r11, pd.Series)
        assert isinstance(periods, dict)
        sample_paths: dict[str, pd.Series] = {}
        for scenario in SCENARIOS:
            for name, weight in CANDIDATES.items():
                path = constant_notional_stack(
                    r14,
                    fund,
                    cash,
                    weight,
                    financing_spread_bps=scenario.spread_bps,
                )
                candidate_returns = path["candidate_return"]
                if scenario.name == "current_100bps":
                    sample_paths[name] = candidate_returns
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
                                candidate_returns,
                                start,
                                end,
                            )
                        )
        paths[sample] = sample_paths
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    family_rows: list[dict[str, object]] = []
    names = list(CANDIDATES)
    for sample in ("normal_actual", "expanded_actual"):
        r14 = samples[sample]["r14"]
        assert isinstance(r14, pd.Series)
        matrix = np.column_stack(
            [
                relative_log_return(r14, paths[sample][name])
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
            normal_r11,
            normal_r14,
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
    for name, weight in CANDIDATES.items():
        for neighbor in (weight - 0.025, weight + 0.025):
            path = constant_notional_stack(
                normal_r14,
                fund,
                cash,
                neighbor,
                financing_spread_bps=100.0,
            )
            candidate_returns = path["candidate_return"]
            aligned = pd.concat(
                [
                    normal_r11.rename("r11"),
                    candidate_returns.rename("candidate"),
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
    monthly_paths: dict[str, pd.Series] = {}
    for name, weight in CANDIDATES.items():
        path = monthly_reset_stack(
            normal_r14,
            fund,
            cash,
            weight,
            financing_spread_bps=100.0,
            one_way_cost_bps=10.0,
        )
        monthly_paths[name] = path["candidate_return"]
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
        trial_metrics = _metrics(aligned["candidate"])
        r11_metrics = _metrics(aligned["r11"])
        monthly_rows.append(
            {
                "candidate": name,
                "overlay_weight": weight,
                **{
                    f"candidate_{key}": value
                    for key, value in trial_metrics.items()
                },
                "r11_max_drawdown": r11_metrics["max_drawdown"],
                "implementation_pass": bool(
                    trial_metrics["cagr"] >= 0.25
                    and trial_metrics["max_drawdown"]
                    >= r11_metrics["max_drawdown"]
                ),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(
        OUTPUT / "monthly_reset_cost_stress.csv",
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

        complete_r11 = row(
            central, "complete_2015_2026", "r11"
        )
        complete_r14 = row(
            central, "complete_2015_2026", "r14"
        )
        development_r11 = row(
            central, "development_2015_2021", "r11"
        )
        development_r14 = row(
            central, "development_2015_2021", "r14"
        )
        holdout_r11 = row(
            central, "holdout_2022_2025", "r11"
        )
        holdout_r14 = row(
            central, "holdout_2022_2025", "r14"
        )
        stress_r11 = row(
            stress, "complete_2015_2026", "r11"
        )
        stress_r14 = row(
            stress, "complete_2015_2026", "r14"
        )
        expanded_complete_r11 = row(
            expanded, "complete_2010_2026", "r11"
        )
        expanded_complete_r14 = row(
            expanded, "complete_2010_2026", "r14"
        )
        expanded_early_r14 = row(
            expanded, "early_2010_2017", "r14"
        )
        expanded_late_r14 = row(
            expanded, "late_2018_2026", "r14"
        )
        candidate_family = family.loc[
            (family["sample"] == "normal_actual")
            & (family["candidate"] == name)
        ]
        family_pass = bool(
            (
                candidate_family[
                    "cumulative_trial_adjusted_p_value"
                ]
                <= 0.05
            ).all()
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
            and expanded_complete_r11[
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
        data_pass = bool(audit["pass"].all())
        product_access_pass = False
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
                "expanded_actual_pass": bool(
                    expanded_complete_r11[
                        "annualized_relative_log_return"
                    ]
                    > 0.0
                    and expanded_complete_r14[
                        "annualized_relative_log_return"
                    ]
                    > 0.0
                    and expanded_complete_r11[
                        "candidate_max_drawdown"
                    ]
                    >= expanded_complete_r11[
                        "baseline_max_drawdown"
                    ]
                    and expanded_early_r14[
                        "annualized_relative_log_return"
                    ]
                    > 0.0
                    and expanded_late_r14[
                        "annualized_relative_log_return"
                    ]
                    > 0.0
                ),
                "data_audit_pass": data_pass,
                "product_access_and_financing_confirmed": (
                    product_access_pass
                ),
                "statistical_economic_pass": (
                    statistical_economic_pass
                ),
                "production_pass": bool(
                    statistical_economic_pass
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

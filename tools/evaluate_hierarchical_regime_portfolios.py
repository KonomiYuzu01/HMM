from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from regime_strategy.portfolio import (
    apply_daily_growth_risk_reduction,
    apply_growth_stress_guard,
)


PRODUCTION = Path("output/paper_core_growth_gold20_daily_risk_netted_ensemble")
DESTINATION = Path("output/hierarchical_regime_portfolios")
PRICE_PATH = Path("data/prices_vix_hedge.csv")
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
DEVELOPMENT = ("2015-01-01", "2021-12-31")
HOLDOUT = ("2022-01-01", "2025-12-31")
COMPLETE = ("2015-01-01", "2025-12-31")
STATE_ORDER = [
    "quiet_bull",
    "normal_bull",
    "fragile_bull",
    "correction",
    "rebound",
    "inflation_defense",
    "deflation_defense",
    "cash_stress",
    "crisis_decline",
    "crisis_recovery",
]
PRODUCTION_ANCHOR = "production_anchor"


def weights(**values: float) -> dict[str, float]:
    result = {asset: 0.0 for asset in ASSETS}
    result.update(values)
    if not np.isclose(sum(result.values()), 1.0):
        raise ValueError("Portfolio weights must sum to one")
    if min(result.values()) < 0.0:
        raise ValueError("Portfolio weights must be long-only")
    return result


PORTFOLIO_LIBRARY = {
    "core_404020": weights(QQQ=0.40, SEMIS=0.40, GOLD=0.20),
    "qqq_quality": weights(QQQ=0.60, SEMIS=0.20, GOLD=0.20),
    "semis_growth": weights(QQQ=0.20, SEMIS=0.60, GOLD=0.20),
    "balanced_growth": weights(QQQ=0.30, SEMIS=0.30, GOLD=0.20, CASH=0.20),
    "gold_defense": weights(QQQ=0.10, SEMIS=0.10, GOLD=0.40, CASH=0.40),
    "bond_defense": weights(QQQ=0.10, SEMIS=0.10, BOND=0.40, CASH=0.40),
    "balanced_defense": weights(
        QQQ=0.10,
        SEMIS=0.10,
        BOND=0.20,
        GOLD=0.20,
        CASH=0.40,
    ),
    "inflation_defense": weights(
        QQQ=0.10,
        SEMIS=0.10,
        GOLD=0.30,
        OIL=0.10,
        USD=0.10,
        CASH=0.30,
    ),
    "cash": weights(CASH=1.0),
}

ECONOMIC_MAP = {
    "quiet_bull": "semis_growth",
    "normal_bull": "core_404020",
    "fragile_bull": "balanced_growth",
    "correction": "gold_defense",
    "rebound": "balanced_growth",
    "inflation_defense": "inflation_defense",
    "deflation_defense": "bond_defense",
    "cash_stress": "cash",
    "crisis_decline": "balanced_defense",
    "crisis_recovery": "balanced_growth",
}

ALLOWED_PORTFOLIOS = {
    "quiet_bull": ["core_404020", "qqq_quality", "semis_growth"],
    "normal_bull": [
        "core_404020",
        "qqq_quality",
        "semis_growth",
        "balanced_growth",
    ],
    "fragile_bull": [
        "qqq_quality",
        "balanced_growth",
        "gold_defense",
        "balanced_defense",
    ],
    "correction": [
        "balanced_growth",
        "gold_defense",
        "balanced_defense",
        "cash",
    ],
    "rebound": [
        "core_404020",
        "qqq_quality",
        "semis_growth",
        "balanced_growth",
    ],
    "inflation_defense": ["inflation_defense", "gold_defense", "cash"],
    "deflation_defense": ["bond_defense", "balanced_defense", "cash"],
    "cash_stress": ["gold_defense", "balanced_defense", "cash"],
    "crisis_decline": [
        "gold_defense",
        "bond_defense",
        "balanced_defense",
        "cash",
    ],
    "crisis_recovery": [
        "balanced_growth",
        "gold_defense",
        "balanced_defense",
        "cash",
    ],
}


def annualized_bipower_volatility(
    returns: pd.Series,
    window: int,
    annualization: int = 252,
) -> pd.Series:
    absolute = returns.abs()
    product = absolute * absolute.shift(1)
    bipower_variance = (np.pi / 2.0) * product.rolling(window).mean()
    return np.sqrt(annualization * bipower_variance.clip(lower=0.0))


def classify_regime_row(row: pd.Series) -> str:
    high_volatility = row["volatility_state"] == "high"
    jump = bool(row["jump_ratio"] >= 1.25)
    backwardation = bool(row["vix_term_ratio"] > 1.0)
    risk_on = bool(row["risk_on_votes"] >= 2)

    crisis_trigger = backwardation or (high_volatility and jump)
    if crisis_trigger and row["growth_fast_trend"] <= 0.0:
        return "crisis_decline"
    if crisis_trigger and high_volatility:
        return "crisis_recovery"
    if risk_on:
        if high_volatility or jump or backwardation:
            return "fragile_bull"
        if row["volatility_state"] == "low":
            return "quiet_bull"
        return "normal_bull"
    if row["growth_fast_trend"] > 0.0:
        return "rebound"
    if (
        row["growth_slow_trend"] > 0.0
        and row["growth_fast_trend"] <= 0.0
    ):
        return "correction"
    if (
        row["bond_trend"] < 0.0
        and row["stock_bond_correlation"] > 0.0
    ):
        return "inflation_defense"
    if row["bond_trend"] > 0.0:
        return "deflation_defense"
    return "cash_stress"


def load_regime_schedule(
    production_directory: Path = PRODUCTION,
) -> pd.DataFrame:
    member_frames: list[pd.DataFrame] = []
    for seed in (7, 42, 123):
        frame = pd.read_csv(
            production_directory
            / "members"
            / f"seed_{seed}"
            / "regimes.csv",
            index_col=0,
            parse_dates=True,
        )
        member_frames.append(
            frame[
                [
                    "paper_risk_on_candidate",
                    "dominant_template",
                    "dominant_probability",
                ]
            ].add_suffix(f"__{seed}")
        )
    schedule = pd.concat(member_frames, axis=1, join="inner")
    risk_columns = [
        f"paper_risk_on_candidate__{seed}" for seed in (7, 42, 123)
    ]
    template_columns = [f"dominant_template__{seed}" for seed in (7, 42, 123)]
    probability_columns = [
        f"dominant_probability__{seed}" for seed in (7, 42, 123)
    ]
    schedule["risk_on_votes"] = schedule[risk_columns].sum(axis=1)
    schedule["dominant_template"] = schedule[template_columns].mode(axis=1)[0]
    schedule["dominant_probability"] = schedule[probability_columns].median(axis=1)
    return schedule[
        ["risk_on_votes", "dominant_template", "dominant_probability"]
    ].copy()


def build_causal_signals(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices[ASSETS].pct_change(fill_method=None)
    spx_returns = returns["SPX"]
    realized_volatility = spx_returns.rolling(20).std(ddof=1) * np.sqrt(252.0)
    known_volatility = realized_volatility.shift(1)
    low_threshold = known_volatility.rolling(756, min_periods=504).quantile(0.45)
    high_threshold = known_volatility.rolling(756, min_periods=504).quantile(0.90)
    slow_bipower = annualized_bipower_volatility(spx_returns, 60).shift(1)

    growth_log_return = (
        np.log1p(returns[["QQQ", "SEMIS"]].clip(lower=-0.999999)).mean(axis=1)
        - np.log1p(returns["CASH"].clip(lower=-0.999999))
    )
    bond_log_return = np.log1p(
        returns["BOND"].clip(lower=-0.999999)
    ) - np.log1p(returns["CASH"].clip(lower=-0.999999))
    gold_log_return = np.log1p(
        returns["GOLD"].clip(lower=-0.999999)
    ) - np.log1p(returns["CASH"].clip(lower=-0.999999))

    signals = pd.DataFrame(index=prices.index)
    signals["realized_volatility"] = known_volatility
    signals["low_volatility_threshold"] = low_threshold
    signals["high_volatility_threshold"] = high_threshold
    signals["volatility_state"] = np.select(
        [
            known_volatility <= low_threshold,
            known_volatility >= high_threshold,
        ],
        ["low", "high"],
        default="medium",
    )
    signals["jump_ratio"] = known_volatility / slow_bipower.replace(0.0, np.nan)
    signals["vix_term_ratio"] = (prices["VIX"] / prices["VIX3M"]).shift(1)
    signals["growth_fast_trend"] = growth_log_return.rolling(63).sum().shift(1)
    signals["growth_slow_trend"] = growth_log_return.rolling(252).sum().shift(1)
    signals["bond_trend"] = bond_log_return.rolling(126).sum().shift(1)
    signals["gold_trend"] = gold_log_return.rolling(126).sum().shift(1)
    signals["stock_bond_correlation"] = (
        returns["SPX"].rolling(60).corr(returns["BOND"]).shift(1)
    )
    return signals


def build_daily_states(
    schedule: pd.DataFrame,
    signals: pd.DataFrame,
    daily_index: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    scheduled = schedule.join(signals, how="left").dropna(
        subset=[
            "realized_volatility",
            "low_volatility_threshold",
            "high_volatility_threshold",
            "jump_ratio",
            "vix_term_ratio",
            "growth_fast_trend",
            "growth_slow_trend",
            "bond_trend",
            "gold_trend",
        ]
    )
    scheduled["state"] = scheduled.apply(classify_regime_row, axis=1)
    daily_state = scheduled["state"].reindex(daily_index).ffill()
    return scheduled, daily_state


def build_factorized_daily_states(
    schedule: pd.DataFrame,
    signals: pd.DataFrame,
    daily_index: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    scheduled_features = schedule.reindex(daily_index).ffill()
    daily = scheduled_features.join(
        signals.reindex(daily_index),
        how="left",
    ).dropna(
        subset=[
            "risk_on_votes",
            "realized_volatility",
            "low_volatility_threshold",
            "high_volatility_threshold",
            "jump_ratio",
            "vix_term_ratio",
            "growth_fast_trend",
            "growth_slow_trend",
            "bond_trend",
            "gold_trend",
        ]
    )
    daily["state"] = daily.apply(classify_regime_row, axis=1)
    daily_state = daily["state"].reindex(daily_index).ffill()
    return daily, daily_state


def portfolio_return_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: asset_returns[ASSETS]
            .mul(pd.Series(allocation))
            .sum(axis=1)
            for name, allocation in PORTFOLIO_LIBRARY.items()
        }
    )


def choose_portfolio(
    state: pd.Series,
    portfolio_returns: pd.DataFrame,
    target_state: str,
    start: str,
    end: str,
    risk_aversion: float = 4.0,
    prior_strength: int = 252,
) -> tuple[str, pd.Series]:
    sample_mask = state.eq(target_state) & (state.index >= start) & (state.index <= end)
    state_sample = portfolio_returns.loc[sample_mask].dropna(how="any")
    if len(state_sample) < 63:
        empty = pd.Series(np.nan, index=portfolio_returns.columns, name=target_state)
        return ECONOMIC_MAP[target_state], empty
    allowed = ALLOWED_PORTFOLIOS[target_state]
    state_sample = state_sample[allowed]
    unconditional = portfolio_returns.loc[start:end].dropna(how="any")
    unconditional = unconditional[allowed]
    state_weight = len(state_sample) / (len(state_sample) + prior_strength)
    shrunk_daily_mean = (
        state_weight * state_sample.mean()
        + (1.0 - state_weight) * unconditional.mean()
    )
    annualized_variance = state_sample.var(ddof=1) * 252.0
    utility = (
        shrunk_daily_mean * 252.0
        - 0.5 * risk_aversion * annualized_variance
    )
    utility.name = target_state
    return str(utility.idxmax()), utility


def development_selection(
    state: pd.Series,
    portfolio_returns: pd.DataFrame,
) -> tuple[dict[str, str], pd.DataFrame]:
    selected: dict[str, str] = {}
    rows: list[dict[str, float | int | str]] = []
    for target_state in STATE_ORDER:
        full_choice, utility = choose_portfolio(
            state,
            portfolio_returns,
            target_state,
            *DEVELOPMENT,
        )
        fold_choices: list[str] = []
        for year in range(2015, 2022):
            fold_state = state.copy()
            fold_state.loc[fold_state.index.year == year] = np.nan
            choice, _ = choose_portfolio(
                fold_state,
                portfolio_returns,
                target_state,
                *DEVELOPMENT,
            )
            fold_choices.append(choice)
        modal_choice, modal_count = Counter(fold_choices).most_common(1)[0]
        stable_share = modal_count / len(fold_choices)
        observations = int(
            (
                state.eq(target_state)
                & (state.index >= DEVELOPMENT[0])
                & (state.index <= DEVELOPMENT[1])
            ).sum()
        )
        accepted = (
            observations >= 126
            and stable_share >= 5.0 / 7.0
            and full_choice == modal_choice
        )
        final_choice = full_choice if accepted else ECONOMIC_MAP[target_state]
        selected[target_state] = final_choice
        rows.append(
            {
                "state": target_state,
                "development_observations": observations,
                "full_sample_choice": full_choice,
                "leave_one_year_out_mode": modal_choice,
                "selection_stability": stable_share,
                "selection_accepted": int(accepted),
                "final_choice": final_choice,
                "economic_fallback": ECONOMIC_MAP[target_state],
                "selected_utility": float(utility.get(full_choice, np.nan)),
            }
        )
    return selected, pd.DataFrame(rows).set_index("state")


def simulate_mapping(
    daily_state: pd.Series,
    asset_returns: pd.DataFrame,
    mapping: dict[str, str],
    cost_bps: float = 7.5,
    no_trade_turnover: float = 0.01,
    apply_production_risk_controls: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    rows: list[dict[str, float]] = []
    weight_rows: list[np.ndarray] = []
    equity = 1.0
    peak = 1.0
    for date in asset_returns.index:
        state = daily_state.get(date)
        turnover = 0.0
        trading_cost = 0.0
        if isinstance(state, str):
            portfolio_name = mapping[state]
            target = np.asarray(
                [PORTFOLIO_LIBRARY[portfolio_name][asset] for asset in ASSETS],
                dtype=float,
            )
            if apply_production_risk_controls:
                historical = asset_returns.loc[asset_returns.index < date]
                target, _ = apply_growth_stress_guard(
                    target,
                    historical,
                    ASSETS,
                    ["QQQ", "SEMIS"],
                    ASSETS.index("CASH"),
                    momentum_days=200,
                    volatility_days=20,
                    volatility_threshold=0.30,
                    growth_multiplier=0.0,
                    signal_asset="QQQ",
                )
                target, _, _ = apply_daily_growth_risk_reduction(
                    target,
                    historical,
                    ASSETS,
                    ["QQQ", "SEMIS"],
                    ASSETS.index("CASH"),
                    fast_days=20,
                    slow_days=60,
                    target_volatility=0.20,
                    volatility_estimator="jump_aware",
                    correlation_estimator="stress_max",
                )
            proposed_turnover = 0.5 * float(np.abs(target - current).sum())
            if proposed_turnover >= no_trade_turnover:
                turnover = proposed_turnover
                trading_cost = 2.0 * turnover * cost_bps / 10_000.0
                current = target
        day_return = asset_returns.loc[date, ASSETS].to_numpy(dtype=float)
        gross_return = float(current @ day_return)
        net_return = gross_return - trading_cost
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        rows.append(
            {
                "gross_return": gross_return,
                "trading_cost": trading_cost,
                "net_return": net_return,
                "turnover": turnover,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
            }
        )
        weight_rows.append(current.copy())
    return (
        pd.DataFrame(rows, index=asset_returns.index),
        pd.DataFrame(weight_rows, index=asset_returns.index, columns=ASSETS),
    )


def risk_budget_preserving_target(
    anchor: np.ndarray,
    regime_portfolio: np.ndarray,
    blend: float,
) -> np.ndarray:
    if not 0.0 <= blend <= 1.0:
        raise ValueError("Regime composition blend must be between zero and one")
    fixed = np.zeros(len(ASSETS), dtype=bool)
    fixed[ASSETS.index("CASH")] = True
    fixed[ASSETS.index("VIX_HEDGE")] = True
    allocatable = ~fixed
    allocatable_total = float(anchor[allocatable].sum())
    desired_total = float(regime_portfolio[allocatable].sum())
    if allocatable_total <= 1e-12 or desired_total <= 1e-12:
        return anchor.copy()
    anchor_unit = anchor[allocatable] / allocatable_total
    regime_unit = regime_portfolio[allocatable] / desired_total
    target = anchor.copy()
    target[allocatable] = allocatable_total * (
        (1.0 - blend) * anchor_unit + blend * regime_unit
    )
    return target


def simulate_anchor_tilt(
    daily_state: pd.Series,
    asset_returns: pd.DataFrame,
    anchor_weights: pd.DataFrame,
    production_daily: pd.DataFrame,
    mapping: dict[str, str],
    blend: float,
    cost_bps: float = 7.5,
    no_trade_turnover: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_index = (
        asset_returns.index
        .intersection(anchor_weights.index)
        .intersection(production_daily.index)
    )
    aligned_returns = asset_returns.reindex(evaluation_index)
    aligned_anchor = anchor_weights.reindex(
        index=evaluation_index,
        columns=ASSETS,
    )
    aligned_production = production_daily.reindex(evaluation_index)
    current_overlay = np.zeros(len(ASSETS), dtype=float)
    rows: list[dict[str, float]] = []
    weight_rows: list[np.ndarray] = []
    equity = 1.0
    peak = 1.0
    for date in evaluation_index:
        state = daily_state.get(date)
        anchor = aligned_anchor.loc[date].to_numpy(dtype=float)
        desired_overlay = np.zeros(len(ASSETS), dtype=float)
        if isinstance(state, str) and blend > 0.0:
            portfolio_name = mapping[state]
            if portfolio_name != PRODUCTION_ANCHOR:
                regime_portfolio = np.asarray(
                    [
                        PORTFOLIO_LIBRARY[portfolio_name][asset]
                        for asset in ASSETS
                    ],
                    dtype=float,
                )
                target = risk_budget_preserving_target(
                    anchor,
                    regime_portfolio,
                    blend,
                )
                desired_overlay = target - anchor
        proposed_turnover = 0.5 * float(
            np.abs(desired_overlay - current_overlay).sum()
        )
        if proposed_turnover >= no_trade_turnover:
            turnover = proposed_turnover
            trading_cost = 2.0 * turnover * cost_bps / 10_000.0
            current_overlay = desired_overlay
        else:
            turnover = 0.0
            trading_cost = 0.0
        target = anchor + current_overlay
        day_return = aligned_returns.loc[date, ASSETS].to_numpy(dtype=float)
        incremental_gross_return = float(current_overlay @ day_return)
        gross_return = (
            float(aligned_production.loc[date, "gross_return"])
            + incremental_gross_return
        )
        production_net_return = float(
            aligned_production.loc[date, "net_return"]
        )
        net_return = (
            production_net_return
            + incremental_gross_return
            - trading_cost
        )
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        rows.append(
            {
                "gross_return": gross_return,
                "production_net_return": production_net_return,
                "incremental_gross_return": incremental_gross_return,
                "incremental_trading_cost": trading_cost,
                "incremental_net_return": (
                    incremental_gross_return - trading_cost
                ),
                "net_return": net_return,
                "turnover": turnover,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
            }
        )
        weight_rows.append(target.copy())
    return (
        pd.DataFrame(rows, index=evaluation_index),
        pd.DataFrame(weight_rows, index=evaluation_index, columns=ASSETS),
    )


def incremental_contribution_by_state(
    daily_state: pd.Series,
    tilt_daily: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for period, (start, end) in {
        "development": DEVELOPMENT,
        "holdout": HOLDOUT,
    }.items():
        sample = tilt_daily.loc[start:end]
        sample_state = daily_state.reindex(sample.index)
        total_days = len(sample)
        for target_state in STATE_ORDER:
            mask = sample_state.eq(target_state)
            state_sample = sample.loc[mask]
            incremental_net = state_sample["incremental_net_return"]
            rows.append(
                {
                    "period": period,
                    "state": target_state,
                    "days": int(mask.sum()),
                    "incremental_gross_sum": float(
                        state_sample["incremental_gross_return"].sum()
                    ),
                    "incremental_cost_sum": float(
                        state_sample["incremental_trading_cost"].sum()
                    ),
                    "incremental_net_sum": float(incremental_net.sum()),
                    "annualized_conditional_alpha": (
                        float(incremental_net.mean() * 252.0)
                        if len(incremental_net)
                        else np.nan
                    ),
                    "annualized_portfolio_contribution": (
                        float(incremental_net.sum() * 252.0 / total_days)
                        if total_days
                        else np.nan
                    ),
                    "positive_incremental_day_rate": (
                        float(incremental_net.gt(0.0).mean())
                        if len(incremental_net)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).set_index(["period", "state"])


def moving_block_bootstrap_alpha(
    incremental_returns: pd.Series,
    block_days: int = 20,
    replications: int = 5_000,
    seed: int = 20_260_724,
) -> pd.Series:
    values = incremental_returns.dropna().to_numpy(dtype=float)
    if len(values) < block_days:
        raise ValueError("Sample must contain at least one complete block")
    generator = np.random.default_rng(seed)
    block_starts = np.arange(len(values) - block_days + 1)
    blocks_per_sample = int(np.ceil(len(values) / block_days))
    annualized_alpha = np.empty(replications, dtype=float)
    for replication in range(replications):
        starts = generator.choice(
            block_starts,
            size=blocks_per_sample,
            replace=True,
        )
        sample = np.concatenate(
            [values[start : start + block_days] for start in starts]
        )[: len(values)]
        annualized_alpha[replication] = float(sample.mean() * 252.0)
    lower, median, upper = np.quantile(
        annualized_alpha,
        [0.025, 0.50, 0.975],
    )
    return pd.Series(
        {
            "annualized_arithmetic_alpha": float(values.mean() * 252.0),
            "bootstrap_median": float(median),
            "ci_2_5": float(lower),
            "ci_97_5": float(upper),
            "probability_alpha_positive": float(
                np.mean(annualized_alpha > 0.0)
            ),
            "block_days": block_days,
            "replications": replications,
        }
    )


def annual_return_comparison(tilt_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    sample = tilt_daily.loc[slice(*COMPLETE)]
    for year, annual in sample.groupby(sample.index.year):
        production_return = float(
            (1.0 + annual["production_net_return"]).prod() - 1.0
        )
        candidate_return = float(
            (1.0 + annual["net_return"]).prod() - 1.0
        )
        rows.append(
            {
                "year": int(year),
                "production_return": production_return,
                "candidate_return": candidate_return,
                "compound_difference": candidate_return - production_return,
                "incremental_net_sum": float(
                    annual["incremental_net_return"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def candidate_overlay_daily(
    daily_state: pd.Series,
    asset_returns: pd.DataFrame,
    anchor_weights: pd.DataFrame,
    target_state: str,
    portfolio_name: str,
    start: str,
    end: str,
    blend: float = 0.25,
    excluded_year: int | None = None,
    cost_bps: float = 7.5,
) -> pd.DataFrame:
    evaluation_index = (
        asset_returns.loc[start:end].index
        .intersection(anchor_weights.index)
    )
    returns = asset_returns.reindex(evaluation_index)[ASSETS]
    anchors = anchor_weights.reindex(
        index=evaluation_index,
        columns=ASSETS,
    )
    overlay = pd.DataFrame(0.0, index=evaluation_index, columns=ASSETS)
    active = daily_state.reindex(evaluation_index).eq(target_state)
    if excluded_year is not None:
        active &= evaluation_index.year != excluded_year
    if portfolio_name != PRODUCTION_ANCHOR and bool(active.any()):
        fixed_assets = ["CASH", "VIX_HEDGE"]
        allocatable_assets = [
            asset for asset in ASSETS if asset not in fixed_assets
        ]
        regime = pd.Series(PORTFOLIO_LIBRARY[portfolio_name])
        regime_unit = (
            regime[allocatable_assets]
            / regime[allocatable_assets].sum()
        )
        anchor_allocatable = anchors.loc[active, allocatable_assets]
        anchor_total = anchor_allocatable.sum(axis=1)
        overlay.loc[active, allocatable_assets] = blend * (
            anchor_total.to_numpy()[:, None] * regime_unit.to_numpy()[None, :]
            - anchor_allocatable.to_numpy()
        )
    overlay_change = overlay.diff()
    if len(overlay_change):
        overlay_change.iloc[0] = overlay.iloc[0]
    turnover = 0.5 * overlay_change.abs().sum(axis=1)
    incremental_cost = 2.0 * turnover * cost_bps / 10_000.0
    incremental_gross = (overlay * returns).sum(axis=1)
    return pd.DataFrame(
        {
            "incremental_gross_return": incremental_gross,
            "incremental_trading_cost": incremental_cost,
            "incremental_net_return": incremental_gross - incremental_cost,
            "turnover": turnover,
        },
        index=evaluation_index,
    )


def choose_overlay_portfolio(
    daily_state: pd.Series,
    asset_returns: pd.DataFrame,
    anchor_weights: pd.DataFrame,
    target_state: str,
    excluded_year: int | None = None,
    blend: float = 0.25,
    prior_strength: int = 252,
    risk_aversion: float = 2.0,
    training_period: tuple[str, str] = DEVELOPMENT,
) -> tuple[str, pd.Series]:
    state = daily_state.loc[slice(*training_period)]
    active = state.eq(target_state)
    if excluded_year is not None:
        active &= active.index.year != excluded_year
    observations = int(active.sum())
    candidates = [PRODUCTION_ANCHOR, *ALLOWED_PORTFOLIOS[target_state]]
    utilities: dict[str, float] = {}
    for candidate in candidates:
        candidate_daily = candidate_overlay_daily(
            daily_state,
            asset_returns,
            anchor_weights,
            target_state,
            candidate,
            *training_period,
            blend=blend,
            excluded_year=excluded_year,
        )
        shrinkage = observations / (observations + prior_strength)
        annualized_shrunk_alpha = 252.0 * (
            shrinkage * candidate_daily["incremental_gross_return"].mean()
            - candidate_daily["incremental_trading_cost"].mean()
        )
        annualized_variance = (
            candidate_daily["incremental_net_return"].var(ddof=1) * 252.0
        )
        utilities[candidate] = float(
            annualized_shrunk_alpha
            - 0.5 * risk_aversion * annualized_variance
        )
    utility = pd.Series(utilities, name=target_state)
    return str(utility.idxmax()), utility


def development_overlay_selection(
    daily_state: pd.Series,
    asset_returns: pd.DataFrame,
    anchor_weights: pd.DataFrame,
    blend: float = 0.25,
    training_period: tuple[str, str] = DEVELOPMENT,
    minimum_observations: int = 126,
    minimum_active_years: int = 4,
    stability_threshold: float = 0.70,
    positive_year_threshold: float = 0.60,
) -> tuple[dict[str, str], pd.DataFrame]:
    selected: dict[str, str] = {}
    rows: list[dict[str, float | int | str]] = []
    for target_state in STATE_ORDER:
        full_choice, utility = choose_overlay_portfolio(
            daily_state,
            asset_returns,
            anchor_weights,
            target_state,
            blend=blend,
            training_period=training_period,
        )
        training_years = range(
            pd.Timestamp(training_period[0]).year,
            pd.Timestamp(training_period[1]).year + 1,
        )
        fold_choices = [
            choose_overlay_portfolio(
                daily_state,
                asset_returns,
                anchor_weights,
                target_state,
                excluded_year=year,
                blend=blend,
                training_period=training_period,
            )[0]
            for year in training_years
        ]
        modal_choice, modal_count = Counter(fold_choices).most_common(1)[0]
        stable_share = modal_count / len(fold_choices)
        development_state = daily_state.loc[slice(*training_period)]
        state_mask = development_state.eq(target_state)
        observations = int(state_mask.sum())
        active_years = sorted(
            set(development_state.index[state_mask].year.tolist())
        )
        chosen_daily = candidate_overlay_daily(
            daily_state,
            asset_returns,
            anchor_weights,
            target_state,
            full_choice,
            *training_period,
            blend=blend,
        )
        annual_net = chosen_daily["incremental_net_return"].groupby(
            chosen_daily.index.year
        ).sum()
        positive_year_share = (
            float(annual_net.reindex(active_years).gt(0.0).mean())
            if active_years
            else 0.0
        )
        accepted = (
            observations >= minimum_observations
            and len(active_years) >= minimum_active_years
            and stable_share >= stability_threshold
            and full_choice == modal_choice
            and full_choice != PRODUCTION_ANCHOR
            and positive_year_share >= positive_year_threshold
            and float(utility[full_choice]) > 0.0
        )
        final_choice = full_choice if accepted else PRODUCTION_ANCHOR
        selected[target_state] = final_choice
        rows.append(
            {
                "state": target_state,
                "development_observations": observations,
                "active_years": len(active_years),
                "full_sample_choice": full_choice,
                "leave_one_year_out_mode": modal_choice,
                "selection_stability": stable_share,
                "positive_year_share": positive_year_share,
                "selection_accepted": int(accepted),
                "final_choice": final_choice,
                "selected_utility": float(utility[full_choice]),
            }
        )
    return selected, pd.DataFrame(rows).set_index("state")


def period_metrics(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    periods = {
        "development_2015_2021": DEVELOPMENT,
        "holdout_2022_2025": HOLDOUT,
        "complete_2015_2025": COMPLETE,
    }
    rows: list[dict[str, float | str]] = []
    for strategy, returns in strategies.items():
        for period, (start, end) in periods.items():
            rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    **performance_metrics(returns.loc[start:end]),
                }
            )
    return pd.DataFrame(rows).set_index(["strategy", "period"])


def state_diagnostics(
    state: pd.Series,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    forward_21 = (
        (1.0 + asset_returns[ASSETS])
        .rolling(21)
        .apply(np.prod, raw=True)
        .shift(-20)
        - 1.0
    )
    rows: list[dict[str, float | int | str]] = []
    for period, (start, end) in {
        "development": DEVELOPMENT,
        "holdout": HOLDOUT,
    }.items():
        sample_state = state.loc[start:end]
        changes = sample_state.ne(sample_state.shift())
        episode = changes.cumsum()
        lengths = sample_state.groupby(episode).size()
        for target_state in STATE_ORDER:
            mask = sample_state.eq(target_state)
            state_episodes = episode.loc[mask].nunique()
            state_lengths = [
                int(length)
                for episode_id, length in lengths.items()
                if bool((episode.loc[mask] == episode_id).any())
            ]
            row: dict[str, float | int | str] = {
                "period": period,
                "state": target_state,
                "days": int(mask.sum()),
                "share": float(mask.mean()),
                "episodes": int(state_episodes),
                "median_episode_days": (
                    float(np.median(state_lengths)) if state_lengths else np.nan
                ),
            }
            for asset in ("QQQ", "SEMIS", "BOND", "GOLD", "CASH"):
                values = forward_21.loc[sample_state.index, asset].loc[mask].dropna()
                row[f"{asset}_forward21_mean"] = float(values.mean())
                row[f"{asset}_forward21_median"] = float(values.median())
            rows.append(row)
    return pd.DataFrame(rows).set_index(["period", "state"])


def write_decision(
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
    overlay_selection: pd.DataFrame,
    diagnostics: pd.DataFrame,
    alpha_uncertainty: pd.DataFrame,
    selected_blend: float,
    overlay_selected_blend: float,
) -> None:
    baseline = metrics.loc[("production", "holdout_2022_2025")]
    learned = metrics.loc[("development_selected", "holdout_2022_2025")]
    economic = metrics.loc[("economic_map", "holdout_2022_2025")]
    tilted = metrics.loc[("risk_budget_tilt", "holdout_2022_2025")]
    overlay_tilted = metrics.loc[
        ("overlay_selected_tilt", "holdout_2022_2025")
    ]
    accepted_states = int(selection["selection_accepted"].sum())
    overlay_accepted_states = int(
        overlay_selection["selection_accepted"].sum()
    )
    sparse_states = diagnostics.loc["development", "days"].lt(126).sum()
    holdout_uncertainty = alpha_uncertainty.loc["holdout"]
    verdict = (
        "进入第二阶段滚动/纸面验证；不升级生产"
        if (
            overlay_tilted["cagr"] > baseline["cagr"]
            and overlay_tilted["max_drawdown"]
            >= baseline["max_drawdown"] - 0.01
            and overlay_tilted["sharpe"] > baseline["sharpe"]
        )
        else "不进入生产引擎；保留状态诊断，重新设计组合映射"
    )
    report = f"""# 分层 Regime × Portfolio 第一阶段研究

## 决策

**{verdict}**

本阶段只使用现有因果信号进行独立模拟，没有改动生产策略。开发期为
2015-2021，评估期为 2022-2025。状态分类参数与组合库在第一次查看评估期前
固定，但 overlay 选择器是在诊断早期失败后新增，因此 2022-2025 已不再是
纯净留出集，结果只能用于筛选，不能视为最终样本外确认。

## 留出期结果

| 策略 | CAGR | 波动 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|
| 生产 Step2 | {baseline['cagr']:.3%} | {baseline['annual_volatility']:.3%} | {baseline['sharpe']:.3f} | {baseline['max_drawdown']:.3%} |
| 经济先验映射 | {economic['cagr']:.3%} | {economic['annual_volatility']:.3%} | {economic['sharpe']:.3f} | {economic['max_drawdown']:.3%} |
| 开发期选择映射 | {learned['cagr']:.3%} | {learned['annual_volatility']:.3%} | {learned['sharpe']:.3f} | {learned['max_drawdown']:.3%} |
| 风险预算内倾斜 | {tilted['cagr']:.3%} | {tilted['annual_volatility']:.3%} | {tilted['sharpe']:.3f} | {tilted['max_drawdown']:.3%} |
| Overlay 目标选择 | {overlay_tilted['cagr']:.3%} | {overlay_tilted['annual_volatility']:.3%} | {overlay_tilted['sharpe']:.3f} | {overlay_tilted['max_drawdown']:.3%} |

Overlay 在评估期的年化算术增量为
{holdout_uncertainty['annualized_arithmetic_alpha']:.3%}；20 日移动区块
bootstrap 的 95% 区间为 {holdout_uncertainty['ci_2_5']:.3%} 至
{holdout_uncertainty['ci_97_5']:.3%}，增量为正的概率为
{holdout_uncertainty['probability_alpha_positive']:.1%}。区间跨越零，
尚无统计确认。

## 状态与选择质量

- {len(STATE_ORDER)} 个状态中，{accepted_states} 个通过了静态组合的“至少
  126 个开发期交易日、留一年度
  选择稳定率至少 5/7、全样本与众数一致”的门槛。
- 以生产策略为锚、直接优化增量收益后，只有 {overlay_accepted_states} 个状态
  同时通过样本数、至少四个活跃年份、留一年度稳定和多数年份扣费后为正的门槛。
- {int(sparse_states)} 个状态开发期样本少于 126 天，必须回退到经济先验。
- VIX 期限结构 backwardation 可在实现波动确认前触发 `crisis_decline`；
  高波动反弹单列为 `crisis_recovery`，避免用同一组合处理下跌与反弹。
- 经济映射与开发期选择映射都继承生产的 200 日趋势/30% 波动压力保护，
  以及 20/60 日 jump-aware 日频成长风险缩放。
- 完整账户静态映射虽复用了成长风险缩放，但没有复现生产引擎的 ensemble
  净额与条件对冲，因此只用于说明全账户映射为何失败。
- 风险预算内倾斜保留生产策略每日现金/gross，并以开发期选定的
  {selected_blend:.1%} 幅度调整非现金资产组成。
- Overlay 目标选择直接以生产逐日净收益为基线，固定现金和 VIX 对冲，仅对
  其余资产做 {overlay_selected_blend:.1%} 的增量配置，并单独扣除额外换手成本。

## 方法限制

附件强调的 0DTE dealer gamma、杠杆 ETF 再平衡和风险平价去杠杆机制无法由
当前日频 ETF/VIX 数据直接识别。本轮仅使用可实时、可复现的代理：
20 日实现波动、60 日 bipower 波动、VIX/VIX3M、63/252 日成长趋势与
126 日债券/黄金趋势。未使用不可稳定获得的 dealer gamma 或流量数据。
"""
    (DESTINATION / "decision.md").write_text(report, encoding="utf-8")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PRICE_PATH, index_col=0, parse_dates=True)
    asset_returns = prices[ASSETS].pct_change(fill_method=None).dropna(how="any")
    schedule = load_regime_schedule()
    signals = build_causal_signals(prices)
    scheduled_states, daily_state = build_daily_states(
        schedule,
        signals,
        asset_returns.index,
    )
    daily_state = daily_state.loc[asset_returns.index]
    portfolio_returns = portfolio_return_matrix(asset_returns)
    learned_map, selection = development_selection(daily_state, portfolio_returns)

    economic_daily, economic_weights = simulate_mapping(
        daily_state,
        asset_returns,
        ECONOMIC_MAP,
    )
    learned_daily, learned_weights = simulate_mapping(
        daily_state,
        asset_returns,
        learned_map,
    )
    production_daily = pd.read_csv(
        PRODUCTION / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    production_returns = production_daily["net_return"].reindex(
        asset_returns.index
    ).dropna()
    anchor_weights = pd.read_csv(
        PRODUCTION / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    overlay_map, overlay_selection = development_overlay_selection(
        daily_state,
        asset_returns,
        anchor_weights,
    )
    tilt_candidates: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
    development_rows: list[dict[str, float]] = []
    production_development = performance_metrics(
        production_returns.loc[slice(*DEVELOPMENT)]
    )
    for blend in (0.125, 0.25, 0.50):
        tilt_daily, tilt_weights = simulate_anchor_tilt(
            daily_state,
            asset_returns,
            anchor_weights,
            production_daily,
            learned_map,
            blend,
        )
        tilt_candidates[blend] = (tilt_daily, tilt_weights)
        development_metric = performance_metrics(
            tilt_daily.loc[slice(*DEVELOPMENT), "net_return"]
        )
        development_rows.append(
            {
                "blend": blend,
                **development_metric,
                "drawdown_gate": int(
                    development_metric["max_drawdown"]
                    >= production_development["max_drawdown"] - 0.01
                ),
            }
        )
    blend_selection = pd.DataFrame(development_rows).set_index("blend")
    eligible_blends = blend_selection[blend_selection["drawdown_gate"].eq(1)]
    selected_blend = float(
        (
            eligible_blends["cagr"].idxmax()
            if not eligible_blends.empty
            else blend_selection["sharpe"].idxmax()
        )
    )
    tilt_daily, tilt_weights = tilt_candidates[selected_blend]
    overlay_tilt_candidates: dict[
        float, tuple[pd.DataFrame, pd.DataFrame]
    ] = {}
    overlay_development_rows: list[dict[str, float]] = []
    for blend in (0.125, 0.25, 0.50):
        candidate_daily, candidate_weights = simulate_anchor_tilt(
            daily_state,
            asset_returns,
            anchor_weights,
            production_daily,
            overlay_map,
            blend,
        )
        overlay_tilt_candidates[blend] = (
            candidate_daily,
            candidate_weights,
        )
        candidate_metric = performance_metrics(
            candidate_daily.loc[slice(*DEVELOPMENT), "net_return"]
        )
        overlay_development_rows.append(
            {
                "blend": blend,
                **candidate_metric,
                "drawdown_gate": int(
                    candidate_metric["max_drawdown"]
                    >= production_development["max_drawdown"] - 0.01
                ),
            }
        )
    overlay_blend_selection = pd.DataFrame(
        overlay_development_rows
    ).set_index("blend")
    eligible_overlay_blends = overlay_blend_selection[
        overlay_blend_selection["drawdown_gate"].eq(1)
    ]
    overlay_selected_blend = float(
        (
            eligible_overlay_blends["cagr"].idxmax()
            if not eligible_overlay_blends.empty
            else overlay_blend_selection["sharpe"].idxmax()
        )
    )
    overlay_tilt_daily, overlay_tilt_weights = overlay_tilt_candidates[
        overlay_selected_blend
    ]
    shared_index = (
        production_returns.index
        .intersection(economic_daily.index)
        .intersection(learned_daily.index)
    )
    metrics = period_metrics(
        {
            "production": production_returns.reindex(shared_index),
            "economic_map": economic_daily.loc[shared_index, "net_return"],
            "development_selected": learned_daily.loc[
                shared_index, "net_return"
            ],
            "risk_budget_tilt": tilt_daily.loc[shared_index, "net_return"],
            "overlay_selected_tilt": overlay_tilt_daily.loc[
                shared_index, "net_return"
            ],
        }
    )
    diagnostics = state_diagnostics(daily_state, asset_returns)
    incremental_contribution = incremental_contribution_by_state(
        daily_state,
        tilt_daily,
    )
    overlay_incremental_contribution = incremental_contribution_by_state(
        daily_state,
        overlay_tilt_daily,
    )
    alpha_uncertainty = pd.DataFrame(
        {
            "development": moving_block_bootstrap_alpha(
                overlay_tilt_daily.loc[
                    slice(*DEVELOPMENT),
                    "incremental_net_return",
                ]
            ),
            "holdout": moving_block_bootstrap_alpha(
                overlay_tilt_daily.loc[
                    slice(*HOLDOUT),
                    "incremental_net_return",
                ]
            ),
            "complete": moving_block_bootstrap_alpha(
                overlay_tilt_daily.loc[
                    slice(*COMPLETE),
                    "incremental_net_return",
                ]
            ),
        }
    ).T
    annual_comparison = annual_return_comparison(overlay_tilt_daily)

    scheduled_states.to_csv(DESTINATION / "scheduled_states.csv")
    daily_state.rename("state").to_csv(DESTINATION / "daily_states.csv")
    selection.to_csv(DESTINATION / "portfolio_selection.csv")
    overlay_selection.to_csv(DESTINATION / "overlay_portfolio_selection.csv")
    metrics.to_csv(DESTINATION / "metrics_by_period.csv")
    diagnostics.to_csv(DESTINATION / "state_diagnostics.csv")
    economic_daily.to_csv(DESTINATION / "economic_map_daily.csv")
    economic_weights.to_csv(DESTINATION / "economic_map_weights.csv")
    learned_daily.to_csv(DESTINATION / "development_selected_daily.csv")
    learned_weights.to_csv(DESTINATION / "development_selected_weights.csv")
    tilt_daily.to_csv(DESTINATION / "risk_budget_tilt_daily.csv")
    tilt_weights.to_csv(DESTINATION / "risk_budget_tilt_weights.csv")
    blend_selection.to_csv(DESTINATION / "blend_selection.csv")
    overlay_tilt_daily.to_csv(DESTINATION / "overlay_selected_tilt_daily.csv")
    overlay_tilt_weights.to_csv(
        DESTINATION / "overlay_selected_tilt_weights.csv"
    )
    overlay_blend_selection.to_csv(
        DESTINATION / "overlay_blend_selection.csv"
    )
    incremental_contribution.to_csv(
        DESTINATION / "incremental_contribution_by_state.csv"
    )
    overlay_incremental_contribution.to_csv(
        DESTINATION / "overlay_incremental_contribution_by_state.csv"
    )
    alpha_uncertainty.to_csv(DESTINATION / "alpha_uncertainty.csv")
    annual_comparison.to_csv(DESTINATION / "annual_return_comparison.csv")
    write_decision(
        metrics,
        selection,
        overlay_selection,
        diagnostics,
        alpha_uncertainty,
        selected_blend,
        overlay_selected_blend,
    )

    print(metrics[["cagr", "annual_volatility", "sharpe", "max_drawdown"]].round(6))
    print("\nPortfolio selection:")
    print(selection.to_string())
    print("\nOverlay portfolio selection:")
    print(overlay_selection.to_string())
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()

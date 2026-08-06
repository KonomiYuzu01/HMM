from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import warnings

# Avoid noisy physical-core probing in restricted macOS execution environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml


DEFAULT_CONFIG = Path("config/research_sector_hmm_shadow.yaml")
DEFAULT_PRICES = Path("data/sector_hmm_shadow_prices.csv")
DEFAULT_MACRO = Path("data/sector_hmm_shadow_macro.csv")
DEFAULT_OUTPUT = Path("output/sector_hmm_shadow")


@dataclass
class SignalResult:
    weights: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path} contains invalid dates")
    return frame


def complete_monthly_prices(
    daily_prices: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if daily_prices.empty:
        raise ValueError("Daily price data is empty")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if as_of is None else as_of
    monthly = daily_prices.resample("ME").last()
    final_daily = daily_prices.index.max()
    if (final_daily.year, final_daily.month) == (now.year, now.month):
        monthly = monthly.iloc[:-1]
    return monthly.dropna(how="all")


def build_monthly_features(
    monthly_prices: pd.DataFrame,
    macro_daily: pd.DataFrame,
    *,
    cpi_release_lag_months: int,
) -> pd.DataFrame:
    required_prices = {"SPY", "HYG", "IEF"}
    required_macro = {"DGS10", "DFII10", "CPIAUCSL", "T10Y2Y", "DTWEXBGS"}
    if missing := sorted(required_prices.difference(monthly_prices.columns)):
        raise ValueError(f"Missing feature prices: {missing}")
    if missing := sorted(required_macro.difference(macro_daily.columns)):
        raise ValueError(f"Missing macro series: {missing}")
    macro = macro_daily.resample("ME").last().ffill()
    cpi_known = macro["CPIAUCSL"].shift(cpi_release_lag_months)
    features = pd.DataFrame(index=monthly_prices.index)
    features["nominal_10y_change_3m"] = macro["DGS10"].reindex(
        features.index
    ).diff(3)
    features["real_10y_change_3m"] = macro["DFII10"].reindex(
        features.index
    ).diff(3)
    features["cpi_yoy"] = cpi_known.reindex(features.index).pct_change(
        12, fill_method=None
    )
    features["cpi_direction_3m"] = cpi_known.reindex(features.index).diff(3)
    features["curve_10y2y"] = macro["T10Y2Y"].reindex(features.index)
    features["credit_hyg_ief_3m"] = np.log(
        monthly_prices["HYG"] / monthly_prices["IEF"]
    ).diff(3)
    features["dollar_change_3m"] = np.log(
        macro["DTWEXBGS"].reindex(features.index)
    ).diff(3)
    features["risk_cycle_spy_3m"] = np.log(monthly_prices["SPY"]).diff(3)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def eligible_assets(
    monthly_prices: pd.DataFrame,
    sectors: list[str],
    signal_date: pd.Timestamp,
    minimum_history_months: int,
) -> list[str]:
    history = monthly_prices.loc[:signal_date, sectors]
    return [
        sector
        for sector in sectors
        if int(history[sector].notna().sum()) >= minimum_history_months
        and pd.notna(history[sector].iloc[-1])
    ]


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    clean = values.dropna().astype(float)
    if clean.empty:
        return clean
    std = clean.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=clean.index)
    return (clean - clean.mean()) / std


def top_k_equal_weight(scores: pd.Series, top_k: int) -> pd.Series:
    clean = scores.dropna().sort_values(ascending=False, kind="mergesort")
    if len(clean) < top_k:
        raise ValueError(f"Need {top_k} eligible assets, found {len(clean)}")
    selected = clean.iloc[:top_k].index
    result = pd.Series(0.0, index=scores.index, dtype=float)
    result.loc[selected] = 1.0 / top_k
    return result


def _fit_hmm_models(
    features: pd.DataFrame,
    *,
    states: int,
    seeds: list[int],
    iterations: int,
) -> list[tuple[StandardScaler, GaussianHMM]]:
    scaler = StandardScaler().fit(features)
    observations = scaler.transform(features)
    models: list[tuple[StandardScaler, GaussianHMM]] = []
    for seed in seeds:
        model = GaussianHMM(
            n_components=states,
            covariance_type="diag",
            n_iter=iterations,
            min_covar=1e-5,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(observations)
        models.append((scaler, model))
    return models


def _predict_hmm_states(
    models: list[tuple[StandardScaler, GaussianHMM]],
    features: pd.DataFrame,
) -> list[np.ndarray]:
    return [
        model.predict(scaler.transform(features))
        for scaler, model in models
    ]


def state_conditioned_scores(
    feature_history: pd.DataFrame,
    forward_excess_returns: pd.DataFrame,
    state_paths: list[np.ndarray],
    assets: list[str],
    *,
    minimum_observations: int,
    shrinkage_observations: float,
) -> pd.Series:
    aligned = forward_excess_returns.reindex(feature_history.index)
    scores: list[pd.Series] = []
    for path in state_paths:
        current_state = int(path[-1])
        state_mask = pd.Series(path == current_state, index=feature_history.index)
        seed_scores: dict[str, float] = {}
        for asset in assets:
            outcomes = aligned[asset].iloc[:-1]
            mask = state_mask.iloc[:-1] & outcomes.notna()
            state_sample = outcomes.loc[mask]
            all_sample = outcomes.dropna()
            if len(state_sample) < minimum_observations or all_sample.empty:
                seed_scores[asset] = np.nan
                continue
            weight = len(state_sample) / (
                len(state_sample) + shrinkage_observations
            )
            seed_scores[asset] = float(
                weight * state_sample.mean()
                + (1.0 - weight) * all_sample.mean()
            )
        scores.append(pd.Series(seed_scores, dtype=float))
    return pd.concat(scores, axis=1).mean(axis=1, skipna=True)


def build_signal_weights(
    monthly_prices: pd.DataFrame,
    features: pd.DataFrame,
    config: dict[str, object],
) -> SignalResult:
    assets = config["assets"]
    signal = config["signal"]
    sectors = [str(value) for value in assets["sectors"]]
    benchmark = str(assets["benchmark"])
    returns = monthly_prices.pct_change(fill_method=None)
    forward_excess = returns[sectors].sub(returns[benchmark], axis=0).shift(-1)
    lookback = int(signal["momentum_lookback_months"])
    skip = int(signal["momentum_skip_months"])
    momentum = monthly_prices[sectors].shift(skip).div(
        monthly_prices[sectors].shift(lookback)
    ) - 1.0
    minimum_training = int(signal["hmm_training_months"])
    refit_months = int(signal["hmm_refit_months"])
    top_k = int(signal["top_k"])
    minimum_history = int(signal["minimum_asset_history_months"])
    weights = {
        name: pd.DataFrame(0.0, index=features.index, columns=sectors)
        for name in ("sector_equal", "momentum_top3", "hmm_top3", "hmm_momentum")
    }
    diagnostics: list[dict[str, object]] = []
    hmm_models: list[tuple[StandardScaler, GaussianHMM]] | None = None
    fitted_length = 0
    for signal_number, signal_date in enumerate(features.index):
        history = features.loc[:signal_date]
        if len(history) < minimum_training:
            continue
        available = eligible_assets(
            monthly_prices,
            sectors,
            signal_date,
            minimum_history,
        )
        if len(available) < top_k:
            continue
        should_refit = (
            hmm_models is None
            or len(history) - fitted_length >= refit_months
        )
        if should_refit:
            hmm_models = _fit_hmm_models(
                history,
                states=int(signal["hmm_states"]),
                seeds=[int(value) for value in signal["hmm_seeds"]],
                iterations=int(signal["hmm_iterations"]),
            )
            fitted_length = len(history)
        assert hmm_models is not None
        state_paths = _predict_hmm_states(hmm_models, history)
        hmm_score = state_conditioned_scores(
            history,
            forward_excess,
            state_paths,
            available,
            minimum_observations=int(signal["state_minimum_observations"]),
            shrinkage_observations=float(signal["shrinkage_observations"]),
        )
        momentum_score = momentum.loc[signal_date, available].dropna()
        common = sorted(set(hmm_score.dropna().index) & set(momentum_score.index))
        if len(common) < top_k:
            continue
        hmm_z = cross_sectional_zscore(hmm_score.loc[common])
        momentum_z = cross_sectional_zscore(momentum_score.loc[common])
        blend_weight = float(signal["blend_hmm_weight"])
        blend = blend_weight * hmm_z + (1.0 - blend_weight) * momentum_z
        weights["sector_equal"].loc[signal_date, available] = (
            1.0 / len(available)
        )
        weights["momentum_top3"].loc[signal_date, common] = top_k_equal_weight(
            momentum_z, top_k
        )
        weights["hmm_top3"].loc[signal_date, common] = top_k_equal_weight(
            hmm_z, top_k
        )
        weights["hmm_momentum"].loc[signal_date, common] = top_k_equal_weight(
            blend, top_k
        )
        diagnostics.append(
            {
                "signal_date": signal_date,
                "eligible_assets": ",".join(available),
                "eligible_count": len(available),
                "momentum_selection": ",".join(
                    momentum_z.nlargest(top_k).index
                ),
                "hmm_selection": ",".join(hmm_z.nlargest(top_k).index),
                "blend_selection": ",".join(blend.nlargest(top_k).index),
                "feature_observations": len(history),
                "signal_number": signal_number,
            }
        )
    active_dates = pd.DatetimeIndex(
        sorted(
            set().union(
                *[
                    set(frame.index[frame.sum(axis=1) > 0])
                    for frame in weights.values()
                ]
            )
        )
    )
    trimmed = {name: frame.loc[active_dates] for name, frame in weights.items()}
    return SignalResult(
        weights=trimmed,
        diagnostics=pd.DataFrame(diagnostics).set_index("signal_date"),
    )


def one_way_turnover(weights: pd.DataFrame) -> pd.Series:
    previous = weights.shift(1).fillna(0.0)
    turnover = 0.5 * weights.sub(previous).abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    return turnover


def simulate_monthly(
    weights_at_signal: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    *,
    one_way_cost_bps: float,
) -> pd.DataFrame:
    asset_returns = monthly_prices[weights_at_signal.columns].pct_change(
        fill_method=None
    )
    realized_weights = weights_at_signal.copy()
    realized_weights.index = realized_weights.index + pd.offsets.MonthEnd(1)
    realized_weights = realized_weights[
        ~realized_weights.index.duplicated(keep="last")
    ]
    aligned_returns = asset_returns.reindex(realized_weights.index)
    gross = (realized_weights * aligned_returns).sum(axis=1, min_count=1)
    turnover = one_way_turnover(realized_weights)
    cost = turnover * one_way_cost_bps / 10_000.0
    return pd.DataFrame(
        {
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
        }
    ).dropna(subset=["gross_return"])


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("Cannot calculate metrics from empty returns")
    equity = (1.0 + clean).cumprod()
    years = len(clean) / 12.0
    volatility = float(clean.std(ddof=1) * np.sqrt(12.0))
    drawdown = equity / equity.cummax() - 1.0
    return {
        "months": int(len(clean)),
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "annual_volatility": volatility,
        "sharpe": (
            float(clean.mean() / clean.std(ddof=1) * np.sqrt(12.0))
            if clean.std(ddof=1) > 0
            else np.nan
        ),
        "max_drawdown": float(drawdown.min()),
    }


def circular_block_bootstrap(
    differences: pd.Series,
    *,
    replications: int,
    block_months: int,
    random_seed: int,
) -> dict[str, float]:
    values = differences.dropna().to_numpy(dtype=float)
    if len(values) < block_months:
        raise ValueError("Bootstrap sample is shorter than one block")
    rng = np.random.default_rng(random_seed)
    means = np.empty(replications)
    block_count = int(np.ceil(len(values) / block_months))
    offsets = np.arange(block_months)
    for replication in range(replications):
        starts = rng.integers(0, len(values), size=block_count)
        indices = (starts[:, None] + offsets[None, :]) % len(values)
        means[replication] = values[indices.ravel()[: len(values)]].mean()
    return {
        "annualized_mean_difference": float(values.mean() * 12.0),
        "probability_mean_not_positive": float(np.mean(means <= 0.0)),
        "bootstrap_p05_annualized": float(np.quantile(means, 0.05) * 12.0),
        "bootstrap_p50_annualized": float(np.quantile(means, 0.50) * 12.0),
        "bootstrap_p95_annualized": float(np.quantile(means, 0.95) * 12.0),
    }


def _strategy_returns(
    signals: SignalResult,
    prices: pd.DataFrame,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    simulations = {
        name: simulate_monthly(frame, prices, one_way_cost_bps=cost_bps)
        for name, frame in signals.weights.items()
    }
    combined = pd.DataFrame(
        {name: frame["net_return"] for name, frame in simulations.items()}
    ).dropna()
    spy = prices["SPY"].pct_change(fill_method=None).reindex(combined.index)
    combined["SPY"] = spy
    return combined.dropna(), simulations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an isolated causal sector-HMM shadow strategy"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--macro", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prices = complete_monthly_prices(load_frame(args.prices))
    features = build_monthly_features(
        prices,
        load_frame(args.macro),
        cpi_release_lag_months=int(
            config["macro"]["cpi_release_lag_months"]
        ),
    )
    signals = build_signal_weights(prices, features, config)
    normal_cost = float(config["implementation"]["one_way_cost_bps"])
    stressed_cost = float(
        config["implementation"]["stressed_one_way_cost_bps"]
    )
    returns, simulations = _strategy_returns(signals, prices, normal_cost)
    stressed_returns, _ = _strategy_returns(signals, prices, stressed_cost)

    periods = {
        "development": (
            returns.index.min().strftime("%Y-%m-%d"),
            str(config["research"]["development_end"]),
        ),
        "holdout_2022_2025": (
            str(config["research"]["holdout_start"]),
            str(config["research"]["holdout_end"]),
        ),
        "post_holdout_2026": ("2026-01-01", None),
        "full": (returns.index.min().strftime("%Y-%m-%d"), None),
    }
    metric_rows: list[dict[str, object]] = []
    for period, (start, end) in periods.items():
        for strategy in returns.columns:
            sample = returns.loc[start:end, strategy]
            if sample.empty:
                continue
            metric_rows.append(
                {
                    "period": period,
                    "strategy": strategy,
                    **performance_metrics(sample),
                }
            )
    metrics = pd.DataFrame(metric_rows).set_index(["period", "strategy"])

    holdout = returns.loc[
        str(config["research"]["holdout_start"]):
        str(config["research"]["holdout_end"])
    ]
    robustness_config = config["robustness"]
    bootstrap = {
        baseline: circular_block_bootstrap(
            holdout["hmm_momentum"] - holdout[baseline],
            replications=int(robustness_config["bootstrap_replications"]),
            block_months=int(robustness_config["bootstrap_block_months"]),
            random_seed=int(robustness_config["random_seed"]),
        )
        for baseline in ("sector_equal", "momentum_top3", "SPY")
    }
    stressed_holdout = stressed_returns.loc[
        str(config["research"]["holdout_start"]):
        str(config["research"]["holdout_end"])
    ]
    robustness = {
        "bootstrap": bootstrap,
        "holdout_stressed_cost_metrics": {
            strategy: performance_metrics(stressed_holdout[strategy])
            for strategy in stressed_holdout.columns
        },
    }

    latest_signal = signals.diagnostics.iloc[-1].to_dict()
    latest_date = signals.diagnostics.index[-1]
    latest_weights = {
        name: {
            asset: float(weight)
            for asset, weight in frame.loc[latest_date].items()
            if weight > 0
        }
        for name, frame in signals.weights.items()
    }
    holdout_metrics = metrics.loc["holdout_2022_2025"]
    candidate = holdout_metrics.loc["hmm_momentum"]
    momentum_baseline = holdout_metrics.loc["momentum_top3"]
    equal_baseline = holdout_metrics.loc["sector_equal"]
    passed_directional_checks = bool(
        candidate["cagr"] > momentum_baseline["cagr"]
        and candidate["cagr"] > equal_baseline["cagr"]
        and candidate["sharpe"] > momentum_baseline["sharpe"]
        and bootstrap["momentum_top3"]["probability_mean_not_positive"] < 0.10
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        "latest_complete_price_month": prices.index.max().date().isoformat(),
        "latest_signal_month": latest_date.date().isoformat(),
        "latest_signal": latest_signal,
        "latest_weights": latest_weights,
        "directional_research_checks_passed": passed_directional_checks,
        "decision": (
            "continue_shadow_research"
            if passed_directional_checks
            else "reject_current_specification"
        ),
        "qualification": (
            "Not production-qualified regardless of historical result; requires "
            "a frozen forward shadow period."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics_by_period.csv")
    returns.to_csv(args.output / "monthly_returns.csv")
    signals.diagnostics.to_csv(args.output / "signal_diagnostics.csv")
    pd.concat(signals.weights, names=["strategy", "signal_date"]).to_csv(
        args.output / "signal_weights.csv"
    )
    pd.DataFrame(
        {
            name: simulation["turnover"]
            for name, simulation in simulations.items()
        }
    ).to_csv(args.output / "monthly_turnover.csv")
    (args.output / "robustness.json").write_text(
        json.dumps(robustness, ensure_ascii=False, indent=2, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

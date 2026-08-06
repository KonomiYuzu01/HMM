from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regime_strategy.data import load_prices
from regime_strategy.ensemble_cli import _load_ensemble_config
from regime_strategy.features import build_causal_features


OUTPUT = Path("output/hmm_feature_geometry")
CONFIGS = {
    "normal": Path("config/research_hmm_gaussian_diagnostics.yaml"),
    "proxy": Path("config/research_hmm_gaussian_diagnostics_20y_proxy.yaml"),
}
LOOKBACK = 1_008


def correlation_geometry(correlation: np.ndarray) -> dict[str, float | int]:
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")
    eigenvalues = np.maximum(np.linalg.eigvalsh((matrix + matrix.T) / 2.0), 0.0)
    positive = eigenvalues[eigenvalues > 1e-10]
    if not len(positive):
        raise ValueError("Correlation matrix has no positive eigenvalues")
    mass = positive / positive.sum()
    effective_rank = float(np.exp(-(mass * np.log(mass)).sum()))
    participation_ratio = float(np.square(positive.sum()) / np.square(positive).sum())
    descending = np.sort(positive)[::-1]
    cumulative = np.cumsum(descending) / descending.sum()
    dimensions = matrix.shape[0]
    upper = np.abs(matrix[np.triu_indices(dimensions, k=1)])
    return {
        "dimensions": dimensions,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "components_80pct": int(np.searchsorted(cumulative, 0.8) + 1),
        "components_90pct": int(np.searchsorted(cumulative, 0.9) + 1),
        "positive_condition_number": float(positive.max() / positive.min()),
        "pairs_abs_correlation_ge_0_8": int(np.sum(upper >= 0.8)),
        "median_abs_correlation": float(np.median(upper)),
    }


def _feature_frame(config_path: Path) -> pd.DataFrame:
    ensemble = _load_ensemble_config(config_path)
    base = yaml.safe_load(Path(str(ensemble["base_config"])).read_text(encoding="utf-8"))
    for key, value in ensemble.get("data_overrides", {}).items():
        base["data"][str(key)] = deepcopy(value)
    prices = load_prices(base["data"], refresh=False)
    feature = base["features"]
    return build_causal_features(
        prices,
        list(base["data"]["regime_assets"]),
        int(feature["volatility_days"]),
        int(feature["momentum_days"]),
        list(feature.get("derived_signals", [])),
        int(base["backtest"]["annualization"]),
        list(feature.get("components", [])) or None,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    window_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for sample, config_path in CONFIGS.items():
        features = _feature_frame(config_path)
        annual_ends = features.groupby(features.index.year).tail(1).index
        endpoint_positions = features.index.get_indexer(annual_ends)
        endpoints = annual_ends[endpoint_positions >= LOOKBACK - 1]
        if features.index[-1] not in endpoints:
            endpoints = endpoints.append(pd.DatetimeIndex([features.index[-1]]))
        for endpoint in endpoints.unique().sort_values():
            position = features.index.get_loc(endpoint)
            if not isinstance(position, int) or position < LOOKBACK - 1:
                continue
            window = features.iloc[position - LOOKBACK + 1 : position + 1]
            correlation = window.corr()
            window_rows.append(
                {
                    "sample": sample,
                    "endpoint": endpoint.date().isoformat(),
                    **correlation_geometry(correlation.to_numpy()),
                }
            )
            for left_position, left in enumerate(correlation.columns):
                for right in correlation.columns[left_position + 1 :]:
                    pair_rows.append(
                        {
                            "sample": sample,
                            "endpoint": endpoint.date().isoformat(),
                            "left": left,
                            "right": right,
                            "abs_correlation": abs(float(correlation.loc[left, right])),
                        }
                    )

    windows = pd.DataFrame(window_rows)
    pairs = pd.DataFrame(pair_rows)
    pair_summary = (
        pairs.groupby(["sample", "left", "right"])["abs_correlation"]
        .agg(mean_abs_correlation="mean", maximum_abs_correlation="max", windows="size")
        .reset_index()
    )
    frequency = (
        pairs.assign(high=pairs["abs_correlation"].ge(0.8))
        .groupby(["sample", "left", "right"])["high"]
        .mean()
        .rename("fraction_ge_0_8")
        .reset_index()
    )
    pair_summary = pair_summary.merge(frequency, on=["sample", "left", "right"])
    pair_summary = pair_summary.sort_values(
        ["sample", "mean_abs_correlation"], ascending=[True, False]
    )
    windows.to_csv(OUTPUT / "window_geometry.csv", index=False)
    pair_summary.to_csv(OUTPUT / "pair_summary.csv", index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_diagnostic",
        "production_changed": False,
        "orders_generated": False,
        "samples": {
            sample: {
                "windows": int(len(frame)),
                "median_effective_rank": float(frame["effective_rank"].median()),
                "median_components_80pct": float(frame["components_80pct"].median()),
                "median_components_90pct": float(frame["components_90pct"].median()),
                "median_high_correlation_pairs": float(
                    frame["pairs_abs_correlation_ge_0_8"].median()
                ),
            }
            for sample, frame in windows.groupby("sample")
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(windows.round(4).to_string(index=False))
    print("\nTop persistent correlated pairs:")
    print(pair_summary.groupby("sample").head(10).round(4).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

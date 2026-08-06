from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.audit_hmm_feature_geometry import CONFIGS, LOOKBACK, _feature_frame


OUTPUT = Path("output/hmm_robust_scaler_feature_leverage")
STEP = 21


def relative_scale_factors(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Values must be a two-dimensional sample")
    standard = matrix.std(axis=0, ddof=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    robust = q75 - q25
    if np.any(standard <= 0.0) or np.any(robust <= 0.0):
        raise ValueError("Every feature must have positive standard and robust scale")
    raw = standard / robust
    return raw / np.median(raw)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for sample, config_path in CONFIGS.items():
        features = _feature_frame(config_path)
        endpoints = range(LOOKBACK - 1, len(features), STEP)
        for endpoint_position in endpoints:
            window = features.iloc[
                endpoint_position - LOOKBACK + 1 : endpoint_position + 1
            ]
            factors = relative_scale_factors(window.to_numpy(dtype=float))
            mean = window.mean().to_numpy(dtype=float)
            median = window.median().to_numpy(dtype=float)
            standard = window.std(ddof=0).to_numpy(dtype=float)
            for feature, factor, center_shift in zip(
                window.columns,
                factors,
                np.abs(mean - median) / standard,
                strict=True,
            ):
                rows.append(
                    {
                        "sample": sample,
                        "endpoint": window.index[-1].date().isoformat(),
                        "feature": feature,
                        "relative_scale_factor": float(factor),
                        "standardized_center_shift": float(center_shift),
                    }
                )
    details = pd.DataFrame(rows)
    feature_summary = (
        details.groupby(["sample", "feature"])
        .agg(
            windows=("relative_scale_factor", "size"),
            median_relative_scale_factor=("relative_scale_factor", "median"),
            p90_relative_scale_factor=("relative_scale_factor", lambda x: x.quantile(0.9)),
            maximum_relative_scale_factor=("relative_scale_factor", "max"),
            fraction_upweighted_25pct=("relative_scale_factor", lambda x: (x >= 1.25).mean()),
            median_standardized_center_shift=("standardized_center_shift", "median"),
        )
        .reset_index()
        .sort_values(
            ["sample", "median_relative_scale_factor"], ascending=[True, False]
        )
    )
    details.to_csv(OUTPUT / "window_feature_scales.csv", index=False)
    feature_summary.to_csv(OUTPUT / "feature_summary.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only_scaler_mechanism_diagnostic",
        "production_changed": False,
        "orders_generated": False,
        "lookback_days": LOOKBACK,
        "step_days": STEP,
        "samples": {
            sample: {
                "windows": int(frame["windows"].max()),
                "highest_median_relative_scale_features": frame.head(5)[
                    "feature"
                ].tolist(),
                "features_upweighted_25pct_in_majority_of_windows": frame.loc[
                    frame["fraction_upweighted_25pct"] > 0.5, "feature"
                ].tolist(),
            }
            for sample, frame in feature_summary.groupby("sample", sort=False)
        },
        "interpretation": (
            "Factors are standard-deviation scale divided by IQR scale, normalized "
            "within each window. Values above one receive more relative geometric "
            "weight under RobustScaler than under StandardScaler."
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(feature_summary.groupby("sample", sort=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

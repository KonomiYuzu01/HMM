from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output/student_t_hmm_validation")
SAMPLES = {
    "normal": {
        "gaussian": Path("output/research_hmm_gaussian_diagnostics"),
        "student_t": Path("output/research_hmm_student_t_df5"),
        "periods": {
            "development_2015_2021": ("2015-01-01", "2021-12-31"),
            "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
            "complete_2015_2025": ("2015-01-01", "2025-12-31"),
            "post_holdout_2026": ("2026-01-01", None),
        },
    },
    "proxy": {
        "gaussian": Path("output/research_hmm_gaussian_diagnostics_20y_proxy"),
        "student_t": Path("output/research_hmm_student_t_df5_20y_proxy"),
        "periods": {
            "early_2006_2014": ("2006-08-01", "2014-12-31"),
            "late_2015_2025": ("2015-01-01", "2025-12-31"),
            "complete_2006_2025": ("2006-08-01", "2025-12-31"),
            "post_holdout_2026": ("2026-01-01", None),
        },
    },
}


def stressed_return(frame: pd.DataFrame) -> pd.Series:
    """Double linear trading costs while leaving financing costs unchanged."""
    return frame["net_return"] - frame["trading_cost"]


def metrics_row(returns: pd.Series) -> dict[str, float]:
    metrics = performance_metrics(returns.dropna())
    return {name: float(value) for name, value in metrics.items()}


def member_diagnostic_rows(
    sample: str,
    gaussian_directory: Path,
    student_directory: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gaussian_path in sorted(
        (gaussian_directory / "members").glob("seed_*/regimes.csv")
    ):
        student_path = student_directory / "members" / gaussian_path.parent.name / "regimes.csv"
        gaussian = pd.read_csv(gaussian_path, index_col=0, parse_dates=True)
        student = pd.read_csv(student_path, index_col=0, parse_dates=True)
        index = gaussian.index.intersection(student.index)
        gaussian = gaussian.loc[index]
        student = student.loc[index]
        threshold = float(
            gaussian["one_step_predictive_log_likelihood"].quantile(0.05)
        )
        rows.append(
            {
                "sample": sample,
                "member": gaussian_path.parent.name,
                "gaussian_state_switches": int(
                    gaussian["paper_risk_on_candidate"].ne(
                        gaussian["paper_risk_on_candidate"].shift(1)
                    ).sum()
                ),
                "student_t_state_switches": int(
                    student["paper_risk_on_candidate"].ne(
                        student["paper_risk_on_candidate"].shift(1)
                    ).sum()
                ),
                "gaussian_mean_entropy": float(gaussian["state_entropy"].mean()),
                "student_t_mean_entropy": float(student["state_entropy"].mean()),
                "gaussian_extreme_loglik_count": int(
                    gaussian["one_step_predictive_log_likelihood"].lt(threshold).sum()
                ),
                "student_t_extreme_loglik_count_at_gaussian_threshold": int(
                    student["one_step_predictive_log_likelihood"].lt(threshold).sum()
                ),
                "gaussian_loglik_p05": threshold,
                "student_t_loglik_p05": float(
                    student["one_step_predictive_log_likelihood"].quantile(0.05)
                ),
            }
        )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for sample, definition in SAMPLES.items():
        gaussian_directory = definition["gaussian"]
        student_directory = definition["student_t"]
        assert isinstance(gaussian_directory, Path)
        assert isinstance(student_directory, Path)
        gaussian = pd.read_csv(
            gaussian_directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        student = pd.read_csv(
            student_directory / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
        index = gaussian.index.intersection(student.index)
        periods = definition["periods"]
        assert isinstance(periods, dict)
        for period, (start, end) in periods.items():
            selected = index[(index >= start) & ((index <= end) if end else True)]
            for scenario, transform in (
                ("current_cost", lambda frame: frame["net_return"]),
                ("double_trading_cost", stressed_return),
            ):
                gaussian_metrics = metrics_row(transform(gaussian.loc[selected]))
                student_metrics = metrics_row(transform(student.loc[selected]))
                metric_rows.append(
                    {
                        "sample": sample,
                        "period": period,
                        "scenario": scenario,
                        **{f"gaussian_{key}": value for key, value in gaussian_metrics.items()},
                        **{f"student_t_{key}": value for key, value in student_metrics.items()},
                        "cagr_delta": student_metrics["cagr"] - gaussian_metrics["cagr"],
                        "max_drawdown_delta": (
                            student_metrics["max_drawdown"]
                            - gaussian_metrics["max_drawdown"]
                        ),
                    }
                )
        diagnostic_rows.extend(
            member_diagnostic_rows(sample, gaussian_directory, student_directory)
        )

    metrics = pd.DataFrame(metric_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    diagnostics.to_csv(OUTPUT / "member_diagnostics.csv", index=False)

    def delta(sample: str, period: str, field: str, scenario: str = "current_cost") -> float:
        return float(
            metrics.loc[
                metrics["sample"].eq(sample)
                & metrics["period"].eq(period)
                & metrics["scenario"].eq(scenario),
                field,
            ].iloc[0]
        )

    gates = {
        "development_cagr_nonnegative": delta(
            "normal", "development_2015_2021", "cagr_delta"
        ) >= 0.0,
        "holdout_cagr_nonnegative": delta(
            "normal", "holdout_2022_2025", "cagr_delta"
        ) >= 0.0,
        "normal_drawdown_within_50bp": delta(
            "normal", "complete_2015_2025", "max_drawdown_delta"
        ) >= -0.005,
        "proxy_cagr_nonnegative": delta(
            "proxy", "complete_2006_2025", "cagr_delta"
        ) >= 0.0,
        "proxy_drawdown_within_50bp": delta(
            "proxy", "complete_2006_2025", "max_drawdown_delta"
        ) >= -0.005,
        "double_cost_cagr_nonnegative": delta(
            "normal",
            "complete_2015_2025",
            "cagr_delta",
            "double_trading_cost",
        ) >= 0.0,
        "fewer_extreme_loglik_observations": bool(
            diagnostics["student_t_extreme_loglik_count_at_gaussian_threshold"].sum()
            < diagnostics["gaussian_extreme_loglik_count"].sum()
        ),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        "degrees_of_freedom": 5.0,
        "gates": gates,
        "research_pass": bool(all(gates.values())),
        "decision": (
            "continue_forward_shadow" if all(gates.values()) else "reject_current_specification"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


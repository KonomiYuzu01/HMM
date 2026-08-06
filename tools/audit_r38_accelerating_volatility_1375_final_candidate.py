from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r12_volatility_managed_risk import (
    robustness_rows,
    simulate_fixed_r11,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30
import tools.evaluate_r38_accelerating_volatility_capacity_fill_1375 as candidate
import tools.evaluate_r38_convex_semiconductor_overlay as r38


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_OUTPUT = (
    ROOT
    / "output/r38_accelerating_volatility_capacity_fill_1375"
)
SPA_OUTPUT = ROOT / "output/strategy_library_frontier_spa"
OUTPUT = (
    ROOT
    / "output/r38_accelerating_volatility_1375_final_candidate_audit"
)
TARGET = "r38_accel1375"
REQUIRED_SAMPLES = (
    "normal_synthetic",
    "proxy_synthetic",
    "normal_live",
)
DAILY_RETURN_ABSOLUTE_TOLERANCE = 5e-6
METRIC_ABSOLUTE_TOLERANCE = 2e-5


@dataclass(frozen=True)
class AuditRecord:
    category: str
    requirement: str
    passed: bool
    evidence: str
    source: str


def path_digest(values: pd.Series) -> str:
    array = np.round(values.to_numpy(dtype=np.float64), 14)
    return sha256(array.tobytes()).hexdigest()


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate_market_frame(
    frame: pd.DataFrame,
) -> dict[str, object]:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    return {
        "rows": len(frame),
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
        "monotonic": bool(frame.index.is_monotonic_increasing),
        "duplicate_dates": int(frame.index.duplicated().sum()),
        "missing_values": int(numeric.isna().sum().sum()),
        "nonfinite_values": int((~finite).sum()),
        "nonpositive_values": int(numeric.le(0.0).sum().sum()),
    }


def freeze_sample(
    settings: dict[str, object],
    cutoff: pd.Timestamp,
) -> dict[str, object]:
    frozen = dict(settings)
    for key in ("opens", "closes"):
        frame = settings[key]
        assert isinstance(frame, pd.DataFrame)
        frozen[key] = frame.loc[:cutoff].copy()
    return frozen


def build_audit() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    acceptance = pd.read_csv(RESEARCH_OUTPUT / "acceptance.csv")
    saved_metrics = pd.read_csv(
        RESEARCH_OUTPUT / "relative_to_r38_metrics.csv"
    )
    neighborhoods = pd.read_csv(
        RESEARCH_OUTPUT / "parameter_neighborhood.csv"
    )
    event_windows = pd.read_csv(
        RESEARCH_OUTPUT / "event_windows.csv"
    )
    spa = pd.read_csv(SPA_OUTPUT / "spa_target_audit.csv")
    samples = r21._build_samples()
    rows: list[AuditRecord] = []

    def add(
        category: str,
        requirement: str,
        passed: bool,
        evidence: str,
        source: str,
    ) -> None:
        rows.append(
            AuditRecord(
                category,
                requirement,
                bool(passed),
                evidence,
                source,
            )
        )

    research_pass = bool(
        acceptance.loc[
            acceptance["gate"].ne("research_pass"),
            "passed",
        ]
        .astype(bool)
        .all()
        and acceptance.loc[
            acceptance["gate"].eq("research_pass"),
            "passed",
        ]
        .astype(bool)
        .all()
    )
    add(
        "research",
        "All preregistered research gates pass",
        research_pass,
        (
            f"passed={int(acceptance['passed'].astype(bool).sum())}/"
            f"{len(acceptance)}"
        ),
        "r38_accelerating_volatility_capacity_fill_1375/acceptance.csv",
    )
    selected_spa = spa.loc[spa["target"].eq(TARGET)]
    spa_pass = bool(
        set(selected_spa["block_days"].astype(int))
        == {21, 63, 126}
        and selected_spa["familywise_spa_p_value"].lt(0.05).all()
    )
    add(
        "multiple_comparisons",
        "Full-library SPA passes at 21, 63, and 126 sessions",
        spa_pass,
        "; ".join(
            f"{int(row.block_days)}d p={row.familywise_spa_p_value:.6f}"
            for row in selected_spa.itertuples(index=False)
        ),
        "strategy_library_frontier_spa/spa_target_audit.csv",
    )

    rebuild_rows: list[dict[str, object]] = []
    year_rows: list[dict[str, object]] = []
    drawdown_rows: list[dict[str, object]] = []
    saved_daily = pd.read_csv(
        RESEARCH_OUTPUT / "normal_synthetic_candidate_daily.csv",
        index_col="date",
        parse_dates=True,
    )["net_return"]
    research_cutoff = saved_daily.index.max()
    for sample_name in REQUIRED_SAMPLES:
        settings = freeze_sample(
            samples[sample_name],
            research_cutoff,
        )
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in r30.COST_SCENARIOS:
            r11_path, _, _ = simulate_fixed_r11(settings, scenario)
            r38_path, _ = r38.simulate_candidate(settings, scenario)
            trial, diagnostics = candidate.simulate_candidate(
                settings,
                scenario,
            )
            common = r38_path.index.intersection(trial.index)
            for period, (start, end) in periods.items():
                selected = common[
                    (common >= start) & (common <= end)
                ]
                rebuilt = metric_delta(
                    r38_path.loc[selected, "net_return"],
                    trial.loc[selected, "net_return"],
                )
                saved = saved_metrics.loc[
                    saved_metrics["sample"].eq(sample_name)
                    & saved_metrics["scenario"].eq(scenario.name)
                    & saved_metrics["period"].eq(period)
                ].iloc[0]
                metric_error = max(
                    abs(
                        float(saved[column])
                        - float(rebuilt[column])
                    )
                    for column in (
                        "baseline_cagr",
                        "baseline_max_drawdown",
                        "candidate_cagr",
                        "candidate_max_drawdown",
                    )
                )
                rebuild_rows.append(
                    {
                        "sample": sample_name,
                        "scenario": scenario.name,
                        "period": period,
                        "metric_max_abs_error": metric_error,
                        "minimum_pre_gde_cash": float(
                            diagnostics["pre_gde_cash_weight"].min()
                        ),
                        "maximum_non_cash": float(
                            diagnostics[
                                "implemented_non_cash_weight"
                            ].max()
                        ),
                    }
                )
            if scenario.name == "current_liquidity":
                years, events = robustness_rows(
                    sample_name,
                    candidate.CANDIDATE,
                    r11_path["net_return"],
                    trial["net_return"],
                )
                year_rows.extend(years)
                drawdown_rows.extend(events)
                if sample_name == "normal_synthetic":
                    daily_error = float(
                        (
                            saved_daily
                            - trial["net_return"].reindex(
                                saved_daily.index
                            )
                        )
                        .abs()
                        .max()
                    )
                    add(
                        "independent_rebuild",
                        "Fresh normal path matches saved daily returns",
                        bool(
                            saved_daily.index.equals(trial.index)
                            and daily_error
                            <= DAILY_RETURN_ABSOLUTE_TOLERANCE
                        ),
                        (
                            f"max daily error={daily_error:.3e}; "
                            f"tolerance="
                            f"{DAILY_RETURN_ABSOLUTE_TOLERANCE:.1e}; "
                            f"cutoff={research_cutoff.date()}; "
                            f"digest={path_digest(trial['net_return'])}"
                        ),
                        "fresh rebuild and saved candidate daily path",
                    )
    rebuild = pd.DataFrame(rebuild_rows)
    leave_year = pd.DataFrame(year_rows)
    leave_drawdown = pd.DataFrame(drawdown_rows)
    add(
        "independent_rebuild",
        "All saved period metrics match fresh rebuilds",
        bool(
            rebuild["metric_max_abs_error"]
            .le(METRIC_ABSOLUTE_TOLERANCE)
            .all()
        ),
        (
            f"rows={len(rebuild)}; max error="
            f"{rebuild['metric_max_abs_error'].max():.3e}; "
            f"tolerance={METRIC_ABSOLUTE_TOLERANCE:.1e}; "
            f"cutoff={research_cutoff.date()}"
        ),
        "fresh rebuild versus relative_to_r38_metrics.csv",
    )
    add(
        "risk_limits",
        "Every rebuild respects -20% cash and 120% non-cash limits",
        bool(
            rebuild["minimum_pre_gde_cash"]
            .ge(candidate.vc.CASH_FLOOR - 1e-12)
            .all()
            and rebuild["maximum_non_cash"].le(1.20 + 1e-12).all()
        ),
        (
            f"minimum cash={rebuild['minimum_pre_gde_cash'].min():.6f}; "
            f"maximum non-cash={rebuild['maximum_non_cash'].max():.6f}"
        ),
        "fresh rebuild diagnostics",
    )
    add(
        "deletion_robustness",
        "Every leave-one-year and leave-one-drawdown result stays positive",
        bool(
            leave_year["annualized_relative_log_return"].gt(0.0).all()
            and leave_drawdown[
                "annualized_relative_log_return"
            ].gt(0.0).all()
        ),
        (
            f"years={len(leave_year)}; drawdowns={len(leave_drawdown)}; "
            f"minimum={min(leave_year['annualized_relative_log_return'].min(), leave_drawdown['annualized_relative_log_return'].min()):.6f}"
        ),
        "fresh leave-one-out calculations versus R11",
    )
    add(
        "parameter_robustness",
        "All declared economic neighbors pass point and hard-limit gates",
        bool(
            neighborhoods.loc[
                ~neighborhoods["neighborhood"].str.startswith(
                    "identity"
                ),
                "point_target_pass",
            ]
            .astype(bool)
            .all()
            and neighborhoods["hard_limit_pass"].astype(bool).all()
        ),
        (
            f"rows={len(neighborhoods)}; "
            f"minimum CAGR="
            f"{neighborhoods['vs_r11_candidate_cagr'].min():.6f}"
        ),
        "parameter_neighborhood.csv",
    )
    add(
        "event_risk",
        "All frozen fast-selloff windows remain inside tolerance",
        bool(
            event_windows["max_drawdown_delta"]
            .ge(-candidate.vc.R38_DRAWDOWN_TOLERANCE)
            .all()
        ),
        (
            f"worst delta="
            f"{event_windows['max_drawdown_delta'].min():.6f}"
        ),
        "event_windows.csv",
    )

    data_rows: list[dict[str, object]] = []
    for sample_name, settings in samples.items():
        for kind in ("opens", "closes"):
            frame = settings[kind]
            assert isinstance(frame, pd.DataFrame)
            data_rows.append(
                {
                    "sample": sample_name,
                    "kind": kind,
                    **validate_market_frame(frame),
                }
            )
    data_audit = pd.DataFrame(data_rows)
    add(
        "data_integrity",
        "All market inputs are ordered, unique, finite, and positive",
        bool(
            data_audit["monotonic"].all()
            and data_audit["duplicate_dates"].eq(0).all()
            and data_audit["missing_values"].eq(0).all()
            and data_audit["nonfinite_values"].eq(0).all()
            and data_audit["nonpositive_values"].eq(0).all()
        ),
        (
            f"frames={len(data_audit)}; "
            f"last={data_audit['last_date'].max()}; "
            f"missing={data_audit['missing_values'].sum()}"
        ),
        "fresh R38 source market frames",
    )
    return (
        pd.DataFrame(asdict(row) for row in rows),
        rebuild,
        leave_year,
        leave_drawdown,
        data_audit,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (
        audit,
        rebuild,
        leave_year,
        leave_drawdown,
        data_audit,
    ) = build_audit()
    audit.to_csv(OUTPUT / "requirements.csv", index=False)
    rebuild.to_csv(OUTPUT / "rebuild_detail.csv", index=False)
    leave_year.to_csv(OUTPUT / "leave_one_year_out.csv", index=False)
    leave_drawdown.to_csv(
        OUTPUT / "leave_one_drawdown_event_out.csv",
        index=False,
    )
    data_audit.to_csv(OUTPUT / "data_audit.csv", index=False)
    source_paths = (
        RESEARCH_OUTPUT / "acceptance.csv",
        RESEARCH_OUTPUT / "relative_to_r38_metrics.csv",
        RESEARCH_OUTPUT / "parameter_neighborhood.csv",
        RESEARCH_OUTPUT / "normal_synthetic_candidate_daily.csv",
        SPA_OUTPUT / "spa_target_audit.csv",
    )
    hashes = {
        str(path.relative_to(ROOT)): file_digest(path)
        for path in source_paths
    }
    (OUTPUT / "source_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    qualification = bool(audit["passed"].astype(bool).all())
    summary = {
        "candidate": candidate.CANDIDATE,
        "qualification_pass": qualification,
        "requirements": len(audit),
        "requirements_passed": int(
            audit["passed"].astype(bool).sum()
        ),
        "research_output": str(RESEARCH_OUTPUT.relative_to(ROOT)),
        "spa_target": TARGET,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(audit.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()

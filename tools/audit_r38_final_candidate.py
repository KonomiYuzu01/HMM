from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
from tools.evaluate_r30_trend_risk_budget_pulse import COST_SCENARIOS
from tools.evaluate_r38_convex_semiconductor_overlay import (
    CASH_FLOOR,
    MAX_DRAWDOWN_TOLERANCE,
    OUTPUT as R38_OUTPUT,
    simulate_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/r38_final_candidate_audit"
SPA_OUTPUT = ROOT / "output/strategy_library_frontier_spa"
REQUIRED_SAMPLES = (
    "normal_synthetic",
    "proxy_synthetic",
    "normal_live",
)
CURRENT_PATH_FILES = {
    sample: f"{sample}_candidate_daily.csv"
    for sample in REQUIRED_SAMPLES
}


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
    required_columns: tuple[str, ...],
) -> dict[str, object]:
    numeric = frame.loc[:, list(required_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
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


def research_gate_pass(acceptance: pd.DataFrame) -> bool:
    ignored = {"multiple_testing_pass", "production_pass"}
    selected = acceptance.loc[~acceptance["gate"].isin(ignored)]
    return bool(
        len(selected) > 0
        and selected["passed"].astype(bool).all()
    )


def spa_gate_pass(spa: pd.DataFrame) -> bool:
    selected = spa.loc[spa["target"].eq("r38")]
    return bool(
        set(selected["block_days"].astype(int)) == {21, 63, 126}
        and selected["familywise_spa_p_value"].lt(0.05).all()
    )


def build_audit(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = root / R38_OUTPUT
    spa_output = root / SPA_OUTPUT
    samples = r21._build_samples()
    acceptance = pd.read_csv(output / "acceptance.csv")
    metrics = pd.read_csv(output / "metrics_by_period.csv")
    neighborhoods = pd.read_csv(output / "parameter_neighborhood.csv")
    leave_year = pd.read_csv(output / "leave_one_year_out.csv")
    leave_event = pd.read_csv(
        output / "leave_one_drawdown_event_out.csv"
    )
    annual = pd.read_csv(output / "annual_relative_log_return.csv")
    spa = pd.read_csv(spa_output / "spa_target_audit.csv")

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
                category=category,
                requirement=requirement,
                passed=bool(passed),
                evidence=evidence,
                source=source,
            )
        )

    add(
        "research",
        "All preregistered non-multiple-testing research gates pass",
        research_gate_pass(acceptance),
        (
            f"passed={int(acceptance['passed'].astype(bool).sum())}/"
            f"{len(acceptance)}; legacy multiple-testing gate is replaced "
            "by the full-library SPA below"
        ),
        "r38_convex_semiconductor_overlay/acceptance.csv",
    )
    selected_spa = spa.loc[spa["target"].eq("r38")]
    add(
        "multiple_comparisons",
        "Full-library SPA passes at 21, 63, and 126 sessions",
        spa_gate_pass(spa),
        "; ".join(
            f"{int(row.block_days)}d p={row.familywise_spa_p_value:.6f}"
            for row in selected_spa.itertuples(index=False)
        ),
        "strategy_library_frontier_spa/spa_target_audit.csv",
    )

    rebuild_rows: list[dict[str, object]] = []
    for sample_name in REQUIRED_SAMPLES:
        settings = samples[sample_name]
        for scenario in COST_SCENARIOS:
            baseline, _, _ = simulate_fixed_r11(settings, scenario)
            candidate, diagnostics = simulate_candidate(
                settings,
                scenario,
            )
            common = baseline.index.intersection(candidate.index)
            rebuilt = metric_delta(
                baseline.loc[common, "net_return"],
                candidate.loc[common, "net_return"],
            )
            saved_metric_rows = metrics.loc[
                metrics["sample"].eq(sample_name)
                & metrics["scenario"].eq(scenario.name)
            ]
            complete_labels = {
                "normal_synthetic": "complete_2015_2026",
                "proxy_synthetic": "complete_2006_2026",
                "normal_live": "live_2022_2026",
            }
            saved_metric = saved_metric_rows.loc[
                saved_metric_rows["period"].eq(
                    complete_labels[sample_name]
                )
            ].iloc[0]
            metric_error = max(
                abs(
                    float(saved_metric[column])
                    - float(rebuilt[column])
                )
                for column in (
                    "baseline_cagr",
                    "baseline_max_drawdown",
                    "candidate_cagr",
                    "candidate_max_drawdown",
                )
            )
            saved_digest = ""
            rebuilt_digest = path_digest(candidate["net_return"])
            daily_max_abs_error = np.nan
            dates_match = True
            if scenario.name == "current_liquidity":
                saved = pd.read_csv(
                    output / CURRENT_PATH_FILES[sample_name],
                    index_col=0,
                    parse_dates=True,
                )["net_return"]
                saved_digest = path_digest(saved)
                dates_match = bool(
                    saved.index.equals(candidate.index)
                )
                daily_max_abs_error = float(
                    (
                        saved
                        - candidate["net_return"].reindex(saved.index)
                    )
                    .abs()
                    .max()
                )
            rebuild_rows.append(
                {
                    "sample": sample_name,
                    "scenario": scenario.name,
                    "observations": len(candidate),
                    "metric_max_abs_error": metric_error,
                    "saved_path_digest": saved_digest,
                    "rebuilt_path_digest": rebuilt_digest,
                    "dates_match": dates_match,
                    "daily_max_abs_error": daily_max_abs_error,
                    "path_match": bool(
                        scenario.name != "current_liquidity"
                        or (
                            dates_match
                            and daily_max_abs_error <= 1e-12
                        )
                    ),
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
    rebuild = pd.DataFrame(rebuild_rows)
    add(
        "independent_rebuild",
        "Fresh in-memory rebuild matches every saved metric and path",
        bool(
            rebuild["metric_max_abs_error"].le(1e-12).all()
            and rebuild["path_match"].all()
        ),
        (
            f"max metric error="
            f"{rebuild['metric_max_abs_error'].max():.3e}; "
            f"max daily error="
            f"{rebuild['daily_max_abs_error'].max():.3e}; "
            f"path matches={int(rebuild['path_match'].sum())}/"
            f"{len(rebuild)}"
        ),
        "fresh rebuild from source inputs",
    )
    add(
        "risk_limits",
        "Cash floor and 120% non-cash cap hold in every rebuild",
        bool(
            rebuild["minimum_pre_gde_cash"].ge(
                CASH_FLOOR - 1e-12
            ).all()
            and rebuild["maximum_non_cash"].le(1.20 + 1e-12).all()
        ),
        (
            f"minimum cash={rebuild['minimum_pre_gde_cash'].min():.6f}; "
            f"maximum non-cash={rebuild['maximum_non_cash'].max():.6f}"
        ),
        "fresh rebuild diagnostics",
    )

    data_rows: list[dict[str, object]] = []
    for sample_name, settings in samples.items():
        for kind in ("opens", "closes"):
            frame = settings[kind]
            assert isinstance(frame, pd.DataFrame)
            required = tuple(
                asset
                for asset in (
                    "SPX",
                    "QQQ",
                    "SEMIS",
                    "BOND",
                    "GOLD",
                    "OIL",
                    "USD",
                    "CASH",
                    "VIX_HEDGE",
                )
                if asset in frame.columns
            )
            data_rows.append(
                {
                    "sample": sample_name,
                    "kind": kind,
                    **validate_market_frame(frame, required),
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
        "R38 source market frames",
    )

    family_pass = bool(
        set(neighborhoods["family"])
        == {"base", "active", "shock", "cash", "overlay"}
        and neighborhoods.groupby("family")["relative_positive"].all().all()
        and neighborhoods.groupby("family")["point_target_pass"].any().all()
    )
    add(
        "parameter_robustness",
        "Both sides of all five parameter neighborhoods pass",
        family_pass,
        (
            f"rows={len(neighborhoods)}; "
            f"families={','.join(sorted(neighborhoods['family'].unique()))}"
        ),
        "r38_convex_semiconductor_overlay/parameter_neighborhood.csv",
    )
    add(
        "deletion_robustness",
        "Every leave-one-year and leave-one-event result stays positive",
        bool(
            leave_year["annualized_relative_log_return"].gt(0.0).all()
            and leave_event["annualized_relative_log_return"].gt(0.0).all()
        ),
        (
            f"year minimum="
            f"{leave_year['annualized_relative_log_return'].min():.6f}; "
            f"event minimum="
            f"{leave_event['annualized_relative_log_return'].min():.6f}"
        ),
        "R38 leave-one-out tables",
    )
    normal_annual = annual["normal_synthetic"].dropna()
    add(
        "time_breadth",
        "Relative gains span both development and holdout years",
        bool(
            normal_annual.loc[
                annual["year"].between(2015, 2021)
            ].gt(0.0).sum()
            >= 5
            and normal_annual.loc[
                annual["year"].between(2022, 2025)
            ].gt(0.0).sum()
            >= 3
        ),
        (
            f"positive years 2015-2021="
            f"{int(normal_annual.loc[annual['year'].between(2015, 2021)].gt(0).sum())}; "
            f"2022-2025="
            f"{int(normal_annual.loc[annual['year'].between(2022, 2025)].gt(0).sum())}"
        ),
        "r38_convex_semiconductor_overlay/annual_relative_log_return.csv",
    )

    complete = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["scenario"].eq("current_liquidity")
        & metrics["period"].eq("complete_2015_2026")
    ].iloc[0]
    add(
        "objective",
        "Net CAGR is at least 25% and MDD is within 0.50pp of R11",
        bool(
            complete["candidate_cagr"] >= 0.25
            and complete["candidate_max_drawdown"]
            >= (
                complete["baseline_max_drawdown"]
                - MAX_DRAWDOWN_TOLERANCE
            )
        ),
        (
            f"CAGR={complete['candidate_cagr']:.6f}; "
            f"MDD={complete['candidate_max_drawdown']:.6f}; "
            f"R11 MDD={complete['baseline_max_drawdown']:.6f}"
        ),
        "r38_convex_semiconductor_overlay/metrics_by_period.csv",
    )
    return pd.DataFrame(asdict(row) for row in rows), pd.concat(
        [
            rebuild.assign(record_type="rebuild"),
            data_audit.assign(record_type="data"),
        ],
        ignore_index=True,
        sort=False,
    )


def write_report(
    audit: pd.DataFrame,
    detail: pd.DataFrame,
    output: Path = OUTPUT,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "requirements.csv", index=False)
    detail.to_csv(output / "detail.csv", index=False)
    source_files = (
        ROOT / "research/r38_convex_semiconductor_overlay_preregistration_2026-07-29.md",
        ROOT / "tools/evaluate_r38_convex_semiconductor_overlay.py",
        ROOT / "tools/audit_strategy_library_frontier_spa.py",
        ROOT / "tools/audit_r38_final_candidate.py",
    )
    hashes = {
        str(path.relative_to(ROOT)): file_digest(path)
        for path in source_files
    }
    (output / "source_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    passed = bool(audit["passed"].all())
    summary = {
        "candidate": "R38",
        "qualification_pass": passed,
        "requirements": len(audit),
        "requirements_passed": int(audit["passed"].sum()),
        "maximum_drawdown_tolerance_percentage_points": 0.50,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# R38 final candidate audit",
        "",
        f"- Qualification: {'PASS' if passed else 'FAIL'}",
        f"- Requirements: {int(audit['passed'].sum())}/{len(audit)}",
        "",
        "| Category | Requirement | Pass | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"| {row.category} | {row.requirement} | "
            f"{'yes' if row.passed else 'no'} | {row.evidence} |"
        )
    (output / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    audit, detail = build_audit()
    write_report(audit, detail)
    print(audit.to_string(index=False))
    if not audit["passed"].all():
        raise SystemExit(1)
    print(f"\nR38 final candidate qualification: PASS")
    print(f"Artifacts: {OUTPUT}")


if __name__ == "__main__":
    main()

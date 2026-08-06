from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "r11_production_qualification_audit"

NORMAL_STRICT = (
    ROOT
    / "output/r10_combined_family_reality_check/"
    "normal_cap15_rel50bp_gde10_band200bp_selected_"
    "r11_risk1p065_gde10_family_reality_check.csv"
)
PROXY_STRICT = (
    ROOT
    / "output/r10_combined_family_reality_check/"
    "proxy_cap15_rel50bp_gde10_band200bp_selected_"
    "r11_risk1p065_gde10_family_reality_check.csv"
)


@dataclass(frozen=True)
class AuditRecord:
    category: str
    requirement: str
    status: str
    blocking: bool
    evidence: str
    source: str


def _full_period_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "normal_synthetic": "complete_2015_2026",
        "proxy_synthetic": "complete_2006_2026",
        "normal_live": "live_2022_2026",
    }
    selected = []
    for sample, period in periods.items():
        selected.append(
            metrics[
                metrics["sample"].eq(sample)
                & metrics["period"].eq(period)
            ]
        )
    return pd.concat(selected, ignore_index=True)


def _metric_range(frame: pd.DataFrame, column: str) -> str:
    return f"{frame[column].min():.6f} to {frame[column].max():.6f}"


def build_audit(root: Path = ROOT) -> pd.DataFrame:
    final_dir = root / "output/r11_final_candidate_audit"
    robust_dir = root / "output/r11_final_candidate_robustness"
    data_dir = root / "output/r10_selected_data_audit"
    production_dir = (
        root
        / "output/paper_core_growth_gold20_r11_diversified_financing"
    )

    metrics = pd.read_csv(final_dir / "metrics_by_period.csv")
    acceptance = pd.read_csv(
        final_dir / "acceptance.csv", index_col=0
    )["value"]
    tracking = pd.read_csv(final_dir / "tracking_uncertainty.csv")
    seeds = pd.read_csv(robust_dir / "seed_metrics.csv")
    leave_year = pd.read_csv(robust_dir / "leave_one_year_out.csv")
    leave_event = pd.read_csv(robust_dir / "leave_one_event_out.csv")
    normal_strict = pd.read_csv(NORMAL_STRICT)
    proxy_strict = pd.read_csv(PROXY_STRICT)
    file_audit = pd.read_csv(data_dir / "file_audit.csv")
    cross_file = pd.read_csv(
        data_dir / "cross_file_diagnostics.csv", index_col=0
    )["value"]
    targets = pd.read_csv(production_dir / "next_target_weights.csv")
    diagnostics = pd.read_csv(
        production_dir / "next_signal_diagnostics.csv", index_col=0
    )["value"]
    config = yaml.safe_load(
        (
            root
            / "config/"
            "paper_core_growth_gold20_r11_diversified_financing.yaml"
        ).read_text()
    )

    rows: list[AuditRecord] = []

    def add(
        category: str,
        requirement: str,
        passed: bool,
        evidence: str,
        source: str,
        *,
        blocking: bool = True,
        warning: bool = False,
    ) -> None:
        if warning:
            status = "warning"
        else:
            status = "pass" if passed else "fail"
        rows.append(
            AuditRecord(
                category=category,
                requirement=requirement,
                status=status,
                blocking=blocking,
                evidence=evidence,
                source=source,
            )
        )

    candidate = str(acceptance["candidate"])
    add(
        "selection",
        "Frozen candidate is the registered R11 path",
        candidate == "risk1.065_gde10",
        f"candidate={candidate}; release={config['release']}",
        "acceptance.csv; production YAML",
    )

    full = _full_period_rows(metrics)
    for scenario in ("current_liquidity", "cost_stress"):
        scenario_rows = full[
            full["scenario"].eq(scenario)
            & full["rollout_share"].eq(1.0)
        ]
        passed = (
            len(scenario_rows) == 3
            and scenario_rows["cagr_delta"].gt(0).all()
            and scenario_rows["sharpe_delta"].gt(0).all()
            and scenario_rows["candidate_max_drawdown"].ge(-0.18).all()
        )
        add(
            "executable_backtest",
            f"Full R11 beats R9 in all full windows under {scenario}",
            bool(passed),
            (
                f"CAGR delta {_metric_range(scenario_rows, 'cagr_delta')}; "
                f"Sharpe delta {_metric_range(scenario_rows, 'sharpe_delta')}; "
                "worst MDD "
                f"{scenario_rows['candidate_max_drawdown'].min():.6f}"
            ),
            "r11_final_candidate_audit/metrics_by_period.csv",
        )

    staged = full[full["rollout_share"].eq(0.25)]
    staged_passed = (
        len(staged) == 6
        and staged["cagr_delta"].gt(0).all()
        and staged["sharpe_delta"].gt(0).all()
        and staged["candidate_max_drawdown"].ge(-0.18).all()
    )
    add(
        "executable_backtest",
        "Initial 25% production rollout beats R9 across samples and costs",
        bool(staged_passed),
        (
            f"CAGR delta {_metric_range(staged, 'cagr_delta')}; "
            f"Sharpe delta {_metric_range(staged, 'sharpe_delta')}; "
            f"worst MDD {staged['candidate_max_drawdown'].min():.6f}"
        ),
        "r11_final_candidate_audit/metrics_by_period.csv",
    )

    independent = metrics[
        metrics["sample"].eq("normal_synthetic")
        & metrics["period"].isin(
            ["development_2015_2021", "holdout_2022_2025"]
        )
        & metrics["rollout_share"].eq(1.0)
    ]
    independent_passed = (
        len(independent) == 4
        and independent["cagr_delta"].gt(0).all()
        and independent["sharpe_delta"].gt(0).all()
        and independent["candidate_max_drawdown"].ge(-0.18).all()
    )
    add(
        "independent_periods",
        "Development and frozen holdout both pass under both cost levels",
        bool(independent_passed),
        (
            f"CAGR delta {_metric_range(independent, 'cagr_delta')}; "
            f"Sharpe delta {_metric_range(independent, 'sharpe_delta')}; "
            f"worst MDD {independent['candidate_max_drawdown'].min():.6f}"
        ),
        "r11_final_candidate_audit/metrics_by_period.csv",
    )

    long_eras = metrics[
        metrics["sample"].eq("proxy_synthetic")
        & metrics["period"].isin(["early_2006_2014", "late_2015_2026"])
        & metrics["rollout_share"].eq(1.0)
    ]
    add(
        "independent_periods",
        "Both long-proxy eras retain positive CAGR advantage",
        bool(
            len(long_eras) == 4
            and long_eras["cagr_delta"].gt(0).all()
            and long_eras["candidate_max_drawdown"].ge(-0.18).all()
        ),
        (
            f"CAGR delta {_metric_range(long_eras, 'cagr_delta')}; "
            f"worst MDD {long_eras['candidate_max_drawdown'].min():.6f}"
        ),
        "r11_final_candidate_audit/metrics_by_period.csv",
    )
    early_sharpe = long_eras[
        long_eras["period"].eq("early_2006_2014")
    ]["sharpe_delta"]
    add(
        "limitations",
        "Early proxy era Sharpe is not uniformly better",
        False,
        f"Sharpe delta {_metric_range(long_eras, 'sharpe_delta')}",
        "r11_final_candidate_audit/metrics_by_period.csv",
        blocking=False,
        warning=bool(early_sharpe.lt(0).any()),
    )

    complete_seed_rows = seeds[seeds["period"].str.startswith("complete")]
    seed_passed = (
        set(complete_seed_rows["seed"]) == {7, 42, 123}
        and complete_seed_rows["cagr_delta"].gt(0).all()
        and complete_seed_rows["sharpe_delta"].gt(0).all()
    )
    add(
        "parameter_robustness",
        "Seeds 7, 42, and 123 retain positive CAGR and Sharpe deltas",
        bool(seed_passed),
        (
            f"CAGR delta {_metric_range(complete_seed_rows, 'cagr_delta')}; "
            f"Sharpe delta {_metric_range(complete_seed_rows, 'sharpe_delta')}"
        ),
        "r11_final_candidate_robustness/seed_metrics.csv",
    )
    add(
        "limitations",
        "One individual seed exceeds the 18% drawdown budget",
        False,
        (
            "worst individual-seed MDD "
            f"{complete_seed_rows['candidate_max_drawdown'].min():.6f}; "
            "production uses the three-seed ensemble"
        ),
        "r11_final_candidate_robustness/seed_metrics.csv",
        blocking=False,
        warning=bool(
            complete_seed_rows["candidate_max_drawdown"].lt(-0.18).any()
        ),
    )

    add(
        "parameter_robustness",
        "Leave-one-year-out relative return remains positive",
        bool(leave_year["annualized_relative_log_return"].gt(0).all()),
        (
            "minimum annualized relative log return "
            f"{leave_year['annualized_relative_log_return'].min():.6f}"
        ),
        "r11_final_candidate_robustness/leave_one_year_out.csv",
    )
    add(
        "parameter_robustness",
        "Leave-one-guard-event-out relative return remains positive",
        bool(leave_event["annualized_relative_log_return"].gt(0).all()),
        (
            "minimum annualized relative log return "
            f"{leave_event['annualized_relative_log_return'].min():.6f}"
        ),
        "r11_final_candidate_robustness/leave_one_event_out.csv",
    )

    family = pd.concat([normal_strict, proxy_strict], ignore_index=True)
    family_passed = (
        len(family) == 6
        and family["selected_rank"].eq(1).all()
        and family["familywise_reality_check_p_value"].lt(0.05).all()
        and family["baseline_reconstruction_error"].le(1e-12).all()
    )
    add(
        "multiple_comparisons",
        "Strict normal/proxy candidate-family Reality Checks pass",
        bool(family_passed),
        (
            f"paths={normal_strict['unique_candidate_paths'].iloc[0]}/"
            f"{proxy_strict['unique_candidate_paths'].iloc[0]}; "
            "familywise p "
            f"{family['familywise_reality_check_p_value'].min():.6f} to "
            f"{family['familywise_reality_check_p_value'].max():.6f}; "
            "max baseline error "
            f"{family['baseline_reconstruction_error'].max():.3e}"
        ),
        "r10_combined_family_reality_check/*.csv",
    )

    tracking_passed = (
        len(tracking) == 6
        and tracking["probability_all_objectives"].ge(0.95).all()
        and tracking["max_drawdown_p05"].ge(-0.18).all()
    )
    add(
        "implementation_uncertainty",
        "GDE tracking-error simulations pass at every rollout and cost level",
        bool(tracking_passed),
        (
            "minimum joint success probability "
            f"{tracking['probability_all_objectives'].min():.4f}; "
            f"worst 5th-percentile MDD {tracking['max_drawdown_p05'].min():.6f}"
        ),
        "r11_final_candidate_audit/tracking_uncertainty.csv",
    )

    file_passed = (
        file_audit["hash_verified"].eq(1).all()
        and file_audit["duplicate_dates"].eq(0).all()
        and file_audit["nonpositive_values"].eq(0).all()
    )
    cross_passed = (
        float(cross_file["lookahead_detected"]) == 0
        and float(cross_file["gde_cache_lags_core_by_calendar_days"]) == 0
        and float(cross_file["selected_execution_delay_trading_days"]) == 1
    )
    add(
        "data_integrity",
        "All selected caches pass hashes, dates, prices, and causal checks",
        bool(file_passed and cross_passed),
        (
            f"files={len(file_audit)}; core/GDE="
            f"{cross_file['core_price_last_date']}/"
            f"{cross_file['gde_open_close_last_date']}; "
            f"lookahead={cross_file['lookahead_detected']}"
        ),
        "r10_selected_data_audit/*.csv",
    )

    target_columns = (
        "staged_account_lower",
        "staged_account_center",
        "staged_account_upper",
    )
    target_sums = targets[list(target_columns)].sum()
    add(
        "production_output",
        "Staged target lower/center/upper portfolios each sum to 100%",
        bool((target_sums.sub(1.0).abs() <= 1e-12).all()),
        "; ".join(
            f"{column}={target_sums[column]:.12f}"
            for column in target_columns
        ),
        "R11 production next_target_weights.csv",
    )
    add(
        "production_output",
        "Production output is staged and orders are safely blocked",
        bool(
            diagnostics["status"] == "staged_production"
            and int(float(diagnostics["orders_executable"])) == 0
            and "missing complete real positions"
            in diagnostics["order_blockers"]
        ),
        (
            f"status={diagnostics['status']}; "
            f"orders_executable={diagnostics['orders_executable']}; "
            f"blockers={diagnostics['order_blockers']}"
        ),
        "R11 production next_signal_diagnostics.csv",
    )

    return pd.DataFrame(asdict(row) for row in rows)


def write_report(audit: pd.DataFrame, output: Path = OUTPUT) -> None:
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "requirements.csv", index=False)

    blocking = audit[audit["blocking"]]
    warnings = audit[audit["status"].eq("warning")]
    lines = [
        "# R11 production qualification audit",
        "",
        f"- Blocking requirements: {len(blocking)}",
        f"- Passed: {blocking['status'].eq('pass').sum()}",
        f"- Failed: {blocking['status'].eq('fail').sum()}",
        f"- Non-blocking warnings: {len(warnings)}",
        "",
        "| Category | Requirement | Status | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"| {row.category} | {row.requirement} | {row.status} | "
            f"{row.evidence} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    audit = build_audit()
    write_report(audit)
    print(audit.to_string(index=False))
    failures = audit[audit["blocking"] & audit["status"].eq("fail")]
    if not failures.empty:
        raise SystemExit(1)
    print(
        "\nQualification result: PASS "
        f"({audit['blocking'].sum()} blocking requirements, "
        f"{audit['status'].eq('warning').sum()} warnings)"
    )
    print(f"Artifacts: {OUTPUT}")


if __name__ == "__main__":
    main()

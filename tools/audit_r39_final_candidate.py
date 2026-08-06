from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.relative_damage import causal_relative_damage
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r39_relative_damage_concentration_veto as candidate


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_OUTPUT = ROOT / "output/r39_relative_damage_concentration_veto"
PARENT_AUDIT = (
    ROOT
    / "output/r38_accelerating_volatility_1375_final_candidate_audit"
)
OUTPUT = ROOT / "output/r39_final_candidate_audit"
METRIC_ABSOLUTE_TOLERANCE = 2e-9
DAILY_ABSOLUTE_TOLERANCE = 2e-12


@dataclass(frozen=True)
class AuditRecord:
    category: str
    requirement: str
    passed: bool
    evidence: str
    source: str


def file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


def validate_market_frame(frame: pd.DataFrame) -> bool:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return bool(
        frame.index.is_monotonic_increasing
        and not frame.index.duplicated().any()
        and not numeric.isna().any().any()
        and np.isfinite(numeric.to_numpy(dtype=float)).all()
        and numeric.gt(0.0).all().all()
    )


def build_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    acceptance = pd.read_csv(RESEARCH_OUTPUT / "acceptance.csv")
    saved_metrics = pd.read_csv(
        RESEARCH_OUTPUT / "metrics_by_period.csv"
    )
    neighborhoods = pd.read_csv(
        RESEARCH_OUTPUT / "parameter_neighborhood.csv"
    )
    events = pd.read_csv(RESEARCH_OUTPUT / "event_windows.csv")
    saved_daily = pd.read_csv(
        RESEARCH_OUTPUT / "normal_synthetic_candidate_daily.csv",
        index_col="date",
        parse_dates=True,
    )["net_return"]
    cutoff = saved_daily.index.max()
    parent = json.loads(
        (PARENT_AUDIT / "summary.json").read_text(encoding="utf-8")
    )
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

    add(
        "parent",
        "Parent R38 final candidate remains qualified",
        bool(parent["qualification_pass"]),
        (
            f"{parent['requirements_passed']}/"
            f"{parent['requirements']} requirements"
        ),
        "r38_accelerating_volatility_1375_final_candidate_audit",
    )
    add(
        "research",
        "Every declared R39 research gate passes",
        bool(acceptance["passed"].astype(bool).all()),
        (
            f"{int(acceptance['passed'].astype(bool).sum())}/"
            f"{len(acceptance)} gates"
        ),
        "r39_relative_damage_concentration_veto/acceptance.csv",
    )

    samples = r21._build_samples()
    rebuild_rows: list[dict[str, object]] = []
    rebuilt_normal: pd.Series | None = None
    for sample_name, original in samples.items():
        settings = freeze_sample(original, cutoff)
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            baseline = candidate.simulate_baseline(settings, scenario)
            trial, diagnostics = candidate.simulate_candidate(
                settings,
                scenario,
            )
            common = baseline.index.intersection(trial.index)
            for period, (start, end) in periods.items():
                selected = common[
                    (common >= start) & (common <= end)
                ]
                rebuilt = metric_delta(
                    baseline.loc[selected, "net_return"],
                    trial.loc[selected, "net_return"],
                )
                saved = saved_metrics.loc[
                    saved_metrics["sample"].eq(sample_name)
                    & saved_metrics["scenario"].eq(scenario.name)
                    & saved_metrics["period"].eq(period)
                ].iloc[0]
                error = max(
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
                        "metric_max_abs_error": error,
                        "maximum_growth_budget_error": float(
                            diagnostics["growth_budget_error"].max()
                        ),
                    }
                )
            if (
                sample_name == "normal_synthetic"
                and scenario.name == "current_liquidity"
            ):
                rebuilt_normal = trial["net_return"]
    rebuild = pd.DataFrame(rebuild_rows)
    assert rebuilt_normal is not None
    daily_error = float(
        (
            saved_daily
            - rebuilt_normal.reindex(saved_daily.index)
        )
        .abs()
        .max()
    )
    add(
        "independent_rebuild",
        "Fresh daily path matches the frozen research path",
        bool(
            rebuilt_normal.index.equals(saved_daily.index)
            and daily_error <= DAILY_ABSOLUTE_TOLERANCE
        ),
        (
            f"max error={daily_error:.3e}; "
            f"cutoff={cutoff.date().isoformat()}"
        ),
        "fresh R39 rebuild",
    )
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
            f"{rebuild['metric_max_abs_error'].max():.3e}"
        ),
        "fresh rebuild versus metrics_by_period.csv",
    )
    add(
        "risk_invariant",
        "Every rebuild conserves the QQQ plus SMH growth budget",
        bool(
            rebuild["maximum_growth_budget_error"].le(1e-12).all()
        ),
        (
            "max error="
            f"{rebuild['maximum_growth_budget_error'].max():.3e}"
        ),
        "fresh R39 diagnostics",
    )
    neighborhood_budget = neighborhoods[
        "entry_account_loss_budget"
    ].fillna(candidate.ENTRY_ACCOUNT_LOSS_BUDGET)
    add(
        "parameter_robustness",
        "All declared neighbors control the current loss budget",
        bool(
            neighborhoods["current_active"].astype(bool).all()
            and (
                neighborhoods[
                    "current_implemented_account_relative_loss"
                ]
                <= neighborhood_budget + 1e-12
            ).all()
        ),
        (
            f"neighbors={len(neighborhoods)}; max implemented="
            f"{neighborhoods['current_implemented_account_relative_loss'].max():.6f}"
        ),
        "parameter_neighborhood.csv",
    )
    add(
        "event_risk",
        "Declared event-window drawdown changes stay within 50 bps",
        bool(events["max_drawdown_delta"].ge(-0.005).all()),
        (
            f"worst delta={events['max_drawdown_delta'].min():.6f}"
        ),
        "event_windows.csv",
    )

    normal_closes = samples["normal_synthetic"]["closes"]
    assert isinstance(normal_closes, pd.DataFrame)
    index = normal_closes.index
    original_damage = causal_relative_damage(
        normal_closes,
        index,
        lookback_days=candidate.LOOKBACK_DAYS,
    )
    changed = normal_closes.copy()
    changed.loc[index[-1], "SEMIS"] *= 0.50
    changed_damage = causal_relative_damage(
        changed,
        index,
        lookback_days=candidate.LOOKBACK_DAYS,
    )
    add(
        "causality",
        "The current close cannot change the current execution signal",
        bool(
            original_damage.iloc[-1].equals(
                changed_damage.iloc[-1]
            )
        ),
        "signal uses a one-session execution lag",
        "causal_relative_damage",
    )
    data_ok = all(
        validate_market_frame(frame)
        for settings in samples.values()
        for key in ("opens", "closes")
        for frame in [settings[key]]
        if isinstance(frame, pd.DataFrame)
    )
    add(
        "data_integrity",
        "All research market frames are ordered, unique, finite, and positive",
        data_ok,
        f"samples={len(samples)}",
        "fresh market frames",
    )
    return pd.DataFrame(asdict(row) for row in rows), rebuild


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit, rebuild = build_audit()
    audit.to_csv(OUTPUT / "requirements.csv", index=False)
    rebuild.to_csv(OUTPUT / "rebuild_detail.csv", index=False)
    sources = (
        RESEARCH_OUTPUT / "acceptance.csv",
        RESEARCH_OUTPUT / "metrics_by_period.csv",
        RESEARCH_OUTPUT / "parameter_neighborhood.csv",
        RESEARCH_OUTPUT / "event_windows.csv",
        RESEARCH_OUTPUT / "normal_synthetic_candidate_daily.csv",
    )
    hashes = {
        str(path.relative_to(ROOT)): file_digest(path)
        for path in sources
    }
    (OUTPUT / "source_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    passed = bool(audit["passed"].astype(bool).all())
    summary = {
        "candidate": candidate.CANDIDATE,
        "qualification_pass": passed,
        "requirements": len(audit),
        "requirements_passed": int(audit["passed"].sum()),
        "research_output": str(RESEARCH_OUTPUT.relative_to(ROOT)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(audit.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT.resolve()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

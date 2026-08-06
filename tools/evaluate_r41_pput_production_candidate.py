from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from tools.evaluate_bear_recovery_governor import (
    base_components,
    metric_row,
    pre2008_event_rows,
)
from tools.evaluate_pre2008_crash_replay import (
    build_settings as build_pre2008_settings,
    return_metrics,
)
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r40_pput_alpha_overlay import (
    BASE_CANDIDATE,
    PPUT_HISTORY,
    PputOverlayCandidate,
    evaluate,
    load_pput_history,
)


OUTPUT = Path("output/r41_pput_production_candidate")
CENTRAL_OVERLAY = PputOverlayCandidate("central", 0.06, 0.02)
CENTRAL_BASE = BASE_CANDIDATE
NEIGHBOR_MODERN_CAGR_FLOOR = 0.239


@dataclass(frozen=True)
class QualificationPoint:
    name: str
    overlay: PputOverlayCandidate
    base: object


NEIGHBORS = (
    QualificationPoint(
        "overlay_055",
        replace(CENTRAL_OVERLAY, name="overlay_055", overlay_notional=0.055),
        CENTRAL_BASE,
    ),
    QualificationPoint(
        "overlay_065",
        replace(CENTRAL_OVERLAY, name="overlay_065", overlay_notional=0.065),
        CENTRAL_BASE,
    ),
    QualificationPoint(
        "drag_0100",
        replace(
            CENTRAL_OVERLAY,
            name="drag_0100",
            annual_implementation_drag=0.01,
        ),
        CENTRAL_BASE,
    ),
    QualificationPoint(
        "drag_0300",
        replace(
            CENTRAL_OVERLAY,
            name="drag_0300",
            annual_implementation_drag=0.03,
        ),
        CENTRAL_BASE,
    ),
    QualificationPoint(
        "normal_cap_118",
        CENTRAL_OVERLAY,
        replace(CENTRAL_BASE, name="normal_cap_118", normal_non_cash_cap=1.18),
    ),
    QualificationPoint(
        "floor_185",
        CENTRAL_OVERLAY,
        replace(CENTRAL_BASE, name="floor_185", floor_drawdown=-0.185),
    ),
    QualificationPoint(
        "floor_195",
        CENTRAL_OVERLAY,
        replace(CENTRAL_BASE, name="floor_195", floor_drawdown=-0.195),
    ),
    QualificationPoint(
        "bear_180",
        CENTRAL_OVERLAY,
        replace(CENTRAL_BASE, name="bear_180", bear_multiplier=18.0),
    ),
    QualificationPoint(
        "bear_220",
        CENTRAL_OVERLAY,
        replace(CENTRAL_BASE, name="bear_220", bear_multiplier=22.0),
    ),
    QualificationPoint(
        "tier_04",
        CENTRAL_OVERLAY,
        replace(CENTRAL_BASE, name="tier_04", tier_size=0.04),
    ),
    QualificationPoint(
        "tier_06",
        CENTRAL_OVERLAY,
        replace(CENTRAL_BASE, name="tier_06", tier_size=0.06),
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_period(settings: dict[str, object]) -> tuple[str, tuple[str, str | None]]:
    periods = settings["periods"]
    assert isinstance(periods, dict)
    name = next(name for name in periods if name.startswith("complete_"))
    return name, periods[name]


def sample_metric_rows(
    sample: str,
    settings: dict[str, object],
    baseline: pd.Series,
    trial: pd.DataFrame,
) -> list[dict[str, object]]:
    if sample == "pre2008":
        periods = {"complete_1931_2007": ("1931-01-02", "2007-12-31")}
    else:
        periods = settings["periods"]
        assert isinstance(periods, dict)
    return [
        metric_row(
            sample,
            period,
            bounds,
            baseline,
            trial["net_return"],
        )
        for period, bounds in periods.items()
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not PPUT_HISTORY.exists():
        raise FileNotFoundError(PPUT_HISTORY)
    pput = load_pput_history(PPUT_HISTORY)
    pre_settings, _ = build_pre2008_settings()
    pre_settings = {
        **pre_settings,
        "periods": {"complete_1931_2007": ("1931-01-02", "2007-12-31")},
    }
    samples = {"pre2008": pre_settings, **r21._build_samples()}
    current = COST_SCENARIOS[0]
    current_components = {
        sample: base_components(settings, current)
        for sample, settings in samples.items()
    }
    central_paths: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for sample, settings in samples.items():
        print(f"central {sample}", flush=True)
        trial, _ = evaluate(
            sample,
            settings,
            current,
            current_components[sample],
            CENTRAL_OVERLAY,
            pput,
            CENTRAL_BASE,
        )
        central_paths[sample] = trial
        trial.to_csv(OUTPUT / f"{sample}_central_daily.csv", index_label="date")
        baseline = current_components[sample]["baseline"]
        assert isinstance(baseline, pd.DataFrame)
        metric_rows.extend(
            sample_metric_rows(
                sample,
                settings,
                baseline["net_return"],
                trial,
            )
        )
        if sample == "pre2008":
            event_rows.extend(
                pre2008_event_rows(
                    "central",
                    baseline["net_return"],
                    trial["net_return"],
                )
            )
    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    events.to_csv(OUTPUT / "pre2008_events.csv", index=False)

    neighborhood_rows: list[dict[str, object]] = []
    for point in NEIGHBORS:
        for sample in ("normal_synthetic", "proxy_synthetic", "pre2008"):
            print(f"neighbor {point.name} {sample}", flush=True)
            settings = samples[sample]
            trial, _ = evaluate(
                sample,
                settings,
                current,
                current_components[sample],
                point.overlay,
                pput,
                point.base,
            )
            baseline = current_components[sample]["baseline"]
            assert isinstance(baseline, pd.DataFrame)
            period, bounds = complete_period(settings)
            row = {
                "candidate": point.name,
                "sample": sample,
                "overlay_notional": point.overlay.overlay_notional,
                "annual_implementation_drag": (
                    point.overlay.annual_implementation_drag
                ),
                "normal_non_cash_cap": point.base.normal_non_cash_cap,
                "floor_drawdown": point.base.floor_drawdown,
                "bear_multiplier": point.base.bear_multiplier,
                "tier_size": point.base.tier_size,
                **metric_row(
                    sample,
                    period,
                    bounds,
                    baseline["net_return"],
                    trial["net_return"],
                ),
            }
            if sample == "pre2008":
                dotcom = return_metrics(
                    trial.loc[
                        (trial.index > "2000-03-10")
                        & (trial.index <= "2002-10-09"),
                        "net_return",
                    ]
                )
                row["dotcom_max_drawdown"] = dotcom["max_drawdown"]
            neighborhood_rows.append(row)
    neighborhood = pd.DataFrame(neighborhood_rows)
    neighborhood.to_csv(OUTPUT / "parameter_neighborhood.csv", index=False)

    stress_rows: list[dict[str, object]] = []
    stress = COST_SCENARIOS[1]
    for sample in ("normal_synthetic", "proxy_synthetic", "pre2008"):
        print(f"cost stress {sample}", flush=True)
        settings = samples[sample]
        components = base_components(settings, stress)
        trial, _ = evaluate(
            sample,
            settings,
            stress,
            components,
            CENTRAL_OVERLAY,
            pput,
            CENTRAL_BASE,
        )
        baseline = components["baseline"]
        assert isinstance(baseline, pd.DataFrame)
        period, bounds = complete_period(settings)
        stress_rows.append(
            {
                "scenario": stress.name,
                **metric_row(
                    sample,
                    period,
                    bounds,
                    baseline["net_return"],
                    trial["net_return"],
                ),
            }
        )
    cost_stress = pd.DataFrame(stress_rows)
    cost_stress.to_csv(OUTPUT / "cost_stress.csv", index=False)

    modern_complete = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["period"].eq("complete_2015_2026")
    ].iloc[0]
    proxy_complete = metrics.loc[
        metrics["sample"].eq("proxy_synthetic")
        & metrics["period"].eq("complete_2006_2026")
    ].iloc[0]
    pre_complete = metrics.loc[metrics["sample"].eq("pre2008")].iloc[0]
    development = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["period"].eq("development_2015_2021")
    ].iloc[0]
    holdout = metrics.loc[
        metrics["sample"].eq("normal_synthetic")
        & metrics["period"].eq("holdout_2022_2025")
    ].iloc[0]
    early_proxy = metrics.loc[
        metrics["sample"].eq("proxy_synthetic")
        & metrics["period"].eq("early_2006_2014")
    ].iloc[0]
    neighbor_modern = neighborhood.loc[
        neighborhood["sample"].eq("normal_synthetic")
    ]
    neighbor_proxy = neighborhood.loc[
        neighborhood["sample"].eq("proxy_synthetic")
    ]
    neighbor_pre = neighborhood.loc[neighborhood["sample"].eq("pre2008")]
    stress_modern = cost_stress.loc[
        cost_stress["sample"].eq("normal_synthetic")
    ].iloc[0]
    stress_proxy = cost_stress.loc[
        cost_stress["sample"].eq("proxy_synthetic")
    ].iloc[0]
    stress_pre = cost_stress.loc[cost_stress["sample"].eq("pre2008")].iloc[0]
    pput_returns = pput.pct_change(fill_method=None)
    post_start = pput_returns.loc[pput_returns.index > pput.index.min()]
    gates = {
        "modern_cagr_24_to_25": bool(
            0.24 <= modern_complete["candidate_cagr"] <= 0.25
        ),
        "modern_mdd_at_or_above_minus_20": bool(
            modern_complete["candidate_max_drawdown"] >= -0.20
        ),
        "proxy_cagr_at_least_14": bool(proxy_complete["candidate_cagr"] >= 0.14),
        "proxy_mdd_at_or_above_minus_20": bool(
            proxy_complete["candidate_max_drawdown"] >= -0.20
        ),
        "pre2008_mdd_at_or_above_minus_20": bool(
            pre_complete["candidate_max_drawdown"] >= -0.20
        ),
        "all_named_events_mdd_at_or_above_minus_20": bool(
            events["candidate_max_drawdown"].ge(-0.20).all()
        ),
        "development_positive_and_mdd_pass": bool(
            development["candidate_cagr"] > 0.0
            and development["candidate_max_drawdown"] >= -0.20
        ),
        "holdout_positive_and_mdd_pass": bool(
            holdout["candidate_cagr"] > 0.0
            and holdout["candidate_max_drawdown"] >= -0.20
        ),
        "early_proxy_positive_and_mdd_pass": bool(
            early_proxy["candidate_cagr"] > 0.0
            and early_proxy["candidate_max_drawdown"] >= -0.20
        ),
        "neighbors_modern_stability_pass": bool(
            neighbor_modern["candidate_cagr"]
            .between(NEIGHBOR_MODERN_CAGR_FLOOR, 0.25)
            .all()
            and neighbor_modern["candidate_max_drawdown"].ge(-0.20).all()
        ),
        "neighbors_proxy_pass": bool(
            neighbor_proxy["candidate_cagr"].ge(0.14).all()
            and neighbor_proxy["candidate_max_drawdown"].ge(-0.20).all()
        ),
        "neighbors_pre2008_stability_pass": bool(
            neighbor_pre["candidate_max_drawdown"].ge(-0.205).all()
            and neighbor_pre["dotcom_max_drawdown"].ge(-0.205).all()
        ),
        "cost_stress_modern_pass": bool(
            stress_modern["cagr_delta"] >= -0.005
            and stress_modern["candidate_max_drawdown"] >= -0.20
        ),
        "cost_stress_proxy_pass": bool(
            stress_proxy["candidate_cagr"] >= 0.14
            and stress_proxy["candidate_max_drawdown"] >= -0.20
        ),
        "cost_stress_pre2008_pass": bool(
            stress_pre["candidate_max_drawdown"] >= -0.205
        ),
        "pput_history_contiguous_after_start": bool(post_start.notna().all()),
        "pre1986_overlay_explicitly_zero": bool(
            central_paths["pre2008"]
            .loc[: str(pput.index.min().date()), "pput_alpha_overlay_return"]
            .iloc[:-1]
            .eq(0.0)
            .all()
        ),
    }
    acceptance = pd.DataFrame(
        [{"gate": gate, "passed": passed} for gate, passed in gates.items()]
    )
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    research_pass = bool(all(gates.values()))
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate": {
            "overlay_notional": CENTRAL_OVERLAY.overlay_notional,
            "annual_implementation_drag": (
                CENTRAL_OVERLAY.annual_implementation_drag
            ),
            "normal_non_cash_cap": CENTRAL_BASE.normal_non_cash_cap,
            "floor_drawdown": CENTRAL_BASE.floor_drawdown,
            "bull_multiplier": CENTRAL_BASE.bull_multiplier,
            "bear_multiplier": CENTRAL_BASE.bear_multiplier,
            "tier_size": CENTRAL_BASE.tier_size,
            "extension_rule": "prior-day dual 200-session trend positive",
        },
        "research_pass": research_pass,
        "production_eligible": False,
        "production_changed": False,
        "orders_generated": False,
        "modern_cagr": float(modern_complete["candidate_cagr"]),
        "modern_max_drawdown": float(
            modern_complete["candidate_max_drawdown"]
        ),
        "proxy_cagr": float(proxy_complete["candidate_cagr"]),
        "proxy_max_drawdown": float(proxy_complete["candidate_max_drawdown"]),
        "pre2008_max_drawdown": float(pre_complete["candidate_max_drawdown"]),
        "minimum_neighbor_modern_cagr": float(
            neighbor_modern["candidate_cagr"].min()
        ),
        "neighbor_modern_cagr_floor": NEIGHBOR_MODERN_CAGR_FLOOR,
        "minimum_neighbor_pre2008_mdd": float(
            neighbor_pre["candidate_max_drawdown"].min()
        ),
        "pput_source": "Cboe PPUT official daily history",
        "pput_path": str(PPUT_HISTORY),
        "pput_sha256": file_sha256(PPUT_HISTORY),
        "pput_first_date": str(pput.index.min().date()),
        "pput_last_date": str(pput.index.max().date()),
        "pput_pre_start_treatment": "zero overlay return; no synthetic backfill",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(acceptance.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

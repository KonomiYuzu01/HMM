from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r30_trend_risk_budget_pulse as r30


ROOT = Path(__file__).resolve().parents[1]
GOVERNANCE_CONFIG = ROOT / "config/r38_anti_overfit_governance.yaml"
R38_CONFIG = ROOT / "config/paper_core_growth_gold20_r38_convex_overlay.yaml"
GOVERNANCE_SUMMARY = ROOT / "output/r38_anti_overfit_governance/summary.json"
R38_METADATA = ROOT / "output/paper_core_growth_gold20_r38_convex_overlay/run_metadata.json"
R38_FORWARD_PATH = ROOT / (
    "output/r38_accelerating_volatility_capacity_fill_1375/"
    "normal_synthetic_candidate_daily.csv"
)
CORE_PRICES = ROOT / "data/adjusted_open_close_2011_present.csv"
EXECUTION_LOG = ROOT / "output/r38_forward_execution_log.csv"
OUTPUT = ROOT / "output/r38_forward_review"

EXECUTION_COLUMNS = (
    "date",
    "signal_logged",
    "positions_reconciled",
    "realized_gde_one_way_cost_bps",
    "realized_financing_spread_bps",
    "unresolved_operational_incident",
)


@dataclass(frozen=True)
class ReviewRecord:
    category: str
    requirement: str
    passed: bool
    evidence: str


def maximum_drawdown(returns: pd.Series) -> float:
    values = returns.astype(float).to_numpy()
    if (values <= -1.0).any():
        raise ValueError("daily returns must be greater than -100%")
    equity = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return float(drawdown.min())


def forward_path_metrics(
    r11_returns: pd.Series,
    r38_returns: pd.Series,
    expected_dates: pd.DatetimeIndex,
    *,
    rollout_share: float = 0.25,
) -> dict[str, float | int | bool]:
    if not 0.0 <= rollout_share <= 1.0:
        raise ValueError("rollout share must be between zero and one")
    r11 = r11_returns.copy()
    r38 = r38_returns.copy()
    r11.index = pd.DatetimeIndex(pd.to_datetime(r11.index)).normalize()
    r38.index = pd.DatetimeIndex(pd.to_datetime(r38.index)).normalize()
    if r11.index.has_duplicates or r38.index.has_duplicates:
        raise ValueError("forward return paths must have unique dates")
    aligned = pd.concat(
        [r11.rename("r11"), r38.rename("r38")], axis=1
    ).reindex(expected_dates)
    complete = bool(not aligned.isna().any().any())
    selected = aligned.dropna()
    staged = (
        (1.0 - rollout_share) * selected["r11"]
        + rollout_share * selected["r38"]
    )
    r11_mdd = maximum_drawdown(selected["r11"])
    r38_mdd = maximum_drawdown(selected["r38"])
    staged_mdd = maximum_drawdown(staged)
    relative_log_return = float(
        (
            np.log1p(selected["r38"])
            - np.log1p(selected["r11"])
        ).sum()
    )
    return {
        "sessions": int(len(expected_dates)),
        "complete": complete,
        "r11_cumulative_return": float(
            (1.0 + selected["r11"]).prod() - 1.0
        ),
        "r38_cumulative_return": float(
            (1.0 + selected["r38"]).prod() - 1.0
        ),
        "staged_cumulative_return": float(
            (1.0 + staged).prod() - 1.0
        ),
        "relative_log_return": relative_log_return,
        "r11_max_drawdown": r11_mdd,
        "r38_max_drawdown": r38_mdd,
        "staged_max_drawdown": staged_mdd,
        "staged_relative_max_drawdown_delta": staged_mdd - r11_mdd,
        "relative_max_drawdown_delta": r38_mdd - r11_mdd,
    }


def _boolean_series(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    converted = normalized.map(mapping)
    if converted.isna().any():
        raise ValueError("execution log contains an invalid boolean value")
    return converted.astype(bool)


def execution_evidence(
    log: pd.DataFrame | None,
    expected_dates: pd.DatetimeIndex,
    *,
    maximum_gde_cost_bps: float,
    maximum_financing_spread_bps: float,
) -> dict[str, object]:
    if log is None:
        return {
            "log_present": False,
            "complete": False,
            "signals_complete": False,
            "positions_reconciled": False,
            "gde_cost_pass": False,
            "financing_cost_pass": False,
            "incident_free": False,
            "maximum_gde_cost_bps": None,
            "maximum_financing_spread_bps": None,
        }
    missing = sorted(set(EXECUTION_COLUMNS).difference(log.columns))
    if missing:
        raise ValueError(f"execution log is missing columns: {missing}")
    frame = log.loc[:, EXECUTION_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame["date"].duplicated().any():
        raise ValueError("execution log dates must be unique")
    frame = frame.set_index("date").sort_index().reindex(expected_dates)
    complete = bool(not frame.isna().any().any())
    if not complete:
        return {
            "log_present": True,
            "complete": False,
            "signals_complete": False,
            "positions_reconciled": False,
            "gde_cost_pass": False,
            "financing_cost_pass": False,
            "incident_free": False,
            "maximum_gde_cost_bps": None,
            "maximum_financing_spread_bps": None,
        }
    signals = _boolean_series(frame["signal_logged"])
    positions = _boolean_series(frame["positions_reconciled"])
    incidents = _boolean_series(frame["unresolved_operational_incident"])
    gde_cost = pd.to_numeric(frame["realized_gde_one_way_cost_bps"])
    financing = pd.to_numeric(frame["realized_financing_spread_bps"])
    return {
        "log_present": True,
        "complete": True,
        "signals_complete": bool(signals.all()),
        "positions_reconciled": bool(positions.all()),
        "gde_cost_pass": bool(gde_cost.le(maximum_gde_cost_bps).all()),
        "financing_cost_pass": bool(
            financing.le(maximum_financing_spread_bps).all()
        ),
        "incident_free": bool((~incidents).all()),
        "maximum_gde_cost_bps": float(gde_cost.max()),
        "maximum_financing_spread_bps": float(financing.max()),
    }


def expected_forward_dates(
    dates: pd.Series,
    *,
    freeze_price_as_of: str,
    current_price_as_of: str,
) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if normalized.has_duplicates or not normalized.is_monotonic_increasing:
        raise ValueError("market dates must be ordered and unique")
    freeze = pd.Timestamp(freeze_price_as_of)
    current = pd.Timestamp(current_price_as_of)
    return normalized[(normalized > freeze) & (normalized <= current)]


def build_review(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, object]]:
    governance_config = yaml.safe_load(
        (root / GOVERNANCE_CONFIG.relative_to(ROOT)).read_text()
    )
    strategy = yaml.safe_load(
        (root / R38_CONFIG.relative_to(ROOT)).read_text()
    )
    governance = json.loads(
        (root / GOVERNANCE_SUMMARY.relative_to(ROOT)).read_text()
    )
    metadata = json.loads(
        (root / R38_METADATA.relative_to(ROOT)).read_text()
    )
    freeze = dict(governance_config["freeze"])
    rollout = dict(strategy["rollout"])
    prices = pd.read_csv(
        root / CORE_PRICES.relative_to(ROOT), usecols=["date"]
    )
    expected_dates = expected_forward_dates(
        prices["date"],
        freeze_price_as_of=str(freeze["price_as_of"]),
        current_price_as_of=str(metadata["price_as_of"]),
    )
    settings = r21._build_samples()["normal_synthetic"]
    r11_path, _, _ = simulate_fixed_r11(settings, r30.COST_SCENARIOS[0])
    r38_path = pd.read_csv(
        root / R38_FORWARD_PATH.relative_to(ROOT),
        index_col="date",
        parse_dates=True,
    )
    path = forward_path_metrics(
        r11_path["net_return"],
        r38_path["net_return"],
        expected_dates,
        rollout_share=float(rollout["initial_r38_share_of_managed_capital"]),
    )
    execution_path = root / EXECUTION_LOG.relative_to(ROOT)
    log = pd.read_csv(execution_path) if execution_path.exists() else None
    evidence = execution_evidence(
        log,
        expected_dates,
        maximum_gde_cost_bps=75.0,
        maximum_financing_spread_bps=150.0,
    )
    rows: list[ReviewRecord] = []

    def add(category: str, requirement: str, passed: bool, detail: str) -> None:
        rows.append(ReviewRecord(category, requirement, bool(passed), detail))

    minimum_sessions = int(freeze["minimum_completed_forward_sessions"])
    drawdown_tolerance = 0.005
    add(
        "governance",
        "Frozen R38 governance remains operational",
        bool(governance["operational_pass"]),
        f"release={governance['strategy_release']}",
    )
    add(
        "sample",
        "Minimum completed forward sessions accumulated",
        len(expected_dates) >= minimum_sessions,
        f"completed={len(expected_dates)}; minimum={minimum_sessions}",
    )
    add(
        "path",
        "R11 and R38 forward paths cover every completed session",
        bool(path["complete"]),
        f"sessions={path['sessions']}",
    )
    add(
        "risk",
        "Full R38 forward drawdown is within 0.50pp of R11",
        float(path["relative_max_drawdown_delta"]) >= -drawdown_tolerance,
        (
            f"R11={float(path['r11_max_drawdown']):.4%}; "
            f"R38={float(path['r38_max_drawdown']):.4%}; "
            f"delta={float(path['relative_max_drawdown_delta']):.4%}"
        ),
    )
    add(
        "execution",
        "Execution evidence covers every completed forward session",
        bool(evidence["complete"]),
        f"path={EXECUTION_LOG.relative_to(ROOT)}",
    )
    add(
        "execution",
        "Signals and positions are completely reconciled",
        bool(evidence["signals_complete"])
        and bool(evidence["positions_reconciled"]),
        "requires confirmed daily account records",
    )
    add(
        "cost",
        "Realized GDE one-way cost does not exceed 75bps",
        bool(evidence["gde_cost_pass"]),
        f"maximum={evidence['maximum_gde_cost_bps']}",
    )
    add(
        "cost",
        "Realized financing spread does not exceed 150bps",
        bool(evidence["financing_cost_pass"]),
        f"maximum={evidence['maximum_financing_spread_bps']}",
    )
    add(
        "operations",
        "No unresolved operational incident exists",
        bool(evidence["incident_free"]),
        "daily incident flag must be false",
    )
    audit = pd.DataFrame(asdict(row) for row in rows)
    review_ready = bool(audit["passed"].all())
    blockers = audit.loc[~audit["passed"], "requirement"].tolist()
    summary: dict[str, object] = {
        "status": "complete",
        "strategy_release": str(governance["strategy_release"]),
        "price_as_of": str(metadata["price_as_of"]),
        "forward_sessions": int(len(expected_dates)),
        "minimum_forward_sessions": minimum_sessions,
        "forward_relative_log_return": path["relative_log_return"],
        "r11_forward_max_drawdown": path["r11_max_drawdown"],
        "r38_forward_max_drawdown": path["r38_max_drawdown"],
        "staged_forward_cumulative_return": path[
            "staged_cumulative_return"
        ],
        "staged_forward_max_drawdown": path["staged_max_drawdown"],
        "staged_relative_max_drawdown_delta": path[
            "staged_relative_max_drawdown_delta"
        ],
        "relative_max_drawdown_delta": path[
            "relative_max_drawdown_delta"
        ],
        "promotion_review_ready": review_ready,
        "automatic_promotion": False,
        "production_changed": False,
        "blockers": blockers,
    }
    return audit, summary


def write_review(
    audit: pd.DataFrame,
    summary: dict[str, object],
    output: Path = OUTPUT,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "requirements.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    template = pd.DataFrame(
        columns=EXECUTION_COLUMNS,
    )
    template.to_csv(output / "execution_log_template.csv", index=False)


def main() -> None:
    audit, summary = build_review()
    write_review(audit, summary)
    print(audit.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

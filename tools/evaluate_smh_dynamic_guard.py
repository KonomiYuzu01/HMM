from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_open_execution import simulate_open_execution
from regime_strategy.report import performance_metrics


OUTPUT = Path("output")
DESTINATION = OUTPUT / "smh_dynamic_guard"
STRATEGY = "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
OPEN_CLOSE = Path("data/adjusted_open_close_2011_present.csv")
ASSETS = [
    "SPX",
    "QQQ",
    "SEMIS",
    "BOND",
    "GOLD",
    "OIL",
    "USD",
    "CASH",
    "VIX_HEDGE",
]
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_present": ("2015-01-01", None),
    "recent_2024_present": ("2024-01-01", None),
}


@dataclass(frozen=True)
class OpenGapGuard:
    name: str
    absolute_gap_trigger: float | None = None
    relative_gap_trigger: float | None = None
    relative_gap_bypass_weighted_loss: float | None = None
    minimum_weighted_gap_loss: float | None = None
    minimum_prior_weighted_loss: float | None = None
    severe_weighted_gap_loss: float | None = None
    semis_multiplier: float = 1.0
    post_trigger_cap: float | None = None
    severe_post_trigger_cap: float | None = None
    continuation_gap_trigger: float | None = None
    continuation_cap: float | None = None
    trigger_delay_days: int = 0
    minimum_semis_weight: float = 0.0
    recovery_signal: str = "relative"
    recovery_confirmations: int = 1
    maximum_hold_days: int = 5
    overflow_asset: str = "QQQ"
    bridge_overflow_asset: str | None = None
    bridge_after_days: int = 0
    emergency_slippage_bps: float = 20.0


def variants() -> list[OpenGapGuard]:
    candidates = [OpenGapGuard("baseline")]
    for absolute_gap in (-0.02, -0.03, -0.04):
        absolute_label = int(round(abs(absolute_gap) * 100))
        for relative_gap in (-0.01, -0.02, -0.03):
            relative_label = int(round(abs(relative_gap) * 100))
            for multiplier in (0.0, 0.25, 0.50):
                multiplier_label = int(round(multiplier * 100))
                for confirmations in (1, 2):
                    for maximum_hold_days in (3, 5):
                        candidates.append(
                            OpenGapGuard(
                                name=(
                                    f"gap{absolute_label}_rel{relative_label}"
                                    f"_keep{multiplier_label}"
                                    f"_confirm{confirmations}"
                                    f"_hold{maximum_hold_days}"
                                ),
                                absolute_gap_trigger=absolute_gap,
                                relative_gap_trigger=relative_gap,
                                semis_multiplier=multiplier,
                                minimum_semis_weight=0.15,
                                recovery_confirmations=confirmations,
                                maximum_hold_days=maximum_hold_days,
                            )
                        )
    for absolute_gap in (-0.02, -0.03):
        absolute_label = int(round(abs(absolute_gap) * 100))
        for minimum_weight in (0.35, 0.50):
            minimum_label = int(round(minimum_weight * 100))
            for post_trigger_cap in (0.15, 0.20, 0.30):
                cap_label = int(round(post_trigger_cap * 100))
                for overflow_asset in ("QQQ", "CASH"):
                    for confirmations in (1, 2):
                        candidates.append(
                            OpenGapGuard(
                                name=(
                                    f"highconc{minimum_label}"
                                    f"_gap{absolute_label}"
                                    f"_cap{cap_label}"
                                    f"_to{overflow_asset}"
                                    f"_confirm{confirmations}"
                                ),
                                absolute_gap_trigger=absolute_gap,
                                post_trigger_cap=post_trigger_cap,
                                minimum_semis_weight=minimum_weight,
                                recovery_confirmations=confirmations,
                                maximum_hold_days=3,
                                overflow_asset=overflow_asset,
                            )
                        )
    for post_trigger_cap in (0.15, 0.20, 0.30):
        cap_label = int(round(post_trigger_cap * 100))
        for recovery_signal in ("relative", "semis", "pair"):
            for confirmations in (1, 2):
                for maximum_hold_days in (3, 5):
                    candidates.append(
                        OpenGapGuard(
                            name=(
                                f"highconc50_gap3_cap{cap_label}_toCASH"
                                f"_{recovery_signal}"
                                f"_confirm{confirmations}"
                                f"_hold{maximum_hold_days}"
                            ),
                            absolute_gap_trigger=-0.03,
                            post_trigger_cap=post_trigger_cap,
                            minimum_semis_weight=0.50,
                            recovery_signal=recovery_signal,
                            recovery_confirmations=confirmations,
                            maximum_hold_days=maximum_hold_days,
                            overflow_asset="CASH",
                        )
                        )
    for absolute_gap in (-0.025, -0.030, -0.035):
        gap_bps = int(round(abs(absolute_gap) * 10_000))
        for minimum_weight in (0.45, 0.50, 0.55, 0.60):
            minimum_label = int(round(minimum_weight * 100))
            for post_trigger_cap in (0.15, 0.20, 0.25, 0.30):
                cap_label = int(round(post_trigger_cap * 100))
                for maximum_hold_days in (4, 5, 6):
                    candidates.append(
                        OpenGapGuard(
                            name=(
                                f"neighborhood_gap{gap_bps}bp"
                                f"_min{minimum_label}"
                                f"_cap{cap_label}"
                                f"_hold{maximum_hold_days}"
                            ),
                            absolute_gap_trigger=absolute_gap,
                            post_trigger_cap=post_trigger_cap,
                            minimum_semis_weight=minimum_weight,
                            recovery_signal="relative",
                            recovery_confirmations=2,
                            maximum_hold_days=maximum_hold_days,
                            overflow_asset="CASH",
                        )
                        )
    for post_trigger_cap in (0.25, 0.30, 0.35):
        initial_label = int(round(post_trigger_cap * 100))
        for continuation_gap in (-0.01, -0.02, -0.03):
            continuation_label = int(round(abs(continuation_gap) * 100))
            for continuation_cap in (0.15, 0.20):
                deep_label = int(round(continuation_cap * 100))
                for confirmations in (1, 2):
                    for maximum_hold_days in (3, 5):
                        candidates.append(
                            OpenGapGuard(
                                name=(
                                    f"staged_initial{initial_label}"
                                    f"_continuegap{continuation_label}"
                                    f"_deep{deep_label}"
                                    f"_confirm{confirmations}"
                                    f"_hold{maximum_hold_days}"
                                ),
                                absolute_gap_trigger=-0.03,
                                post_trigger_cap=post_trigger_cap,
                                continuation_gap_trigger=continuation_gap,
                                continuation_cap=continuation_cap,
                                minimum_semis_weight=0.50,
                                recovery_signal="relative",
                                recovery_confirmations=confirmations,
                                maximum_hold_days=maximum_hold_days,
                                overflow_asset="CASH",
                            )
                        )
                        candidates.append(
                            OpenGapGuard(
                                name=(
                                    f"staged_initial{initial_label}"
                                    f"_continuegap{continuation_label}"
                                    f"_deep{deep_label}"
                                    f"_confirm{confirmations}"
                                    f"_hold{maximum_hold_days}"
                                    "_toQQQ"
                                ),
                                absolute_gap_trigger=-0.03,
                                post_trigger_cap=post_trigger_cap,
                                continuation_gap_trigger=continuation_gap,
                                continuation_cap=continuation_cap,
                                minimum_semis_weight=0.50,
                                recovery_signal="relative",
                                recovery_confirmations=confirmations,
                                maximum_hold_days=maximum_hold_days,
                                overflow_asset="QQQ",
                            )
                        )
    for maximum_hold_days in (5, 6):
        candidates.append(
            OpenGapGuard(
                name=(
                    "neighborhood_gap300bp_min50_cap15"
                    f"_hold{maximum_hold_days}_delay1"
                ),
                absolute_gap_trigger=-0.03,
                post_trigger_cap=0.15,
                trigger_delay_days=1,
                minimum_semis_weight=0.50,
                recovery_signal="relative",
                recovery_confirmations=2,
                maximum_hold_days=maximum_hold_days,
                overflow_asset="CASH",
            )
        )
    for post_trigger_cap in (0.20, 0.25, 0.35):
        cap_label = int(round(post_trigger_cap * 100))
        candidates.append(
            OpenGapGuard(
                name=(
                    "causal_gap250bp_min60"
                    f"_cap{cap_label}_confirm2_hold4"
                ),
                absolute_gap_trigger=-0.025,
                post_trigger_cap=post_trigger_cap,
                trigger_delay_days=1,
                minimum_semis_weight=0.60,
                recovery_signal="relative",
                recovery_confirmations=2,
                maximum_hold_days=4,
                overflow_asset="CASH",
            )
        )
    for bridge_after_days in (1, 2, 3):
        for maximum_hold_days in (5, 6):
            candidates.append(
                OpenGapGuard(
                    name=(
                        "cash_first_gap3_min50_cap15"
                        f"_bridgeQQQday{bridge_after_days}"
                        f"_hold{maximum_hold_days}"
                    ),
                    absolute_gap_trigger=-0.03,
                    post_trigger_cap=0.15,
                    minimum_semis_weight=0.50,
                    recovery_signal="relative",
                    recovery_confirmations=2,
                    maximum_hold_days=maximum_hold_days,
                    overflow_asset="CASH",
                    bridge_overflow_asset="QQQ",
                    bridge_after_days=bridge_after_days,
                )
            )
    for weighted_loss in (0.018, 0.020, 0.022, 0.024, 0.026):
        loss_label = int(round(weighted_loss * 10_000))
        for post_trigger_cap in (0.15, 0.20, 0.25):
            cap_label = int(round(post_trigger_cap * 100))
            for maximum_hold_days in (4, 5, 6):
                candidates.append(
                    OpenGapGuard(
                        name=(
                            f"accountloss{loss_label}bp"
                            f"_cap{cap_label}"
                            f"_hold{maximum_hold_days}"
                        ),
                        absolute_gap_trigger=-0.02,
                        minimum_weighted_gap_loss=weighted_loss,
                        post_trigger_cap=post_trigger_cap,
                        minimum_semis_weight=0.20,
                        recovery_signal="relative",
                        recovery_confirmations=2,
                        maximum_hold_days=maximum_hold_days,
                        overflow_asset="CASH",
                    )
                )
    for severe_weighted_loss in (0.022, 0.024):
        severe_label = int(round(severe_weighted_loss * 10_000))
        for post_trigger_cap in (0.25, 0.30, 0.35):
            cap_label = int(round(post_trigger_cap * 100))
            for severe_post_trigger_cap in (0.15, 0.20, 0.25):
                deep_label = int(round(severe_post_trigger_cap * 100))
                for maximum_hold_days in (4, 5, 6):
                    candidates.append(
                        OpenGapGuard(
                            name=(
                                "tierloss200bp"
                                f"_severe{severe_label}bp"
                                f"_cap{cap_label}"
                                f"_deep{deep_label}"
                                f"_hold{maximum_hold_days}"
                            ),
                            absolute_gap_trigger=-0.02,
                            minimum_weighted_gap_loss=0.020,
                            severe_weighted_gap_loss=severe_weighted_loss,
                            post_trigger_cap=post_trigger_cap,
                            severe_post_trigger_cap=severe_post_trigger_cap,
                            minimum_semis_weight=0.20,
                            recovery_signal="relative",
                            recovery_confirmations=2,
                            maximum_hold_days=maximum_hold_days,
                            overflow_asset="CASH",
                        )
                    )
    for continuation_gap in (-0.01, -0.02, -0.03):
        continuation_label = int(round(abs(continuation_gap) * 100))
        for continuation_cap in (0.15, 0.20):
            deep_label = int(round(continuation_cap * 100))
            for maximum_hold_days in (5, 6):
                candidates.append(
                    OpenGapGuard(
                        name=(
                            "accountloss200bp_cap25"
                            f"_continuegap{continuation_label}"
                            f"_deep{deep_label}"
                            f"_hold{maximum_hold_days}"
                        ),
                        absolute_gap_trigger=-0.02,
                        minimum_weighted_gap_loss=0.020,
                        post_trigger_cap=0.25,
                        continuation_gap_trigger=continuation_gap,
                        continuation_cap=continuation_cap,
                        minimum_semis_weight=0.20,
                        recovery_signal="relative",
                        recovery_confirmations=2,
                        maximum_hold_days=maximum_hold_days,
                        overflow_asset="CASH",
                    )
                )
    candidates.append(
        OpenGapGuard(
            name=(
                "accountloss200bp_rel2_or220bp_cap25"
                "_continuegap1_deep15_hold5_delay1"
            ),
            absolute_gap_trigger=-0.02,
            relative_gap_trigger=-0.02,
            relative_gap_bypass_weighted_loss=0.022,
            minimum_weighted_gap_loss=0.020,
            post_trigger_cap=0.25,
            continuation_gap_trigger=-0.01,
            continuation_cap=0.15,
            trigger_delay_days=1,
            minimum_semis_weight=0.20,
            recovery_signal="relative",
            recovery_confirmations=2,
            maximum_hold_days=5,
            overflow_asset="CASH",
        )
    )
    for prior_weighted_loss in (0.015, 0.020, 0.025, 0.030):
        prior_loss_label = int(round(prior_weighted_loss * 10_000))
        for post_trigger_cap in (0.15, 0.20, 0.25):
            cap_label = int(round(post_trigger_cap * 100))
            for maximum_hold_days in (2, 3, 4, 5):
                candidates.append(
                    OpenGapGuard(
                        name=(
                            f"priorloss{prior_loss_label}bp"
                            f"_cap{cap_label}"
                            f"_hold{maximum_hold_days}"
                        ),
                        minimum_prior_weighted_loss=prior_weighted_loss,
                        post_trigger_cap=post_trigger_cap,
                        minimum_semis_weight=0.20,
                        recovery_signal="relative",
                        recovery_confirmations=2,
                        maximum_hold_days=maximum_hold_days,
                        overflow_asset="CASH",
                    )
                )
    for post_trigger_cap in (0.15, 0.20, 0.25):
        cap_label = int(round(post_trigger_cap * 100))
        for maximum_hold_days in (4, 5, 6):
            candidates.append(
                OpenGapGuard(
                    name=(
                        "accountloss200bp_rel2_or220bp"
                        f"_cap{cap_label}"
                        f"_hold{maximum_hold_days}"
                    ),
                    absolute_gap_trigger=-0.02,
                    relative_gap_trigger=-0.02,
                    relative_gap_bypass_weighted_loss=0.022,
                    minimum_weighted_gap_loss=0.020,
                    post_trigger_cap=post_trigger_cap,
                    minimum_semis_weight=0.20,
                    recovery_signal="relative",
                    recovery_confirmations=2,
                    maximum_hold_days=maximum_hold_days,
                    overflow_asset="CASH",
                )
            )
    for continuation_gap in (-0.01, -0.02, -0.03):
        continuation_label = int(round(abs(continuation_gap) * 100))
        for continuation_cap in (0.15, 0.20):
            deep_label = int(round(continuation_cap * 100))
            for maximum_hold_days in (5, 6):
                candidates.append(
                    OpenGapGuard(
                        name=(
                            "accountloss200bp_rel2_or220bp_cap25"
                            f"_continuegap{continuation_label}"
                            f"_deep{deep_label}"
                            f"_hold{maximum_hold_days}"
                        ),
                        absolute_gap_trigger=-0.02,
                        relative_gap_trigger=-0.02,
                        relative_gap_bypass_weighted_loss=0.022,
                        minimum_weighted_gap_loss=0.020,
                        post_trigger_cap=0.25,
                        continuation_gap_trigger=continuation_gap,
                        continuation_cap=continuation_cap,
                        minimum_semis_weight=0.20,
                        recovery_signal="relative",
                        recovery_confirmations=2,
                        maximum_hold_days=maximum_hold_days,
                        overflow_asset="CASH",
                    )
                )
    for delay_days in (1,):
        candidates.append(
            OpenGapGuard(
                name=f"staged_initial25_deep15_hold5_delay{delay_days}",
                absolute_gap_trigger=-0.03,
                post_trigger_cap=0.25,
                continuation_gap_trigger=-0.01,
                continuation_cap=0.15,
                trigger_delay_days=delay_days,
                minimum_semis_weight=0.50,
                recovery_signal="relative",
                recovery_confirmations=2,
                maximum_hold_days=5,
                overflow_asset="CASH",
            )
        )
    return candidates


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = OUTPUT / STRATEGY
    weights = pd.read_csv(
        root / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        root / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    prices = pd.read_csv(OPEN_CLOSE, index_col=0, parse_dates=True)
    opens = prices[[f"open_{asset}" for asset in ASSETS]].copy()
    closes = prices[[f"close_{asset}" for asset in ASSETS]].copy()
    opens.columns = ASSETS
    closes.columns = ASSETS
    return weights, daily, opens.astype(float), closes.astype(float)


def transfer_semis(
    weights: np.ndarray,
    semis_multiplier: float,
    overflow_asset: str,
    post_trigger_cap: float | None = None,
) -> np.ndarray:
    adjusted = weights.copy()
    semis_index = ASSETS.index("SEMIS")
    overflow_index = ASSETS.index(overflow_asset)
    retained = float(adjusted[semis_index]) * semis_multiplier
    if post_trigger_cap is not None:
        retained = min(retained, post_trigger_cap)
    released = float(adjusted[semis_index]) - retained
    adjusted[semis_index] = retained
    adjusted[overflow_index] += released
    return adjusted


def restore_pair_mix(
    current: np.ndarray,
    baseline_weights: np.ndarray,
    overflow_asset: str,
) -> np.ndarray:
    adjusted = current.copy()
    semis_index = ASSETS.index("SEMIS")
    overflow_index = ASSETS.index(overflow_asset)
    pair_total = float(adjusted[overflow_index] + adjusted[semis_index])
    baseline_pair_total = float(
        baseline_weights[overflow_index] + baseline_weights[semis_index]
    )
    if pair_total <= 1e-12 or baseline_pair_total <= 1e-12:
        return adjusted
    semis_share = float(
        baseline_weights[semis_index] / baseline_pair_total
    )
    adjusted[semis_index] = pair_total * semis_share
    adjusted[overflow_index] = pair_total * (1.0 - semis_share)
    return adjusted


def switch_guard_overflow(
    current: np.ndarray,
    baseline_weights: np.ndarray,
    from_asset: str,
    to_asset: str,
) -> np.ndarray:
    if from_asset == to_asset:
        return current.copy()
    adjusted = current.copy()
    semis_index = ASSETS.index("SEMIS")
    from_index = ASSETS.index(from_asset)
    to_index = ASSETS.index(to_asset)
    group_indices = [semis_index, from_index, to_index]
    group_total = float(adjusted[group_indices].sum())
    baseline_group_total = float(baseline_weights[group_indices].sum())
    if group_total <= 1e-12 or baseline_group_total <= 1e-12:
        return adjusted
    target_from = float(
        group_total
        * baseline_weights[from_index]
        / baseline_group_total
    )
    adjusted[from_index] = target_from
    adjusted[to_index] = (
        group_total
        - float(adjusted[semis_index])
        - target_from
    )
    return adjusted


def account_weighted_gap_loss(
    semis_weight: float,
    semis_gap: float,
) -> float:
    return max(semis_weight, 0.0) * max(-semis_gap, 0.0)


def trigger_cap_for_loss(
    weighted_gap_loss: float,
    post_trigger_cap: float | None,
    severe_weighted_gap_loss: float | None,
    severe_post_trigger_cap: float | None,
) -> float | None:
    if (
        severe_weighted_gap_loss is not None
        and severe_post_trigger_cap is not None
        and weighted_gap_loss >= severe_weighted_gap_loss
    ):
        return severe_post_trigger_cap
    return post_trigger_cap


def loss_trigger_source(
    gap_rule_configured: bool,
    gap_rule_passed: bool,
    prior_weighted_loss: float,
    minimum_prior_weighted_loss: float | None,
) -> str | None:
    gap_triggered = gap_rule_configured and gap_rule_passed
    prior_triggered = (
        minimum_prior_weighted_loss is not None
        and prior_weighted_loss >= minimum_prior_weighted_loss
    )
    if gap_triggered and prior_triggered:
        return "both"
    if gap_triggered:
        return "gap"
    if prior_triggered:
        return "prior_close"
    return None


def relative_gap_filter_passed(
    relative_gap: float,
    relative_gap_trigger: float | None,
    weighted_gap_loss: float,
    bypass_weighted_gap_loss: float | None,
) -> bool:
    return (
        relative_gap_trigger is None
        or relative_gap <= relative_gap_trigger
        or (
            bypass_weighted_gap_loss is not None
            and weighted_gap_loss >= bypass_weighted_gap_loss
        )
    )


def simulate(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    guard: OpenGapGuard,
    cost_bps: float = 7.5,
    start_date: str = "2015-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = (
        weights.index.intersection(base_daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    dates = dates[dates >= start_date]
    baseline_weight_values = weights.loc[dates, ASSETS].to_numpy(dtype=float)
    base_trade_values = (
        base_daily.loc[dates, "turnover"].to_numpy(dtype=float) > 1e-14
    )
    open_values = opens.loc[dates, ASSETS].to_numpy(dtype=float)
    close_values = closes.loc[dates, ASSETS].to_numpy(dtype=float)
    overnight_asset_return_values = np.zeros_like(open_values)
    overnight_asset_return_values[1:] = (
        open_values[1:] / close_values[:-1] - 1.0
    )
    intraday_asset_return_values = close_values / open_values - 1.0
    close_to_close_return_values = np.zeros_like(close_values)
    close_to_close_return_values[1:] = (
        close_values[1:] / close_values[:-1] - 1.0
    )
    current = np.zeros(len(ASSETS), dtype=float)
    current[ASSETS.index("CASH")] = 1.0
    previous_date: pd.Timestamp | None = None
    active = False
    current_guard_cap: float | None = None
    active_days = 0
    positive_relative_days = 0
    current_overflow_asset = guard.overflow_asset
    rows: list[dict[str, float | int | bool]] = []
    weight_rows: list[np.ndarray] = []

    for position, date in enumerate(dates):
        was_active_at_open = active
        overnight_asset_returns = overnight_asset_return_values[position]
        if previous_date is None:
            semis_gap = 0.0
            qqq_gap = 0.0
            prior_relative_return = 0.0
            prior_semis_return = 0.0
            prior_qqq_return = 0.0
            prior_weighted_loss = 0.0
        else:
            semis_gap = float(
                overnight_asset_returns[ASSETS.index("SEMIS")]
            )
            qqq_gap = float(overnight_asset_returns[ASSETS.index("QQQ")])
            previous_close_return = (
                intraday_asset_return_values[position - 1]
                if position == 1
                else close_to_close_return_values[position - 1]
            )
            prior_relative_return = float(
                previous_close_return[ASSETS.index("SEMIS")]
                - previous_close_return[ASSETS.index("QQQ")]
            )
            prior_semis_return = float(
                previous_close_return[ASSETS.index("SEMIS")]
            )
            prior_qqq_return = float(
                previous_close_return[ASSETS.index("QQQ")]
            )
            prior_weighted_loss = account_weighted_gap_loss(
                float(
                    baseline_weight_values[
                        position - 1,
                        ASSETS.index("SEMIS"),
                    ]
                ),
                prior_semis_return,
            )

        overnight_return = float(current @ overnight_asset_returns)
        current = current * (1.0 + overnight_asset_returns) / (
            1.0 + overnight_return
        )

        base_trade = bool(base_trade_values[position])
        desired: np.ndarray | None = None
        trade_reason = "none"
        emergency_trade = False
        if base_trade:
            desired = baseline_weight_values[position].copy()
            trade_reason = "base"

        recovered = False
        if active:
            active_days += 1
            recovery_signal_positive = (
                prior_relative_return > 0.0
                if guard.recovery_signal == "relative"
                else (
                    prior_semis_return > 0.0
                    if guard.recovery_signal == "semis"
                    else prior_semis_return > 0.0 and prior_qqq_return > 0.0
                )
            )
            positive_relative_days = (
                positive_relative_days + 1
                if recovery_signal_positive
                else 0
            )
            recovered = (
                positive_relative_days >= guard.recovery_confirmations
                or active_days >= guard.maximum_hold_days
            )
            if recovered:
                reference = (
                    desired
                    if desired is not None
                    else current
                )
                desired = restore_pair_mix(
                    reference,
                    baseline_weight_values[position],
                    current_overflow_asset,
                )
                trade_reason = "recovery"
                emergency_trade = True
                active = False
                current_guard_cap = None
                current_overflow_asset = guard.overflow_asset
                active_days = 0
                positive_relative_days = 0

        bridged = (
            active
            and was_active_at_open
            and guard.bridge_overflow_asset is not None
            and guard.bridge_after_days > 0
            and active_days >= guard.bridge_after_days
            and current_overflow_asset != guard.bridge_overflow_asset
        )
        if bridged:
            reference = desired if desired is not None else current
            desired = switch_guard_overflow(
                reference,
                baseline_weight_values[position],
                current_overflow_asset,
                guard.bridge_overflow_asset,
            )
            current_overflow_asset = guard.bridge_overflow_asset
            trade_reason = "bridge_overflow"
            emergency_trade = True

        relative_gap = semis_gap - qqq_gap
        signal_position = position - guard.trigger_delay_days
        trigger_semis_gap = (
            float(
                overnight_asset_return_values[
                    signal_position,
                    ASSETS.index("SEMIS"),
                ]
            )
            if signal_position >= 0
            else 0.0
        )
        trigger_qqq_gap = (
            float(
                overnight_asset_return_values[
                    signal_position,
                    ASSETS.index("QQQ"),
                ]
            )
            if signal_position >= 0
            else 0.0
        )
        trigger_relative_gap = trigger_semis_gap - trigger_qqq_gap
        deepened = (
            active
            and was_active_at_open
            and guard.continuation_gap_trigger is not None
            and guard.continuation_cap is not None
            and semis_gap <= guard.continuation_gap_trigger
            and (
                current_guard_cap is None
                or current_guard_cap > guard.continuation_cap
            )
        )
        if deepened:
            reference = desired if desired is not None else current
            desired = transfer_semis(
                reference,
                guard.semis_multiplier,
                current_overflow_asset,
                guard.continuation_cap,
            )
            current_guard_cap = guard.continuation_cap
            trade_reason = "deepen"
            emergency_trade = True
        trigger_reference = (
            desired[ASSETS.index("SEMIS")]
            if desired is not None
            else current[ASSETS.index("SEMIS")]
        )
        weighted_gap_loss = account_weighted_gap_loss(
            float(trigger_reference),
            trigger_semis_gap,
        )
        gap_rule_configured = (
            guard.absolute_gap_trigger is not None
            or guard.relative_gap_trigger is not None
            or guard.minimum_weighted_gap_loss is not None
        )
        gap_rule_passed = (
            (
                guard.absolute_gap_trigger is None
                or trigger_semis_gap <= guard.absolute_gap_trigger
            )
            and (
                relative_gap_filter_passed(
                    trigger_relative_gap,
                    guard.relative_gap_trigger,
                    weighted_gap_loss,
                    guard.relative_gap_bypass_weighted_loss,
                )
            )
            and (
                guard.minimum_weighted_gap_loss is None
                or weighted_gap_loss >= guard.minimum_weighted_gap_loss
            )
        )
        trigger_source = loss_trigger_source(
            gap_rule_configured,
            gap_rule_passed,
            prior_weighted_loss,
            guard.minimum_prior_weighted_loss,
        )
        triggered = (
            not active
            and trigger_source is not None
            and trigger_reference >= guard.minimum_semis_weight
        )
        if triggered:
            reference = desired if desired is not None else current
            selected_trigger_cap = trigger_cap_for_loss(
                weighted_gap_loss,
                guard.post_trigger_cap,
                guard.severe_weighted_gap_loss,
                guard.severe_post_trigger_cap,
            )
            desired = transfer_semis(
                reference,
                guard.semis_multiplier,
                guard.overflow_asset,
                selected_trigger_cap,
            )
            trade_reason = "trigger"
            emergency_trade = True
            active = True
            current_guard_cap = selected_trigger_cap
            current_overflow_asset = guard.overflow_asset
            active_days = 0
            positive_relative_days = 0
        elif active and base_trade and desired is not None:
            desired = transfer_semis(
                desired,
                guard.semis_multiplier,
                current_overflow_asset,
                current_guard_cap,
            )

        turnover = (
            0.0
            if desired is None
            else 0.5 * float(np.abs(desired - current).sum())
        )
        if turnover > 0.0 and desired is not None:
            current = desired
        trading_cost = 2.0 * turnover * cost_bps / 10_000.0
        slippage_cost = (
            2.0
            * turnover
            * guard.emergency_slippage_bps
            / 10_000.0
            if emergency_trade
            else 0.0
        )

        intraday_asset_returns = intraday_asset_return_values[position]
        intraday_return = float(current @ intraday_asset_returns)
        financing_cost = (
            max(-float(current[ASSETS.index("CASH")]), 0.0)
            * 100.0
            / 10_000.0
            / 252.0
        )
        net_return = (
            (1.0 + overnight_return) * (1.0 + intraday_return)
            - 1.0
            - trading_cost
            - slippage_cost
            - financing_cost
        )
        rows.append(
            {
                "net_return": net_return,
                "overnight_return": overnight_return,
                "intraday_return": intraday_return,
                "trading_cost": trading_cost,
                "slippage_cost": slippage_cost,
                "turnover": turnover,
                "semis_gap": semis_gap,
                "qqq_gap": qqq_gap,
                "relative_gap": relative_gap,
                "trigger_semis_gap": trigger_semis_gap,
                "trigger_relative_gap": trigger_relative_gap,
                "weighted_gap_loss": weighted_gap_loss,
                "prior_weighted_loss": prior_weighted_loss,
                "trigger_source": trigger_source or "none",
                "prior_relative_return": prior_relative_return,
                "prior_semis_return": prior_semis_return,
                "prior_qqq_return": prior_qqq_return,
                "base_trade": base_trade,
                "triggered": triggered,
                "deepened": deepened,
                "bridged": bridged,
                "active": active,
                "recovered": recovered,
                "overflow_asset": current_overflow_asset,
                "trade_reason": trade_reason,
            }
        )
        weight_rows.append(current.copy())

        gross_intraday_growth = 1.0 + intraday_return
        if gross_intraday_growth <= 0.0:
            raise ValueError("Portfolio lost all capital intraday")
        current = (
            current * (1.0 + intraday_asset_returns)
            / gross_intraday_growth
        )
        previous_date = date

    return (
        pd.DataFrame(rows, index=dates),
        pd.DataFrame(weight_rows, index=dates, columns=ASSETS),
    )


def positive_capture(candidate: pd.Series, baseline: pd.Series) -> float:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    positive = aligned["baseline"] > 0.0
    denominator = float(aligned.loc[positive, "baseline"].sum())
    if denominator <= 1e-12:
        return float("nan")
    return float(aligned.loc[positive, "candidate"].sum() / denominator)


def metrics(
    guard: OpenGapGuard,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    baseline: pd.Series,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for period, (start, end) in PERIODS.items():
        selected = daily.loc[start:end]
        reference = baseline.reindex(selected.index)
        values = performance_metrics(selected["net_return"])
        rows.append(
            {
                "variant": guard.name,
                "period": period,
                **values,
                "worst_day": float(selected["net_return"].min()),
                "positive_capture": positive_capture(
                    selected["net_return"],
                    reference,
                ),
                "maximum_semis_weight": float(
                    weights.loc[selected.index, "SEMIS"].max()
                ),
                "annualized_turnover": float(
                    selected["turnover"].mean() * 252.0
                ),
                "annualized_trading_cost": float(
                    (
                        selected["trading_cost"]
                        + selected["slippage_cost"]
                    ).mean()
                    * 252.0
                ),
                "trigger_count": int(selected["triggered"].sum()),
                "recovery_count": int(selected["recovered"].sum()),
            }
        )
    return rows


def add_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index(["variant", "period"])
    for column in (
        "cagr",
        "max_drawdown",
        "worst_day",
        "positive_capture",
    ):
        baseline = indexed.xs("baseline", level="variant")[column]
        indexed[f"{column}_delta_vs_baseline"] = [
            float(row[column] - baseline.loc[period])
            for (_, period), row in indexed.iterrows()
        ]
    return indexed.reset_index()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    weights, base_daily, opens, closes = load_inputs()
    all_metrics: list[dict[str, float | int | str]] = []
    event_frames: list[pd.DataFrame] = []
    baseline_returns: pd.Series | None = None
    reconstruction_rows: list[dict[str, float]] = []
    reference_open_execution = simulate_open_execution(
        STRATEGY,
        opens,
        closes,
        start_date="2015-01-01",
        end_date=None,
    )

    for guard in variants():
        daily, candidate_weights = simulate(
            weights,
            base_daily,
            opens,
            closes,
            guard,
        )
        if guard.name == "baseline":
            baseline_returns = daily["net_return"].copy()
        if baseline_returns is None:
            raise RuntimeError("Baseline must be evaluated first")
        all_metrics.extend(
            metrics(
                guard,
                daily,
                candidate_weights,
                baseline_returns,
            )
        )
        if guard.name == "baseline":
            common = daily.index.intersection(reference_open_execution.index)
            reconstruction_rows.append(
                {
                    "maximum_return_difference_vs_existing_engine": float(
                        (
                            daily.loc[common, "net_return"]
                            - reference_open_execution.loc[
                                common,
                                "net_return",
                            ]
                        )
                        .abs()
                        .max()
                    )
                }
            )
        triggered = daily.loc[daily["triggered"]].copy()
        if not triggered.empty:
            triggered["variant"] = guard.name
            triggered["semis_weight"] = candidate_weights.loc[
                triggered.index,
                "SEMIS",
            ]
            event_frames.append(triggered.reset_index(names="date"))

    metric_frame = add_deltas(pd.DataFrame(all_metrics))
    events = pd.concat(event_frames, ignore_index=True)
    reconstruction = pd.DataFrame(reconstruction_rows)
    metric_frame.to_csv(DESTINATION / "metrics.csv", index=False)
    events.to_csv(DESTINATION / "events.csv", index=False)
    reconstruction.to_csv(
        DESTINATION / "reconstruction_check.csv",
        index=False,
    )

    holdout = metric_frame.loc[
        metric_frame["period"] == "holdout_2022_2025"
    ].set_index("variant")
    development = metric_frame.loc[
        metric_frame["period"] == "development_2015_2021"
    ].set_index("variant")
    summary = holdout[
        [
            "cagr_delta_vs_baseline",
            "max_drawdown_delta_vs_baseline",
            "worst_day_delta_vs_baseline",
            "positive_capture_delta_vs_baseline",
            "trigger_count",
            "annualized_turnover",
        ]
    ].join(
        development[
            [
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
            ]
        ].rename(
            columns={
                "cagr_delta_vs_baseline": "development_cagr_delta",
                "max_drawdown_delta_vs_baseline": "development_mdd_delta",
            }
        )
    )
    print("Reconstruction:")
    print(reconstruction.to_string(index=False))
    print("\nTop holdout candidates:")
    print(
        summary.sort_values(
            [
                "worst_day_delta_vs_baseline",
                "cagr_delta_vs_baseline",
            ],
            ascending=False,
        )
        .head(30)
        .round(6)
        .to_string()
    )
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()

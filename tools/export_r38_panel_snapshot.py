from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regime_strategy.operations import next_us_equity_session


ROOT = Path(__file__).resolve().parents[1]
R9_OUTPUT = (
    ROOT
    / "output/paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
)
R38_OUTPUT = (
    ROOT / "output/paper_core_growth_gold20_r38_convex_overlay"
)
R39_OUTPUT = (
    ROOT
    / "output/paper_core_growth_gold20_r39_relative_damage_veto"
)
R38_QUALIFICATION = ROOT / "output/r38_production_qualification_audit/summary.json"
R39_RESEARCH_QUALIFICATION = ROOT / "output/r39_final_candidate_audit/summary.json"
R39_PRODUCTION_QUALIFICATION = ROOT / "output/r39_production_qualification_audit/summary.json"
R40_OUTPUT = ROOT / "output/paper_core_growth_gold20_r40_recursive_trend_cushion/account_protection_spec.json"
R40_QUALIFICATION = ROOT / "output/r40_production_qualification_audit/summary.json"
R41_OUTPUT = ROOT / "output/paper_core_growth_gold20_r41_pput_protected_capacity/protection_spec.json"
R41_QUALIFICATION = ROOT / "output/r41_production_qualification_audit/summary.json"
ANTI_OVERFIT_GOVERNANCE = ROOT / "output/r38_anti_overfit_governance/summary.json"
DECISION_AUTHORITY_OUTPUT = (
    ROOT
    / "output/paper_core_growth_gold20_r11_decision_authority"
)
MARKET_PRICES = ROOT / "data/prices_vix_hedge.csv"
CORE_METADATA = (
    ROOT / "data/adjusted_open_close_2011_present.csv.metadata.json"
)
GDE_PRICES = ROOT / "data/retail_alternatives_open_close.csv"
DESTINATION = ROOT / "strategy-panel/app/strategy-live-data.ts"


def scalar_diagnostics(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, index_col=0)
    return {
        str(index): str(value)
        for index, value in frame.iloc[:, 0].items()
    }


def weights(frame: pd.DataFrame, column: str) -> dict[str, float]:
    selected = frame[column].astype(float)
    return {
        "QQQ": float(selected["QQQ"]),
        "SMH": float(selected["SEMIS"]),
        "GLD": float(selected["GOLD"]),
        "GDE": float(selected["GDE"]),
        "cash": float(selected["CASH"]),
        "VIXY": float(selected["VIX_HEDGE"]),
    }


def read_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def effective_execution_window_status(
    declared_status: str,
    next_open_et: pd.Timestamp,
    *,
    now_utc: pd.Timestamp | None = None,
) -> str:
    current = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if current.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    if current >= next_open_et.tz_convert("UTC"):
        return "MISSED"
    return declared_status


def r40_eligibility(
    *,
    qualification_pass: bool,
    r40_price_as_of: str,
    active_price_as_of: str,
    successor_promotion_allowed: bool,
) -> dict[str, bool]:
    same_date = r40_price_as_of == active_price_as_of
    return {
        "productionQualificationPass": qualification_pass,
        "promotionAllowed": successor_promotion_allowed,
        "draftEligible": qualification_pass and same_date,
        "productionEligible": (
            qualification_pass and successor_promotion_allowed
        ),
        "sameDate": same_date,
    }


def panel_status(
    *,
    use_r39: bool,
    active_release: str,
    active_price_as_of: pd.Timestamp,
) -> dict[str, object]:
    r38_qualification = read_optional_json(R38_QUALIFICATION)
    r39_research = read_optional_json(R39_RESEARCH_QUALIFICATION)
    r39_production = read_optional_json(R39_PRODUCTION_QUALIFICATION)
    r39_metadata = read_optional_json(R39_OUTPUT / "run_metadata.json")
    governance = read_optional_json(ANTI_OVERFIT_GOVERNANCE)
    successor_promotion_allowed = bool(
        governance.get("successor_promotion_allowed", False)
    )
    r39_price = str(r39_metadata.get("price_as_of", "unavailable"))
    r39_research_pass = bool(r39_research.get("qualification_pass", False))
    r39_production_pass = bool(
        r39_production.get("production_qualification_pass", False)
    )
    r39_current = r39_price == active_price_as_of.date().isoformat()
    if use_r39 and not successor_promotion_allowed:
        raise ValueError("anti-overfit governance blocks R39 promotion")
    if use_r39:
        return {
            "requestedStrategy": "R39",
            "activeStrategy": "R39",
            "activeRelease": active_release,
            "fallbackActive": False,
            "fallbackReason": None,
            "activeQualification": (
                f"{int(r39_production.get('requirements_passed', 0))} / "
                f"{int(r39_production.get('requirements', 0))}"
            ),
            "r39ResearchQualificationPass": r39_research_pass,
            "r39ProductionQualificationPass": r39_production_pass,
            "r39PriceAsOf": r39_price,
            "successorPromotionAllowed": successor_promotion_allowed,
        }
    if not successor_promotion_allowed:
        reason = (
            "R38 已进入反过拟合前瞻冻结期；当前只维持 75% R11 + 25% R38，"
            "R39 至 R41 不参与生产晋级。"
        )
    elif not r39_research_pass:
        reason = (
            "R39 的最新研究资格审计未通过，已自动使用同日合格的 R38。"
        )
    elif not r39_current:
        reason = (
            f"R39 快照仅截至 {r39_price}，已自动使用截至 "
            f"{active_price_as_of.date().isoformat()} 的合格 R38。"
        )
    elif not r39_production_pass:
        reason = "R39 的生产资格审计未通过，已自动使用同日合格的 R38。"
    else:
        reason = "当前刷新明确选择 R38，未启用 R39 账户保护层。"
    return {
        "requestedStrategy": "R39",
        "activeStrategy": "R38",
        "activeRelease": active_release,
        "fallbackActive": True,
        "fallbackReason": reason,
        "activeQualification": (
            f"{int(r38_qualification.get('requirements_passed', 0))} / "
            f"{int(r38_qualification.get('requirements', 0))}"
        ),
        "r39ResearchQualificationPass": r39_research_pass,
        "r39ProductionQualificationPass": r39_production_pass,
        "r39PriceAsOf": r39_price,
        "successorPromotionAllowed": successor_promotion_allowed,
    }


def main(*, use_r39: bool = False) -> None:
    r9_metadata = json.loads(
        (R9_OUTPUT / "run_metadata.json").read_text(encoding="utf-8")
    )
    r38_metadata = json.loads(
        (R38_OUTPUT / "run_metadata.json").read_text(encoding="utf-8")
    )
    r39_metadata = (
        json.loads(
            (R39_OUTPUT / "run_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        if use_r39
        else None
    )
    core_metadata = json.loads(
        CORE_METADATA.read_text(encoding="utf-8")
    )
    r40_spec = read_optional_json(R40_OUTPUT)
    r40_qualification = read_optional_json(R40_QUALIFICATION)
    r41_spec = read_optional_json(R41_OUTPUT)
    r41_qualification = read_optional_json(R41_QUALIFICATION)
    governance = read_optional_json(ANTI_OVERFIT_GOVERNANCE)
    successor_promotion_allowed = bool(
        governance.get("successor_promotion_allowed", False)
    )
    r9_diagnostics = pd.read_csv(
        R9_OUTPUT / "next_signal_diagnostics.csv",
        index_col=0,
    )
    r38_diagnostics = scalar_diagnostics(
        R38_OUTPUT / "next_signal_diagnostics.csv"
    )
    r39_diagnostics = (
        scalar_diagnostics(
            R39_OUTPUT / "next_signal_diagnostics.csv"
        )
        if use_r39
        else None
    )
    targets = pd.read_csv(
        (
            R39_OUTPUT
            if use_r39
            else R38_OUTPUT
        )
        / "next_target_weights.csv",
        index_col="asset",
    )
    decision_authority = json.loads(
        (
            DECISION_AUTHORITY_OUTPUT / "decision_authority.json"
        ).read_text(encoding="utf-8")
    )
    market = pd.read_csv(
        MARKET_PRICES,
        index_col="date",
        parse_dates=True,
    )
    gde = pd.read_csv(
        GDE_PRICES,
        index_col="date",
        parse_dates=True,
    )
    release_metadata = r39_metadata or r38_metadata
    price_as_of = pd.Timestamp(release_metadata["price_as_of"])
    r40_price_as_of = str(r40_spec.get("price_as_of", "unavailable"))
    r40_status = r40_eligibility(
        qualification_pass=bool(
            r40_qualification.get("production_qualification_pass", False)
        ),
        r40_price_as_of=r40_price_as_of,
        active_price_as_of=price_as_of.date().isoformat(),
        successor_promotion_allowed=successor_promotion_allowed,
    )
    latest = market.loc[price_as_of]
    next_session = pd.Timestamp(release_metadata["next_session"])
    next_open_et = (
        next_session.tz_localize("America/New_York")
        + pd.Timedelta(hours=9, minutes=30)
    )
    next_open_jst = next_open_et.tz_convert("Asia/Tokyo")
    after_fill_review_et = (
        next_session.tz_localize("America/New_York")
        + pd.Timedelta(hours=16, minutes=15)
    )
    after_fill_review_jst = after_fill_review_et.tz_convert("Asia/Tokyo")
    following_session = next_us_equity_session(next_session)
    following_open_et = (
        following_session.tz_localize("America/New_York")
        + pd.Timedelta(hours=9, minutes=30)
    )
    following_open_jst = following_open_et.tz_convert("Asia/Tokyo")
    generated_at_utc = pd.Timestamp(
        release_metadata["generated_at_utc"]
    )
    generated_at_jst = generated_at_utc.tz_convert("Asia/Tokyo")
    execution_window_status = effective_execution_window_status(
        str(
            release_metadata.get(
                "execution_window_status",
                "UPCOMING",
            )
        ),
        next_open_et,
    )
    actions = sorted(set(r9_diagnostics.loc["action"].astype(str)))
    if len(actions) != 1:
        raise RuntimeError(f"R9 action disagreement: {actions}")
    member_votes = []
    for path in sorted(
        (R9_OUTPUT / "members").glob("seed_*/regimes.csv")
    ):
        regimes = pd.read_csv(path, index_col=0)
        member_votes.append(
            int(regimes.iloc[-1]["risk_on_leverage"] > 0)
        )

    staged_prefix = "r39_account" if use_r39 else "staged_account"
    staged = weights(targets, f"{staged_prefix}_center")
    payload = {
        "release": str(release_metadata["release"]),
        "parentRelease": str(
            release_metadata.get(
                "parent_release",
                r38_metadata["release"],
            )
        ),
        "generatedAtUtc": generated_at_utc.isoformat(),
        "generatedAtJst": generated_at_jst.strftime(
            "%Y-%m-%d %H:%M 日本时间"
        ),
        "priceAsOf": price_as_of.date().isoformat(),
        "expectedPriceAsOf": str(
            release_metadata["expected_price_as_of"]
        ),
        "dataIsFresh": bool(release_metadata["data_is_fresh"]),
        "generationWindowStatus": str(
            release_metadata["generation_window_status"]
        ),
        "generationMode": str(
            release_metadata.get(
                "generation_mode",
                "STANDARD_REFRESH",
            )
        ),
        "executionWindowStatus": execution_window_status,
        "nextSession": next_session.date().isoformat(),
        "nextExecutionEt": next_open_et.strftime(
            "%Y-%m-%d %H:%M 美东时间"
        ),
        "nextExecutionJst": next_open_jst.strftime(
            "%Y-%m-%d %H:%M 日本时间"
        ),
        "nextExecutionWeekdayZh": "周"
        + "一二三四五六日"[next_open_et.weekday()],
        "panelStatus": panel_status(
            use_r39=use_r39,
            active_release=str(release_metadata["release"]),
            active_price_as_of=price_as_of,
        ),
        "overfitGovernance": {
            "release": str(governance.get("release", "unavailable")),
            "operationalPass": bool(
                governance.get("operational_pass", False)
            ),
            "forwardSessions": int(governance.get("forward_sessions", 0)),
            "minimumForwardSessions": int(
                governance.get("minimum_forward_sessions", 63)
            ),
            "forwardReviewEligible": bool(
                governance.get("forward_review_eligible", False)
            ),
            "currentR38Share": float(
                governance.get("current_r38_share", 0.25)
            ),
            "successorPromotionAllowed": successor_promotion_allowed,
            "blockedSuccessors": list(
                governance.get("blocked_successors", ["R39", "R40", "R41"])
            ),
            "historicalEvidenceClassification": str(
                governance.get(
                    "historical_evidence_classification",
                    "IN_SAMPLE_REUSED",
                )
            ),
            "forwardEvidenceUse": str(
                governance.get("forward_evidence_use", "GO_NO_GO_ONLY")
            ),
        },
        "accountBuildTiming": {
            "requiredNewCompletedCloses": 1,
            "afterCurrentTrancheReviewEt": (
                after_fill_review_et.strftime(
                    "%Y-%m-%d %H:%M 美东时间"
                )
            ),
            "afterCurrentTrancheReviewJst": (
                after_fill_review_jst.strftime(
                    "%Y-%m-%d %H:%M 日本时间"
                )
            ),
            "followingTrancheEarliestExecutionEt": (
                following_open_et.strftime(
                    "%Y-%m-%d %H:%M 美东时间"
                )
            ),
            "followingTrancheEarliestExecutionJst": (
                following_open_jst.strftime(
                    "%Y-%m-%d %H:%M 日本时间"
                )
            ),
        },
        "estimatedRebalanceDate": str(
            r9_diagnostics.loc[
                "estimated_rebalance_date"
            ].iloc[0]
        ),
        "sessionsToRebalance": int(
            float(
                r9_diagnostics.loc[
                    "sessions_to_rebalance"
                ].iloc[0]
            )
        ),
        "r9Action": actions[0],
        "riskOnVotes": sum(member_votes),
        "riskOnVoteTotal": len(member_votes),
        "vix": float(latest["VIX"]),
        "vix3m": float(latest["VIX3M"]),
        "vixTermRatio": float(latest["VIX"] / latest["VIX3M"]),
        "referencePrices": {
            "QQQ": float(latest["QQQ"]),
            "SMH": float(latest["SEMIS"]),
            "GLD": float(latest["GOLD"]),
            "GDE": float(gde.loc[price_as_of, "close_GDE"]),
            "BIL": float(latest["CASH"]),
            "VIXY": float(latest["VIX_HEDGE"]),
        },
        "r11Reference": weights(targets, "r11_reference_target"),
        "fullTarget": weights(targets, "r38_full_center"),
        "parentStagedTarget": weights(
            targets,
            "r38_staged_account_center",
        ) if use_r39 else staged,
        "stagedTarget": staged,
        "stagedLower": {
            "GDE": float(
                targets.loc["GDE", f"{staged_prefix}_lower"]
            ),
        },
        "stagedUpper": {
            "GDE": float(
                targets.loc["GDE", f"{staged_prefix}_upper"]
            ),
        },
        "goldSleeveTarget": staged["GLD"] + staged["GDE"],
        "riskBudget": {
            "trendPermission": bool(
                int(float(r38_diagnostics["trend_permission"]))
            ),
            "oneSessionShockActive": bool(
                int(
                    float(
                        r38_diagnostics[
                            "one_session_shock_active"
                        ]
                    )
                )
            ),
            "volatilityAccelerationBlock": bool(
                int(
                    float(
                        r38_diagnostics[
                            "volatility_acceleration_block"
                        ]
                    )
                )
            ),
            "stateActiveMultiplier": float(
                r38_diagnostics["state_active_multiplier"]
            ),
            "acceptedAbsoluteMultiplier": float(
                r38_diagnostics["accepted_absolute_multiplier"]
            ),
            "cashHardLimitActive": bool(
                int(
                    float(
                        r38_diagnostics[
                            "cash_hard_limit_active"
                        ]
                    )
                )
            ),
            "fullPreGdeCashWeight": float(
                r38_diagnostics["r38_pre_gde_cash_weight"]
            ),
        },
        "recursiveTrendCushion": {
            "release": str(r40_spec.get("release", "not-built")),
            **r40_status,
            "priceAsOf": r40_price_as_of,
            "floorDrawdown": float(
                dict(r40_spec.get("parameters", {})).get("floor_drawdown", -0.19)
            ),
            "bullMultiplier": float(
                dict(r40_spec.get("parameters", {})).get("bull_multiplier", 100.0)
            ),
            "bearMultiplier": float(
                dict(r40_spec.get("parameters", {})).get("bear_multiplier", 9.5)
            ),
            "tierSize": float(
                dict(r40_spec.get("parameters", {})).get("tier_size", 0.05)
            ),
            "dualTrendPositive": bool(
                r40_spec.get("dual_trend_positive", False)
            ),
            "historicalPre2008MaxDrawdown": float(
                r40_qualification.get("historical_pre2008_max_drawdown", 0.0)
            ),
            "modernCagr": float(r40_qualification.get("modern_cagr", 0.0)),
            "modernCagrDelta": float(
                r40_qualification.get("modern_cagr_delta", 0.0)
            ),
            "proxyCagrDelta": float(
                r40_qualification.get("proxy_cagr_delta", 0.0)
            ),
        },
        "pputProtectedCapacity": {
            "release": str(r41_spec.get("release", "not-built")),
            "productionEligible": bool(
                r41_qualification.get("production_qualification_pass", False)
            ) and successor_promotion_allowed,
            "active": False,
            "priceAsOf": str(r41_spec.get("price_as_of", "unavailable")),
            "sameDate": str(r41_spec.get("price_as_of", "unavailable"))
            == price_as_of.date().isoformat(),
            "targetCoverage": float(
                dict(r41_spec.get("parameters", {})).get("overlay_notional", 0.06)
            ),
            "minimumCoverage": 0.05,
            "maximumCoverage": 0.07,
            "normalNonCashCap": float(
                dict(r41_spec.get("parameters", {})).get("normal_non_cash_cap", 1.0)
            ),
            "floorDrawdown": float(
                dict(r41_spec.get("parameters", {})).get("floor_drawdown", -0.19)
            ),
            "bearMultiplier": float(
                dict(r41_spec.get("parameters", {})).get("bear_multiplier", 20.0)
            ),
            "tierSize": float(
                dict(r41_spec.get("parameters", {})).get("tier_size", 0.05)
            ),
            "instrument": str(
                dict(r41_spec.get("protection", {})).get(
                    "production_instrument", "SPYM listed long put"
                )
            ),
            "strikeRule": str(
                dict(r41_spec.get("protection", {})).get("strike_rule", "")
            ),
            "expirationRule": str(
                dict(r41_spec.get("protection", {})).get("expiration_rule", "")
            ),
            "modernCagr": float(r41_qualification.get("modern_cagr", 0.0)),
            "modernMaxDrawdown": float(
                r41_qualification.get("modern_max_drawdown", 0.0)
            ),
            "proxyCagr": float(r41_qualification.get("proxy_cagr", 0.0)),
            "historicalPre2008MaxDrawdown": float(
                r41_qualification.get("historical_pre2008_max_drawdown", 0.0)
            ),
            "example500kContracts": int(
                r41_qualification.get("example_500k_contracts", 0)
            ),
            "example500kCoverage": float(
                r41_qualification.get("example_500k_coverage", 0.0)
            ),
            "activationRule": str(r41_qualification.get("activation_rule", "")),
            "orderBlockers": list(r41_spec.get("order_blockers", [])),
        },
        "semiconductorOverlay": {
            "baseShare": float(
                r38_diagnostics["base_semis_growth_share"]
            ),
            "fullSignalShare": float(
                r38_diagnostics[
                    "full_signal_semis_growth_share"
                ]
            ),
            "implementedShare": float(
                r38_diagnostics["r38_semis_growth_share"]
            ),
            "overlayFraction": float(
                r38_diagnostics["overlay_fraction"]
            ),
        },
        "relativeDamageVeto": (
            {
                "active": bool(
                    int(
                        float(
                            r39_diagnostics[
                                "relative_damage_veto_active"
                            ]
                        )
                    )
                ),
                "lookbackTradingDays": int(
                    float(
                        r39_diagnostics[
                            "lookback_trading_days"
                        ]
                    )
                ),
                "relativeLoss": float(
                    r39_diagnostics["relative_loss"]
                ),
                "accountRelativeLossBudget": float(
                    r39_diagnostics[
                        "account_relative_loss_budget"
                    ]
                ),
                "proposedAccountRelativeLoss": float(
                    r39_diagnostics[
                        "proposed_account_relative_loss"
                    ]
                ),
                "implementedAccountRelativeLoss": float(
                    r39_diagnostics[
                        "implemented_account_relative_loss"
                    ]
                ),
                "baseSemisGrowthShare": float(
                    r39_diagnostics[
                        "base_semis_growth_share"
                    ]
                ),
                "implementedSemisGrowthShare": float(
                    r39_diagnostics[
                        "implemented_semis_growth_share"
                    ]
                ),
                "triggerMinimumSemisGrowthShare": float(
                    r39_diagnostics[
                        "minimum_smh_share_of_growth_sleeve"
                    ]
                ),
                "maximumActiveSemisGrowthShare": float(
                    r39_diagnostics[
                        "maximum_active_smh_share_of_growth_sleeve"
                    ]
                ),
                "maximumShareActivationMultiple": float(
                    r39_diagnostics[
                        "maximum_share_activation_multiple"
                    ]
                ),
                "maximumShareGuardPermitted": bool(
                    int(
                        float(
                            r39_diagnostics[
                                "maximum_share_guard_permitted"
                            ]
                        )
                    )
                ),
                "maximumShareGuardActive": bool(
                    int(
                        float(
                            r39_diagnostics[
                                "maximum_share_guard_active"
                            ]
                        )
                    )
                ),
                "removedSemisWeight": float(
                    r39_diagnostics["removed_semis_weight"]
                ),
                "growthBudgetBefore": float(
                    r39_diagnostics["growth_budget_before"]
                ),
                "growthBudgetAfter": float(
                    r39_diagnostics["growth_budget_after"]
                ),
            }
            if r39_diagnostics
            else {
                "active": False,
                "lookbackTradingDays": 21,
                "relativeLoss": 0.0,
                "accountRelativeLossBudget": 0.03,
                "proposedAccountRelativeLoss": 0.0,
                "implementedAccountRelativeLoss": 0.0,
                "baseSemisGrowthShare": float(
                    r38_diagnostics["r38_semis_growth_share"]
                ),
                "implementedSemisGrowthShare": float(
                    r38_diagnostics["r38_semis_growth_share"]
                ),
                "triggerMinimumSemisGrowthShare": 0.6,
                "maximumActiveSemisGrowthShare": 0.5,
                "maximumShareActivationMultiple": 1.1,
                "maximumShareGuardPermitted": False,
                "maximumShareGuardActive": False,
                "removedSemisWeight": 0.0,
                "growthBudgetBefore": staged["QQQ"] + staged["SMH"],
                "growthBudgetAfter": staged["QQQ"] + staged["SMH"],
            }
        ),
        "smhGuardTriggered": bool(
            int(float(r38_diagnostics["smh_guard_triggered"]))
        ),
        "smhGuardActive": bool(
            int(float(r38_diagnostics["smh_guard_active"]))
        ),
        "smhSignalGap": float(r38_diagnostics["smh_signal_gap"]),
        "smhRelativeSignalGap": float(
            r38_diagnostics["smh_relative_signal_gap"]
        ),
        "coreAndGdeAligned": (
            r38_diagnostics["core_price_as_of"]
            == r38_diagnostics["gde_price_as_of"]
            == price_as_of.date().isoformat()
        ),
        "recentSessionsContinuous": True,
        "flatProxyBars": core_metadata.get("flat_proxy_bars", {}),
        "ordersExecutable": bool(
            release_metadata["orders_executable"]
        ),
        "orderBlockers": list(
            release_metadata["order_blockers"]
        ),
        "sourceFreshness": {
            "R9": bool(r9_metadata["data_is_fresh"]),
            "R38": bool(r38_metadata["data_is_fresh"]),
            "R39": bool(
                r39_metadata["data_is_fresh"]
                if r39_metadata
                else False
            ),
        },
        "decisionAuthority": {
            "release": decision_authority["release"],
            "marketEnvironment": decision_authority[
                "market_environment"
            ],
            "marketEnvironmentInformationalOnly": decision_authority[
                "market_environment_informational_only"
            ],
            "qqq63Return": decision_authority["environment_inputs"][
                "qqq_63d_return"
            ],
            "semis63Return": decision_authority[
                "environment_inputs"
            ]["semis_63d_return"],
            "semis126Return": decision_authority[
                "environment_inputs"
            ]["semis_126d_return"],
            "vixTermRatio": decision_authority[
                "environment_inputs"
            ]["vix_term_ratio"],
            "unapprovedRealCapitalShare": decision_authority[
                "unapproved_real_capital_share"
            ],
            "stateChangeOrdersAllowed": decision_authority[
                "state_change_orders_allowed"
            ],
            "maximumTargetChange": decision_authority[
                "maximum_target_change_from_authority"
            ],
        },
    }
    DESTINATION.write_text(
        "export const strategyLiveData = "
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + " as const;\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "destination": str(DESTINATION),
                "release": payload["release"],
                "price_as_of": payload["priceAsOf"],
                "next_session": payload["nextSession"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from regime_strategy.operations import next_us_equity_session


ROOT = Path(__file__).resolve().parents[1]
R9_OUTPUT = ROOT / "output/paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
R11_OUTPUT = ROOT / "output/paper_core_growth_gold20_r11_diversified_financing"
REENTRY_BRAKE_OUTPUT = (
    ROOT
    / "output/paper_core_growth_gold20_r11_reentry_brake_shadow"
)
DECISION_AUTHORITY_OUTPUT = (
    ROOT
    / "output/paper_core_growth_gold20_r11_decision_authority"
)
MARKET_PRICES = ROOT / "data/prices_vix_hedge.csv"
CORE_METADATA = ROOT / "data/adjusted_open_close_2011_present.csv.metadata.json"
GDE_PRICES = ROOT / "data/retail_alternatives_open_close.csv"
DESTINATION = ROOT / "strategy-panel/app/strategy-live-data.ts"


def scalar_diagnostics(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, index_col=0)
    return {
        str(index): str(value)
        for index, value in frame.iloc[:, 0].items()
    }


def main() -> None:
    r9_metadata = json.loads(
        (R9_OUTPUT / "run_metadata.json").read_text(encoding="utf-8")
    )
    r11_metadata = json.loads(
        (R11_OUTPUT / "run_metadata.json").read_text(encoding="utf-8")
    )
    core_metadata = json.loads(CORE_METADATA.read_text(encoding="utf-8"))
    r9_diagnostics = pd.read_csv(
        R9_OUTPUT / "next_signal_diagnostics.csv",
        index_col=0,
    )
    r11_diagnostics = scalar_diagnostics(
        R11_OUTPUT / "next_signal_diagnostics.csv"
    )
    reentry_diagnostics = scalar_diagnostics(
        REENTRY_BRAKE_OUTPUT / "next_signal_diagnostics.csv"
    )
    reentry_targets = pd.read_csv(
        REENTRY_BRAKE_OUTPUT / "next_target_weights.csv",
        index_col="asset",
    )
    decision_authority = json.loads(
        (DECISION_AUTHORITY_OUTPUT / "decision_authority.json").read_text(
            encoding="utf-8"
        )
    )
    targets = pd.read_csv(
        R11_OUTPUT / "next_target_weights.csv",
        index_col="asset",
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

    price_as_of = pd.Timestamp(r11_metadata["price_as_of"])
    latest = market.loc[price_as_of]
    next_session = pd.Timestamp(r11_metadata["next_session"])
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
    generated_at_utc = pd.Timestamp(r11_metadata["generated_at_utc"])
    generated_at_jst = generated_at_utc.tz_convert("Asia/Tokyo")
    actions = sorted(set(r9_diagnostics.loc["action"].astype(str)))
    if len(actions) != 1:
        raise RuntimeError(f"R9 action disagreement: {actions}")

    member_votes = []
    for path in sorted((R9_OUTPUT / "members").glob("seed_*/regimes.csv")):
        regimes = pd.read_csv(path, index_col=0)
        member_votes.append(int(regimes.iloc[-1]["risk_on_leverage"] > 0))

    staged_center = targets["staged_account_center"].astype(float)
    staged_lower = targets["staged_account_lower"].astype(float)
    staged_upper = targets["staged_account_upper"].astype(float)
    r9_reference = targets["r9_reference_target"].astype(float)
    gold_sleeve_target = float(
        staged_center["GOLD"] + staged_center["GDE"]
    )
    payload = {
        "generatedAtUtc": generated_at_utc.isoformat(),
        "generatedAtJst": generated_at_jst.strftime(
            "%Y-%m-%d %H:%M 日本时间"
        ),
        "priceAsOf": price_as_of.date().isoformat(),
        "expectedPriceAsOf": str(r11_metadata["expected_price_as_of"]),
        "dataIsFresh": bool(r11_metadata["data_is_fresh"]),
        "generationWindowStatus": str(
            r11_metadata["generation_window_status"]
        ),
        "nextSession": next_session.date().isoformat(),
        "nextExecutionEt": next_open_et.strftime(
            "%Y-%m-%d %H:%M 美东时间"
        ),
        "nextExecutionJst": next_open_jst.strftime(
            "%Y-%m-%d %H:%M 日本时间"
        ),
        "nextExecutionWeekdayZh": "周"
        + "一二三四五六日"[next_open_et.weekday()],
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
            r9_diagnostics.loc["estimated_rebalance_date"].iloc[0]
        ),
        "sessionsToRebalance": int(
            float(r9_diagnostics.loc["sessions_to_rebalance"].iloc[0])
        ),
        "r9Action": actions[0],
        "riskOnVotes": sum(member_votes),
        "riskOnVoteTotal": len(member_votes),
        "vix": float(latest["VIX"]),
        "vix3m": float(latest["VIX3M"]),
        "vixTermRatio": float(latest["VIX"] / latest["VIX3M"]),
        "vixHedgeWeight": float(staged_center["VIX_HEDGE"]),
        "smhGuardTriggered": bool(
            int(float(r11_diagnostics["smh_guard_triggered"]))
        ),
        "smhGuardActive": bool(
            int(float(r11_diagnostics["smh_guard_active"]))
        ),
        "smhSignalGap": float(r11_diagnostics["smh_signal_gap"]),
        "smhRelativeSignalGap": float(
            r11_diagnostics["smh_relative_signal_gap"]
        ),
        "referencePrices": {
            "QQQ": float(latest["QQQ"]),
            "SMH": float(latest["SEMIS"]),
            "GLD": float(latest["GOLD"]),
            "GDE": float(gde.loc[price_as_of, "close_GDE"]),
            "BIL": float(latest["CASH"]),
            "VIXY": float(latest["VIX_HEDGE"]),
        },
        "r9Reference": {
            "QQQ": float(r9_reference["QQQ"]),
            "SMH": float(r9_reference["SEMIS"]),
            "GLD": float(r9_reference["GOLD"]),
            "GDE": float(r9_reference["GDE"]),
            "cash": float(r9_reference["CASH"]),
        },
        "stagedTarget": {
            "QQQ": float(staged_center["QQQ"]),
            "SMH": float(staged_center["SEMIS"]),
            "GLD": float(staged_center["GOLD"]),
            "GDE": float(staged_center["GDE"]),
            "cash": float(staged_center["CASH"]),
        },
        "stagedLower": {
            "GDE": float(staged_lower["GDE"]),
        },
        "stagedUpper": {
            "GDE": float(staged_upper["GDE"]),
        },
        "goldSleeveTarget": gold_sleeve_target,
        "coreAndGdeAligned": (
            r11_diagnostics["core_price_as_of"]
            == r11_diagnostics["gde_price_as_of"]
            == price_as_of.date().isoformat()
        ),
        "recentSessionsContinuous": True,
        "flatProxyBars": core_metadata.get("flat_proxy_bars", {}),
        "ordersExecutable": bool(r11_metadata["orders_executable"]),
        "orderBlockers": list(r11_metadata["order_blockers"]),
        "sourceFreshness": {
            "R9": bool(r9_metadata["data_is_fresh"]),
            "R11": bool(r11_metadata["data_is_fresh"]),
        },
        "decisionAuthority": {
            "release": decision_authority["release"],
            "marketEnvironment": decision_authority[
                "market_environment"
            ],
            "marketEnvironmentInformationalOnly": decision_authority[
                "market_environment_informational_only"
            ],
            "previousGrowthReturn": decision_authority[
                "environment_inputs"
            ]["previous_growth_return"],
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
            "approvedAuthorities": decision_authority[
                "approved_authorities"
            ],
            "blockedAuthorities": decision_authority[
                "blocked_authorities"
            ],
            "unapprovedRealCapitalShare": decision_authority[
                "unapproved_real_capital_share"
            ],
            "stateChangeOrdersAllowed": decision_authority[
                "state_change_orders_allowed"
            ],
            "maximumTargetChange": decision_authority[
                "maximum_target_change_from_authority"
            ],
            "economicTargetUnchanged": decision_authority[
                "economic_target_unchanged"
            ],
            "ordersExecutable": decision_authority[
                "orders_executable"
            ],
        },
        "reentryBrakeShadow": {
            "release": reentry_diagnostics["release"],
            "state": reentry_diagnostics["shadow_state"],
            "triggered": bool(
                int(float(reentry_diagnostics["triggered"]))
            ),
            "active": bool(int(float(reentry_diagnostics["active"]))),
            "engaged": bool(int(float(reentry_diagnostics["engaged"]))),
            "released": bool(
                int(float(reentry_diagnostics["released"]))
            ),
            "previousGrowthWeight": float(
                reentry_diagnostics["previous_growth_weight"]
            ),
            "previousWeightedGrowthReturn": float(
                reentry_diagnostics[
                    "previous_weighted_growth_return"
                ]
            ),
            "qqq63Return": float(
                reentry_diagnostics["qqq_63d_return"]
            ),
            "semis63Return": float(
                reentry_diagnostics["semis_63d_return"]
            ),
            "semis126Return": float(
                reentry_diagnostics["semis_126d_return"]
            ),
            "vixTermRatio": float(
                reentry_diagnostics["vix_term_ratio"]
            ),
            "productionGrowthWeight": float(
                reentry_diagnostics["baseline_growth_weight"]
            ),
            "shadowGrowthWeight": float(
                reentry_diagnostics["shadow_growth_weight"]
            ),
            "stagedTargetDifference": float(
                0.5
                * reentry_targets["staged_difference"].abs().sum()
            ),
            "pointEstimateGatePass": bool(
                int(
                    float(
                        reentry_diagnostics[
                            "point_estimate_gate_pass"
                        ]
                    )
                )
            ),
            "multipleComparisonsGatePass": bool(
                int(
                    float(
                        reentry_diagnostics[
                            "multiple_comparisons_gate_pass"
                        ]
                    )
                )
            ),
            "familywisePValueMax": float(
                reentry_diagnostics["familywise_p_value_max"]
            ),
            "productionEligible": bool(
                int(float(reentry_diagnostics["production_eligible"]))
            ),
            "realCapitalShare": float(
                reentry_diagnostics["real_capital_share"]
            ),
            "ordersExecutable": bool(
                int(float(reentry_diagnostics["orders_executable"]))
            ),
        },
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    DESTINATION.write_text(
        "export const strategyLiveData = "
        + serialized
        + " as const;\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "destination": str(DESTINATION),
                "price_as_of": payload["priceAsOf"],
                "next_session": payload["nextSession"],
                "data_is_fresh": payload["dataIsFresh"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

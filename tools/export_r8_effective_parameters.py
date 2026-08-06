from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from regime_strategy.ensemble_cli import _load_ensemble_config


CONFIG = Path(
    "config/paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_"
    "inverse_momentum_netted_ensemble.yaml"
)
DESTINATION = Path("output/current_operational_panel/strategy_parameters.csv")


def main() -> None:
    ensemble = _load_ensemble_config(CONFIG)
    base = yaml.safe_load(
        Path(str(ensemble["base_config"])).read_text(encoding="utf-8")
    )
    member = ensemble["member_overrides"]
    portfolio = member["portfolio_overrides"]
    turning = portfolio["turning_point_reentry"]
    bridge = turning["zero_entry_bridge"]
    constructive = bridge["constructive_high_volatility_override"]
    momentum = portfolio["relative_momentum_core"]
    risk = portfolio["asset_aware_growth_risk"]
    vix = portfolio["conditional_vix_hedge"]
    rows = [
        ("model", "random_seeds", ",".join(map(str, ensemble["random_seeds"]))),
        ("model", "lookback_days", base["model"]["lookback_days"]),
        ("model", "validation_days", base["model"]["validation_days"]),
        ("model", "candidate_states", ",".join(map(str, base["model"]["candidate_states"]))),
        ("model", "model_refit_days", base["model"]["model_refit_days"]),
        ("execution", "ensemble_rebalance_days", ensemble["ensemble"]["rebalance_every_days"]),
        ("execution", "no_trade_turnover", ensemble["ensemble"]["no_trade_turnover"]),
        ("execution", "cost_bps_per_dollar_traded", ensemble["ensemble"]["cost_bps_per_dollar_traded"]),
        ("risk", "maximum_gross_leverage", member["max_gross_leverage"]),
        ("risk", "fast_volatility_days", risk["fast_days"]),
        ("risk", "slow_volatility_days", risk["slow_days"]),
        ("risk", "growth_target_volatility", risk["target_volatility"]),
        ("risk", "volatility_estimator", risk["volatility_estimator"]),
        ("allocation", "strategic_core_qqq", portfolio["strategic_core_weights"]["QQQ"]),
        ("allocation", "strategic_core_semis", portfolio["strategic_core_weights"]["SEMIS"]),
        ("allocation", "strategic_core_gold", portfolio["strategic_core_weights"]["GOLD"]),
        ("allocation", "relative_momentum_days", momentum["horizons"][0]),
        ("allocation", "high_vol_inverse_volatility_days", momentum["volatility_regime_gate"]["fallback_lookback_days"]),
        ("reentry", "established_floor_bull", turning["state_account_shares"]["bull"]),
        ("reentry", "established_floor_bear", turning["state_account_shares"]["bear"]),
        ("reentry", "established_floor_correction", turning["state_account_shares"]["correction"]),
        ("reentry", "established_floor_rebound", turning["state_account_shares"]["rebound"]),
        ("reentry", "zero_entry_bridge_share", bridge["state_account_shares"]["bull"]),
        ("reentry", "bridge_core_qqq", bridge["core_weights"]["QQQ"]),
        ("reentry", "constructive_window_days", constructive["window_days"]),
        ("reentry", "maximum_downside_variation_share", constructive["maximum_downside_variation_share"]),
        ("reentry", "minimum_slow_excess_return", constructive["minimum_slow_excess_return"]),
        ("hedge", "vixy_account_share", vix["account_share"]),
        ("hedge", "spot_signal", vix["spot_signal"]),
        ("hedge", "three_month_signal", vix["three_month_signal"]),
    ]
    frame = pd.DataFrame(rows, columns=["section", "parameter", "value"])
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DESTINATION, index=False)
    print(frame.to_string(index=False))
    print(f"\nArtifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from regime_strategy.decision_authority import (
    classify_market_environment,
    validate_authority_registry,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config/paper_core_growth_gold20_r11_decision_authority.yaml"
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def latest_environment_inputs(
    market: pd.DataFrame,
    r9_weights: pd.DataFrame,
) -> dict[str, float | str]:
    dates = market.index.intersection(r9_weights.index)
    if len(dates) < 127:
        raise ValueError("Insufficient completed history for classification")
    date = dates[-1]
    position = market.index.get_loc(date)
    if not isinstance(position, int) or position < 126:
        raise ValueError("Classification date lacks causal history")
    prior_date = dates[-2]
    close_returns = market[
        ["SPX", "QQQ", "SEMIS"]
    ].pct_change(fill_method=None)
    prior_weights = r9_weights.loc[
        prior_date,
        ["SPX", "QQQ", "SEMIS"],
    ]
    previous_growth_return = float(
        prior_weights
        @ close_returns.loc[date, ["SPX", "QQQ", "SEMIS"]]
    )
    return {
        "price_as_of": date.date().isoformat(),
        "signal_max_date": date.date().isoformat(),
        "previous_weight_date": prior_date.date().isoformat(),
        "previous_growth_return": previous_growth_return,
        "qqq_63d_return": float(
            market.loc[date, "QQQ"]
            / market["QQQ"].iloc[position - 63]
            - 1.0
        ),
        "semis_63d_return": float(
            market.loc[date, "SEMIS"]
            / market["SEMIS"].iloc[position - 63]
            - 1.0
        ),
        "semis_126d_return": float(
            market.loc[date, "SEMIS"]
            / market["SEMIS"].iloc[position - 126]
            - 1.0
        ),
        "vix_term_ratio": float(
            market.loc[date, "VIX"] / market.loc[date, "VIX3M"]
        ),
    }


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_authority_registry(config["authorities"])
    data = dict(config["data"])
    r9_output = ROOT / str(data["r9_output"])
    r11_output = ROOT / str(data["r11_output"])
    market_path = ROOT / str(data["market_prices"])
    output = ROOT / str(data["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    r11_targets = pd.read_csv(
        r11_output / "next_target_weights.csv",
        index_col="asset",
    )
    base_target = r11_targets["staged_account_center"].astype(float)
    r11_diagnostics = pd.read_csv(
        r11_output / "next_signal_diagnostics.csv",
        index_col=0,
    ).iloc[:, 0]
    r11_metadata = json.loads(
        (r11_output / "run_metadata.json").read_text(encoding="utf-8")
    )
    market = pd.read_csv(
        market_path,
        index_col=0,
        parse_dates=True,
    )
    r9_weights = pd.read_csv(
        r9_output / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    inputs = latest_environment_inputs(market, r9_weights)
    if inputs["price_as_of"] != r11_metadata["price_as_of"]:
        raise RuntimeError(
            "Decision-authority inputs are not synchronized with R11"
        )
    environment = classify_market_environment(
        qqq_63d_return=float(inputs["qqq_63d_return"]),
        semis_63d_return=float(inputs["semis_63d_return"]),
        semis_126d_return=float(inputs["semis_126d_return"]),
        vix_term_ratio=float(inputs["vix_term_ratio"]),
        previous_growth_return=float(inputs["previous_growth_return"]),
        shock_threshold=float(
            config["market_environment"][
                "shock_threshold_account_return"
            ]
        ),
    )

    final_target = base_target.copy()
    proof = pd.DataFrame(index=base_target.index)
    proof.index.name = "asset"
    proof["r11_approved_base_target"] = base_target
    proof["authority_controlled_final_target"] = final_target
    proof["difference"] = final_target - base_target
    proof.to_csv(output / "target_authority_proof.csv")

    rules = config["authorities"]
    approved = [
        name
        for name, rule in rules.items()
        if rule["status"] == "approved"
    ]
    blocked = [
        name
        for name, rule in rules.items()
        if rule["status"] != "approved"
    ]
    maximum_difference = float(proof["difference"].abs().max())
    orders_executable = bool(
        int(float(r11_diagnostics["orders_executable"]))
    )
    payload = {
        "release": config["release"],
        "status": config["status"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "price_as_of": inputs["price_as_of"],
        "next_session": str(r11_diagnostics["next_session"]),
        "market_environment": environment.value,
        "market_environment_informational_only": True,
        "environment_inputs": inputs,
        "approved_authorities": approved,
        "blocked_authorities": blocked,
        "unapproved_real_capital_share": 0.0,
        "state_change_orders_allowed": False,
        "maximum_target_change_from_authority": maximum_difference,
        "economic_target_unchanged": bool(maximum_difference <= 1e-12),
        "orders_executable": orders_executable,
        "order_blockers": str(r11_diagnostics["order_blockers"]),
        "source_hashes": {
            str(CONFIG.relative_to(ROOT)): file_sha256(CONFIG),
            str(market_path.relative_to(ROOT)): file_sha256(market_path),
            str(
                (r11_output / "next_target_weights.csv").relative_to(
                    ROOT
                )
            ): file_sha256(r11_output / "next_target_weights.csv"),
            str(
                (r11_output / "run_metadata.json").relative_to(ROOT)
            ): file_sha256(r11_output / "run_metadata.json"),
        },
    }
    (output / "decision_authority.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nArtifacts: {output.resolve()}")


if __name__ == "__main__":
    main()

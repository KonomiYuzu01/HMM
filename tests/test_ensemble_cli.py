from pathlib import Path

from regime_strategy.ensemble_cli import (
    _deep_merge,
    _load_ensemble_config,
    _member_specs,
)


def test_member_specs_preserve_default_names_and_expand_lookbacks() -> None:
    assert _member_specs([7, 42], None) == [
        ("seed_7", 7, None),
        ("seed_42", 42, None),
    ]
    assert _member_specs([7, 42], [756, 1008]) == [
        ("seed_7_lookback_756", 7, 756),
        ("seed_42_lookback_756", 42, 756),
        ("seed_7_lookback_1008", 7, 1008),
        ("seed_42_lookback_1008", 42, 1008),
    ]


def test_member_specs_reject_invalid_lookbacks() -> None:
    for lookbacks in ([0], [756, 756]):
        try:
            _member_specs([7], lookbacks)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid lookbacks must be rejected")


def test_deep_merge_preserves_nested_parent_values() -> None:
    merged = _deep_merge(
        {"member": {"portfolio": {"quality": True, "memory": False}}},
        {"member": {"portfolio": {"memory": True}}},
    )

    assert merged == {
        "member": {"portfolio": {"quality": True, "memory": True}}
    }


def test_load_ensemble_config_supports_small_parent_overrides(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        "base_config: base.yaml\nmember_overrides:\n  max_gross_leverage: 1.1\n",
        encoding="utf-8",
    )
    child.write_text(
        f"parent_config: {parent}\n"
        "member_overrides:\n"
        "  backtest_overrides:\n"
        '    oos_start: "2012-01-03"\n',
        encoding="utf-8",
    )

    config = _load_ensemble_config(child)

    assert config["base_config"] == "base.yaml"
    assert config["member_overrides"]["max_gross_leverage"] == 1.1
    assert (
        config["member_overrides"]["backtest_overrides"]["oos_start"]
        == "2012-01-03"
    )

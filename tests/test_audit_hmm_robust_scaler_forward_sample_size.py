from tools.audit_hmm_robust_scaler_forward_sample_size import (
    required_blocks_for_mean,
)


def test_required_blocks_decrease_with_larger_effect() -> None:
    small = required_blocks_for_mean(0.01, 0.10)
    large = required_blocks_for_mean(0.02, 0.10)

    assert small is not None
    assert large is not None
    assert large < small


def test_required_blocks_handles_zero_effect_and_variance() -> None:
    assert required_blocks_for_mean(0.0, 0.1) is None
    assert required_blocks_for_mean(0.01, 0.0) == 1

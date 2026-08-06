import numpy as np
import pandas as pd
from types import SimpleNamespace
from hmmlearn.hmm import GaussianHMM

from regime_strategy.model import (
    StudentTDiagHMM,
    RegimeTemplate,
    WassersteinTemplateTracker,
    _fit_hmm,
    _parameter_count,
    assign_templates,
    equal_weight_predictive_moments,
    empirical_bayes_state_return_means,
    emission_marginal_variances,
    clip_features,
    feature_clip_bounds,
    filtered_state_posteriors,
    gaussian_w2_diag,
    hmm_fit_diagnostics,
    inverse_emission_marginals,
    make_feature_scaler,
    normalized_probability_entropy,
    probability_margin,
    probability_weighted_predictive_moments,
    validation_predictive_score,
    weighted_huber_location,
)


def _one_state_fit(mean: float, variance: float) -> SimpleNamespace:
    return SimpleNamespace(
        order=1,
        raw_means=np.array([[mean]]),
        raw_variances=np.array([[variance]]),
        state_return_means=np.array([[0.0]]),
        state_return_covariances=np.array([[[1.0]]]),
    )


def test_template_mapping_can_skip_parameter_update() -> None:
    tracker = WassersteinTemplateTracker(1, 0.5)
    tracker.templates = [
        RegimeTemplate(
            np.array([0.0]), np.array([4.0]), np.array([0.0]), np.array([[1.0]])
        )
    ]

    probabilities, mapping = tracker.map_and_update(
        _one_state_fit(2.0, 9.0), np.array([1.0]), update=False
    )

    assert mapping == [0]
    assert np.allclose(probabilities, np.array([1.0]))
    assert np.allclose(tracker.templates[0].feature_mean, np.array([0.0]))
    assert np.allclose(tracker.templates[0].feature_variance, np.array([4.0]))


def test_wasserstein_template_update_interpolates_standard_deviation() -> None:
    tracker = WassersteinTemplateTracker(
        1, 0.5, update_geometry="wasserstein_diag"
    )
    tracker.templates = [
        RegimeTemplate(
            np.array([0.0]), np.array([4.0]), np.array([0.0]), np.array([[1.0]])
        )
    ]

    tracker.map_and_update(_one_state_fit(2.0, 9.0), np.array([1.0]))

    assert np.allclose(tracker.templates[0].feature_mean, np.array([1.0]))
    assert np.allclose(tracker.templates[0].feature_variance, np.array([6.25]))


def test_wasserstein_diagonal_gaussian_distance() -> None:
    mean = np.array([1.0, -2.0])
    variance = np.array([4.0, 9.0])
    assert gaussian_w2_diag(mean, variance, mean, variance) == 0.0
    shifted = gaussian_w2_diag(mean, variance, mean + 1.0, variance)
    assert np.isclose(shifted, 2.0)
    assert np.isclose(
        gaussian_w2_diag(mean, variance, mean + 1.0, variance),
        gaussian_w2_diag(mean + 1.0, variance, mean, variance),
    )


def test_equal_weight_predictive_moments_includes_model_disagreement() -> None:
    mean, covariance = equal_weight_predictive_moments(
        [np.array([0.0, 0.0]), np.array([2.0, 0.0])],
        [np.eye(2), 3.0 * np.eye(2)],
    )

    assert np.allclose(mean, np.array([1.0, 0.0]))
    assert np.allclose(covariance, np.array([[3.0, 0.0], [0.0, 2.0]]))


def test_equal_weight_predictive_moments_rejects_misaligned_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        equal_weight_predictive_moments([np.zeros(2)], [])


def test_hungarian_template_assignment_prevents_nearest_neighbor_collisions() -> None:
    distances = np.array([[0.1, 0.2, 5.0], [0.1, 0.3, 5.0]])

    assert assign_templates(distances, "nearest") == [0, 0]
    hungarian = assign_templates(distances, "hungarian")
    assert len(set(hungarian)) == 2
    assert hungarian == [1, 0]


def test_probability_weighted_moments_are_label_invariant() -> None:
    means = np.array([[0.0, 1.0], [2.0, -1.0]])
    covariances = np.array([np.eye(2), 2.0 * np.eye(2)])
    probabilities = np.array([0.25, 0.75])
    mean, covariance = probability_weighted_predictive_moments(
        means, covariances, probabilities
    )
    permuted_mean, permuted_covariance = probability_weighted_predictive_moments(
        means[::-1], covariances[::-1], probabilities[::-1]
    )

    assert np.allclose(mean, permuted_mean)
    assert np.allclose(covariance, permuted_covariance)


def test_probability_diagnostics_have_expected_boundaries() -> None:
    assert normalized_probability_entropy(np.array([1.0, 0.0])) == 0.0
    assert normalized_probability_entropy(np.array([0.5, 0.5])) == 1.0
    assert probability_margin(np.array([1.0, 0.0])) == 1.0
    assert probability_margin(np.array([0.5, 0.5])) == 0.0


def test_forward_filtered_posteriors_are_normalized_and_match_final_smoother() -> None:
    observations = np.array([[-2.0], [-1.5], [1.0], [2.0]])
    model = GaussianHMM(n_components=2, covariance_type="diag")
    model.startprob_ = np.array([0.8, 0.2])
    model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
    model.means_ = np.array([[-1.0], [1.0]])
    model.covars_ = np.array([[0.5], [0.5]])

    filtered = filtered_state_posteriors(model, observations)

    assert np.allclose(filtered.sum(axis=1), 1.0)
    assert np.allclose(filtered[-1], model.predict_proba(observations)[-1])
    assert not np.allclose(filtered[0], model.predict_proba(observations)[0])


def test_validation_score_can_continue_from_training_endpoint() -> None:
    class LengthScoreModel:
        def score(self, values: np.ndarray) -> float:
            return float(len(values) ** 2)

    train = np.zeros((2, 1))
    validation = np.zeros((3, 1))
    model = LengthScoreModel()

    assert validation_predictive_score(model, train, validation, False) == 3.0
    assert validation_predictive_score(model, train, validation, True) == 7.0


def test_robust_feature_scaler_centers_on_the_median() -> None:
    values = np.array([[0.0], [1.0], [2.0], [100.0]])
    transformed = make_feature_scaler("robust").fit_transform(values)
    assert np.isclose(np.median(transformed), 0.0)


def test_quantile_normal_feature_scaler_is_finite_and_deterministic() -> None:
    values = np.linspace(-3.0, 4.0, 1_001).reshape(-1, 1)

    first = make_feature_scaler("quantile_normal").fit_transform(values)
    second = make_feature_scaler("quantile_normal").fit_transform(values)

    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second)
    assert np.isclose(np.median(first[:, 0]), 0.0, atol=1e-12)


def test_quantile_normal_scaler_caps_quantiles_at_training_sample_size() -> None:
    values = np.linspace(-2.0, 2.0, 101).reshape(-1, 1)
    scaler = make_feature_scaler("quantile_normal").fit(values)

    assert scaler.n_quantiles == len(values)
    assert scaler.n_quantiles_ == len(values)


def test_inverse_emission_marginals_preserves_linear_scaler_formula() -> None:
    scaler = make_feature_scaler("standard").fit(
        np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    )
    means = np.array([[0.0, 1.0]])
    variances = np.array([[0.5, 2.0]])

    raw_means, raw_variances = inverse_emission_marginals(
        scaler, means, variances
    )

    np.testing.assert_allclose(raw_means, scaler.inverse_transform(means))
    np.testing.assert_allclose(
        raw_variances, variances * np.square(scaler.scale_)[None, :]
    )


def test_inverse_emission_marginals_supports_quantile_scaler() -> None:
    values = np.column_stack(
        [np.linspace(1.0, 5.0, 1_001), np.linspace(10.0, 40.0, 1_001)]
    )
    scaler = make_feature_scaler("quantile_normal").fit(values)

    raw_means, raw_variances = inverse_emission_marginals(
        scaler,
        np.zeros((2, 2)),
        np.array([[0.5, 1.0], [1.5, 0.25]]),
    )

    assert raw_means.shape == (2, 2)
    assert raw_variances.shape == (2, 2)
    assert np.isfinite(raw_means).all()
    assert np.isfinite(raw_variances).all()
    assert (raw_variances > 0.0).all()


def test_pca90_scaler_reduces_redundant_features_and_restores_marginals() -> None:
    generator = np.random.default_rng(7)
    factors = generator.normal(size=(1_001, 2))
    values = np.column_stack(
        [
            factors[:, 0],
            factors[:, 0] + 0.01 * generator.normal(size=1_001),
            factors[:, 1],
            factors[:, 1] + 0.01 * generator.normal(size=1_001),
        ]
    )
    scaler = make_feature_scaler("pca90").fit(values)
    transformed = scaler.transform(values)

    assert transformed.shape[1] < values.shape[1]
    raw_means, raw_variances = inverse_emission_marginals(
        scaler,
        np.zeros((2, transformed.shape[1])),
        np.ones((2, transformed.shape[1])),
    )
    assert raw_means.shape == (2, values.shape[1])
    assert raw_variances.shape == (2, values.shape[1])
    assert np.isfinite(raw_variances).all()
    assert (raw_variances >= 0.0).all()
    standard = scaler.named_steps["standard"]
    components = scaler.named_steps["pca"].components_
    expected_one_state = (
        np.ones(transformed.shape[1]) @ np.square(components)
    ) * np.square(standard.scale_)
    np.testing.assert_allclose(raw_variances[0], expected_one_state)


def test_pca_whiten_scaler_preserves_dimensions_and_whitens_covariance() -> None:
    generator = np.random.default_rng(17)
    base = generator.normal(size=(1_001, 3))
    values = base @ np.array(
        [[1.0, 0.8, 0.2], [0.0, 1.0, 0.6], [0.4, 0.0, 1.0]]
    )
    scaler = make_feature_scaler("pca_whiten").fit(values)
    transformed = scaler.transform(values)

    assert transformed.shape == values.shape
    np.testing.assert_allclose(
        np.cov(transformed, rowvar=False), np.eye(values.shape[1]), atol=1e-10
    )
    raw_means, raw_variances = inverse_emission_marginals(
        scaler,
        np.zeros((1, transformed.shape[1])),
        np.ones((1, transformed.shape[1])),
    )
    assert raw_means.shape == (1, values.shape[1])
    assert raw_variances.shape == (1, values.shape[1])
    assert np.isfinite(raw_variances).all()
    assert (raw_variances > 0.0).all()


def test_empirical_bayes_state_means_shrink_toward_unconditional_mean() -> None:
    returns = np.array([[-1.0], [0.0], [0.0], [1.0]])
    posterior = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    means = np.array([[-0.5], [0.5]])
    covariances = np.array([[[1.0]], [[1.0]]])

    shrunk = empirical_bayes_state_return_means(
        returns, posterior, means, covariances
    )

    assert np.all(np.abs(shrunk) < np.abs(means))


def test_weighted_huber_location_reduces_outlier_influence() -> None:
    values = np.concatenate([np.zeros(60), np.array([20.0])])
    weights = np.ones(len(values))
    robust = weighted_huber_location(values, weights)

    assert abs(robust) < abs(float(values.mean())) / 2.0


def test_weighted_huber_location_preserves_symmetric_center() -> None:
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert np.isclose(weighted_huber_location(values, np.ones(5)), 0.0)


def test_student_t_emission_is_less_surprised_by_an_outlier() -> None:
    gaussian = StudentTDiagHMM(
        n_components=1,
        covariance_type="diag",
        degrees_of_freedom=1_000_000.0,
    )
    heavy_tailed = StudentTDiagHMM(
        n_components=1,
        covariance_type="diag",
        degrees_of_freedom=5.0,
    )
    for model in (gaussian, heavy_tailed):
        model.n_features = 1
        model.means_ = np.array([[0.0]])
        model._covars_ = np.array([[1.0]])
    outlier = np.array([[10.0]])
    assert (
        heavy_tailed._compute_log_likelihood(outlier)[0, 0]
        > gaussian._compute_log_likelihood(outlier)[0, 0]
    )


def test_student_t_em_reduces_outlier_influence_on_the_mean() -> None:
    observations = np.concatenate(
        [np.zeros((60, 1)), np.array([[20.0]])],
        axis=0,
    )
    model = StudentTDiagHMM(
        n_components=1,
        covariance_type="diag",
        degrees_of_freedom=5.0,
        n_iter=100,
        random_state=7,
    )
    model.fit(observations)
    assert abs(float(model.means_[0, 0])) < float(observations.mean()) / 2.0


def test_sticky_transition_prior_is_applied_without_changing_emissions() -> None:
    observations = np.random.default_rng(7).normal(size=(120, 2))
    model = _fit_hmm(
        observations,
        states=2,
        n_iter=20,
        restarts=1,
        random_seed=7,
        covariance_floor=1e-6,
        sticky_transition_prior=10.0,
    )
    np.testing.assert_allclose(
        model.transmat_prior,
        np.array([[11.0, 1.0], [1.0, 11.0]]),
    )


def test_hmm_fit_records_restart_and_terminal_diagnostics() -> None:
    observations = np.random.default_rng(11).normal(size=(160, 2))
    model = _fit_hmm(
        observations,
        states=2,
        n_iter=20,
        restarts=3,
        random_seed=11,
        covariance_floor=1e-6,
    )

    diagnostics = hmm_fit_diagnostics(model)

    assert len(model.restart_diagnostics_) == 3
    assert model.best_restart_ in {0, 1, 2}
    assert diagnostics["hmm_fit_iterations"] > 0
    assert np.isfinite(diagnostics["hmm_fit_final_delta"])
    assert diagnostics["hmm_restart_score_range"] >= 0.0


def test_tied_covariance_parameter_count_and_marginals() -> None:
    assert _parameter_count(3, 4, "diag") == 2 + 6 + 12 + 12
    assert _parameter_count(3, 4, "tied") == 2 + 6 + 12 + 10
    observations = np.random.default_rng(17).normal(size=(180, 3))
    model = _fit_hmm(
        observations,
        states=2,
        n_iter=20,
        restarts=1,
        random_seed=17,
        covariance_floor=1e-6,
        covariance_type="tied",
    )
    assert emission_marginal_variances(model).shape == (2, 3)


def test_feature_clipping_uses_fixed_training_bounds() -> None:
    training = np.arange(200, dtype=float)
    frame = pd.DataFrame({"feature": training})
    lower, upper = feature_clip_bounds(frame, 0.005)
    current = pd.DataFrame({"feature": [-100.0, 500.0]})
    clipped = clip_features(current, lower, upper)
    assert clipped.iloc[0, 0] == lower["feature"]
    assert clipped.iloc[1, 0] == upper["feature"]

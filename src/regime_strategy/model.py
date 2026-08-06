from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings

from hmmlearn.base import BaseHMM
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln, logsumexp


def gaussian_w2_diag(
    mean_a: np.ndarray,
    variance_a: np.ndarray,
    mean_b: np.ndarray,
    variance_b: np.ndarray,
) -> float:
    """Squared 2-Wasserstein distance for diagonal Gaussian distributions."""
    mean_term = np.square(mean_a - mean_b).sum()
    covariance_term = np.square(
        np.sqrt(np.maximum(variance_a, 0.0))
        - np.sqrt(np.maximum(variance_b, 0.0))
    ).sum()
    return float(mean_term + covariance_term)


def normalized_probability_entropy(probabilities: np.ndarray) -> float:
    """Return entropy on a zero-to-one scale for a finite probability vector."""
    values = np.maximum(np.asarray(probabilities, dtype=float), 0.0)
    if values.ndim != 1 or not len(values) or values.sum() <= 1e-12:
        raise ValueError("Probabilities must be a positive one-dimensional vector")
    values /= values.sum()
    if len(values) == 1:
        return 0.0
    positive = values[values > 0.0]
    return float(-(positive * np.log(positive)).sum() / np.log(len(values)))


def probability_margin(probabilities: np.ndarray) -> float:
    """Return the difference between the two largest probabilities."""
    values = np.maximum(np.asarray(probabilities, dtype=float), 0.0)
    if values.ndim != 1 or not len(values) or values.sum() <= 1e-12:
        raise ValueError("Probabilities must be a positive one-dimensional vector")
    values /= values.sum()
    if len(values) == 1:
        return 1.0
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2])


def filtered_state_posteriors(
    model: BaseHMM,
    observations: np.ndarray,
) -> np.ndarray:
    """Return normalized one-sided forward state probabilities for every row."""
    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("HMM observations must be a non-empty two-dimensional array")
    log_emissions = np.asarray(model._compute_log_likelihood(values), dtype=float)
    states = log_emissions.shape[1]
    log_start = np.log(np.maximum(np.asarray(model.startprob_, dtype=float), 1e-300))
    log_transition = np.log(
        np.maximum(np.asarray(model.transmat_, dtype=float), 1e-300)
    )
    log_alpha = np.empty((len(values), states), dtype=float)
    log_alpha[0] = log_start + log_emissions[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for position in range(1, len(values)):
        log_alpha[position] = log_emissions[position] + logsumexp(
            log_alpha[position - 1][:, None] + log_transition,
            axis=0,
        )
        log_alpha[position] -= logsumexp(log_alpha[position])
    return np.exp(log_alpha)


def validation_predictive_score(
    model: BaseHMM,
    train: np.ndarray,
    validation: np.ndarray,
    conditioned_on_train: bool,
) -> float:
    """Return average validation log likelihood with optional state continuation."""
    if not len(validation):
        raise ValueError("Validation observations cannot be empty")
    if conditioned_on_train:
        joint = np.concatenate([train, validation], axis=0)
        score = model.score(joint) - model.score(train)
    else:
        score = model.score(validation)
    return float(score / len(validation))


class CausalQuantileTransformer(QuantileTransformer):
    """Empirical Gaussianization with a causal sample-size-aware quantile cap."""

    def __init__(self, maximum_quantiles: int = 1_000):
        if maximum_quantiles < 2:
            raise ValueError("Maximum quantiles must be at least two")
        self.maximum_quantiles = int(maximum_quantiles)
        super().__init__(
            n_quantiles=self.maximum_quantiles,
            output_distribution="normal",
            subsample=None,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> CausalQuantileTransformer:
        self.n_quantiles = min(self.maximum_quantiles, len(X))
        return super().fit(X, y)


FeatureScaler = StandardScaler | RobustScaler | QuantileTransformer | Pipeline


def make_feature_scaler(mode: str) -> FeatureScaler:
    if mode == "standard":
        return StandardScaler()
    if mode == "robust":
        return RobustScaler()
    if mode == "quantile_normal":
        return CausalQuantileTransformer(maximum_quantiles=1_000)
    if mode == "pca90":
        return Pipeline(
            [
                ("standard", StandardScaler()),
                ("pca", PCA(n_components=0.9, svd_solver="full")),
            ]
        )
    if mode == "pca_whiten":
        return Pipeline(
            [
                ("standard", StandardScaler()),
                ("pca", PCA(whiten=True, svd_solver="full")),
            ]
        )
    raise ValueError(f"Unsupported feature scaler: {mode}")


def inverse_emission_marginals(
    scaler: FeatureScaler,
    transformed_means: np.ndarray,
    transformed_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map latent diagonal emission moments back to original feature units."""
    means = np.asarray(transformed_means, dtype=float)
    variances = np.asarray(transformed_variances, dtype=float)
    if means.shape != variances.shape or means.ndim != 2:
        raise ValueError("Emission means and variances must be aligned matrices")
    raw_means = np.asarray(scaler.inverse_transform(means), dtype=float)
    scale = getattr(scaler, "scale_", None)
    if scale is not None:
        return raw_means, variances * np.square(np.asarray(scale, dtype=float))[None, :]

    raw_variances = np.zeros_like(raw_means)
    standard_deviations = np.sqrt(np.maximum(variances, 0.0))
    for dimension in range(means.shape[1]):
        plus = means.copy()
        minus = means.copy()
        plus[:, dimension] += standard_deviations[:, dimension]
        minus[:, dimension] -= standard_deviations[:, dimension]
        raw_plus = np.asarray(scaler.inverse_transform(plus), dtype=float)
        raw_minus = np.asarray(scaler.inverse_transform(minus), dtype=float)
        raw_variances += np.square((raw_plus - raw_minus) / 2.0)
    if not np.isfinite(raw_means).all() or not np.isfinite(raw_variances).all():
        raise ValueError("Inverse-transformed emission moments must be finite")
    return raw_means, raw_variances


def feature_clip_bounds(
    features: pd.DataFrame,
    quantile: float,
) -> tuple[pd.Series, pd.Series]:
    """Return training-only per-feature clipping bounds."""
    if not 0.0 <= quantile < 0.5:
        raise ValueError("Feature clip quantile must lie inside [0, 0.5)")
    if quantile == 0.0:
        return (
            pd.Series(-np.inf, index=features.columns, dtype=float),
            pd.Series(np.inf, index=features.columns, dtype=float),
        )
    return (
        features.quantile(quantile).astype(float),
        features.quantile(1.0 - quantile).astype(float),
    )


def clip_features(
    features: pd.DataFrame,
    lower: pd.Series,
    upper: pd.Series,
) -> pd.DataFrame:
    """Apply aligned bounds learned from a causal training window."""
    return features.clip(
        lower=lower.reindex(features.columns),
        upper=upper.reindex(features.columns),
        axis="columns",
    )


class StudentTDiagHMM(GaussianHMM):
    """Diagonal multivariate Student-t HMM with a fixed degrees of freedom.

    The EM update includes the Student-t latent scale for every observation and
    state. This is intentionally limited to diagonal scale matrices so the
    parameter count remains aligned with the production Gaussian model.
    """

    def __init__(self, *args: object, degrees_of_freedom: float = 5.0, **kwargs: object):
        if float(degrees_of_freedom) <= 2.0:
            raise ValueError("Student-t degrees of freedom must be greater than two")
        covariance_type = str(kwargs.get("covariance_type", "diag"))
        if covariance_type != "diag":
            raise ValueError("StudentTDiagHMM supports only diagonal covariance")
        self.degrees_of_freedom = float(degrees_of_freedom)
        super().__init__(*args, **kwargs)

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        observations = np.asarray(X, dtype=float)
        scale = np.maximum(np.asarray(self._covars_, dtype=float), self.min_covar)
        difference = observations[:, None, :] - self.means_[None, :, :]
        mahalanobis = np.sum(
            np.square(difference) / scale[None, :, :], axis=2
        )
        dimensions = observations.shape[1]
        degrees = self.degrees_of_freedom
        normalizer = (
            gammaln((degrees + dimensions) / 2.0)
            - gammaln(degrees / 2.0)
            - 0.5
            * (
                dimensions * np.log(degrees * np.pi)
                + np.log(scale).sum(axis=1)
            )
        )
        return normalizer[None, :] - 0.5 * (degrees + dimensions) * np.log1p(
            mahalanobis / degrees
        )

    def _initialize_sufficient_statistics(self) -> dict[str, np.ndarray | float | int]:
        stats = BaseHMM._initialize_sufficient_statistics(self)
        stats["post"] = np.zeros(self.n_components)
        stats["weighted_post"] = np.zeros(self.n_components)
        stats["weighted_obs"] = np.zeros((self.n_components, self.n_features))
        stats["weighted_obs2"] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(
        self,
        stats: dict[str, np.ndarray | float | int],
        X: np.ndarray,
        lattice: np.ndarray,
        posteriors: np.ndarray,
        fwdlattice: np.ndarray,
        bwdlattice: np.ndarray,
    ) -> None:
        BaseHMM._accumulate_sufficient_statistics(
            self,
            stats=stats,
            X=X,
            lattice=lattice,
            posteriors=posteriors,
            fwdlattice=fwdlattice,
            bwdlattice=bwdlattice,
        )
        scale = np.maximum(np.asarray(self._covars_, dtype=float), self.min_covar)
        difference = X[:, None, :] - self.means_[None, :, :]
        mahalanobis = np.sum(
            np.square(difference) / scale[None, :, :], axis=2
        )
        latent_scale = (
            self.degrees_of_freedom + self.n_features
        ) / (self.degrees_of_freedom + mahalanobis)
        weighted = posteriors * latent_scale
        stats["post"] += posteriors.sum(axis=0)
        stats["weighted_post"] += weighted.sum(axis=0)
        stats["weighted_obs"] += weighted.T @ X
        stats["weighted_obs2"] += weighted.T @ np.square(X)

    def _do_mstep(self, stats: dict[str, np.ndarray | float | int]) -> None:
        BaseHMM._do_mstep(self, stats)
        weighted_post = np.maximum(
            np.asarray(stats["weighted_post"], dtype=float), 1e-12
        )[:, None]
        weighted_obs = np.asarray(stats["weighted_obs"], dtype=float)
        if "m" in self.params:
            self.means_ = weighted_obs / weighted_post
        if "c" in self.params:
            weighted_obs2 = np.asarray(stats["weighted_obs2"], dtype=float)
            numerator = (
                weighted_obs2
                - 2.0 * self.means_ * weighted_obs
                + np.square(self.means_) * weighted_post
            )
            posterior_mass = np.maximum(
                np.asarray(stats["post"], dtype=float), 1e-12
            )[:, None]
            self._covars_ = np.maximum(
                numerator / posterior_mass,
                self.min_covar,
            )


def assign_templates(distance_matrix: np.ndarray, mode: str) -> list[int]:
    """Assign fitted states to persistent templates under the configured constraint."""
    distances = np.asarray(distance_matrix, dtype=float)
    if distances.ndim != 2 or not np.isfinite(distances).all():
        raise ValueError("Template distances must be a finite two-dimensional matrix")
    if mode == "nearest":
        return [int(value) for value in np.argmin(distances, axis=1)]
    if mode != "hungarian":
        raise ValueError(f"Unsupported template assignment mode: {mode}")
    if distances.shape[0] > distances.shape[1]:
        raise ValueError("Hungarian template assignment requires at least one template per state")
    rows, columns = linear_sum_assignment(distances)
    mapping = [-1] * distances.shape[0]
    for row, column in zip(rows, columns, strict=True):
        mapping[int(row)] = int(column)
    if any(value < 0 for value in mapping):
        raise RuntimeError("Hungarian template assignment left a state unmapped")
    return mapping


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def equal_weight_predictive_moments(
    means: list[np.ndarray],
    covariances: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine model forecasts, including covariance from model disagreement."""
    if not means or len(means) != len(covariances):
        raise ValueError("Means and covariances must be non-empty and aligned")
    mean_stack = np.asarray(means, dtype=float)
    covariance_stack = np.asarray(covariances, dtype=float)
    if mean_stack.ndim != 2 or covariance_stack.ndim != 3:
        raise ValueError("Predictive moments have invalid dimensions")
    if covariance_stack.shape != (
        len(mean_stack),
        mean_stack.shape[1],
        mean_stack.shape[1],
    ):
        raise ValueError("Predictive means and covariances have incompatible shapes")

    combined_mean = mean_stack.mean(axis=0)
    disagreement = mean_stack - combined_mean
    combined_covariance = np.mean(
        covariance_stack
        + disagreement[:, :, None] * disagreement[:, None, :],
        axis=0,
    )
    return combined_mean, _nearest_psd(combined_covariance)


def probability_weighted_predictive_moments(
    means: np.ndarray,
    covariances: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the law of total covariance to a finite predictive mixture."""
    mean_stack = np.asarray(means, dtype=float)
    covariance_stack = np.asarray(covariances, dtype=float)
    weights = np.maximum(np.asarray(probabilities, dtype=float), 0.0)
    if mean_stack.ndim != 2 or covariance_stack.shape != (
        len(mean_stack),
        mean_stack.shape[1],
        mean_stack.shape[1],
    ):
        raise ValueError("Predictive means and covariances have incompatible shapes")
    if weights.shape != (len(mean_stack),) or weights.sum() <= 1e-12:
        raise ValueError("Predictive probabilities must be positive and aligned")
    weights /= weights.sum()
    combined_mean = weights @ mean_stack
    disagreement = mean_stack - combined_mean
    combined_covariance = np.sum(
        weights[:, None, None]
        * (
            covariance_stack
            + disagreement[:, :, None] * disagreement[:, None, :]
        ),
        axis=0,
    )
    return combined_mean, _nearest_psd(combined_covariance)


def _weighted_return_moments(
    returns: np.ndarray,
    posterior: np.ndarray,
    covariance_floor: float,
    location_mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.maximum(np.asarray(posterior, dtype=float), 0.0)
    total = weights.sum()
    if total <= 1e-8:
        weights = np.ones(len(returns), dtype=float)
        total = float(len(returns))
    normalized = weights / total
    if location_mode == "huber":
        mean = np.asarray(
            [
                weighted_huber_location(returns[:, column], normalized)
                for column in range(returns.shape[1])
            ]
        )
    elif location_mode == "mean":
        mean = normalized @ returns
    else:
        raise ValueError(f"Unsupported return location estimator: {location_mode}")
    centered = returns - mean

    # Ledoit-Wolf on posterior-weighted centered samples stabilizes sparse states.
    scaled = centered * np.sqrt(normalized[:, None] * len(returns))
    covariance = LedoitWolf(assume_centered=True).fit(scaled).covariance_
    covariance = _nearest_psd(covariance, covariance_floor)
    return mean, covariance


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Weighted quantile must lie inside [0, 1]")
    selected = np.asarray(values, dtype=float)
    mass = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if selected.ndim != 1 or mass.shape != selected.shape or mass.sum() <= 1e-12:
        raise ValueError("Weighted quantile inputs must be aligned and positive")
    order = np.argsort(selected)
    selected = selected[order]
    mass = mass[order]
    threshold = quantile * mass.sum()
    index = min(int(np.searchsorted(np.cumsum(mass), threshold, side="left")), len(selected) - 1)
    return float(selected[index])


def weighted_huber_location(
    values: np.ndarray,
    weights: np.ndarray,
    tuning: float = 1.345,
    maximum_iterations: int = 50,
) -> float:
    """Return a posterior-weighted Huber location with robust scale."""
    if tuning <= 0.0 or maximum_iterations < 1:
        raise ValueError("Huber tuning and iteration count must be positive")
    selected = np.asarray(values, dtype=float)
    mass = np.maximum(np.asarray(weights, dtype=float), 0.0)
    mass /= mass.sum()
    location = weighted_quantile(selected, mass, 0.5)
    mad = weighted_quantile(np.abs(selected - location), mass, 0.5)
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.sqrt(mass @ np.square(selected - location)))
    if scale <= 1e-12:
        return location
    cutoff = tuning * scale
    for _ in range(maximum_iterations):
        residual = selected - location
        robust_weights = np.minimum(
            1.0,
            cutoff / np.maximum(np.abs(residual), 1e-18),
        )
        combined = mass * robust_weights
        updated = float(combined @ selected / combined.sum())
        if abs(updated - location) <= 1e-12 * max(1.0, abs(location)):
            return updated
        location = updated
    return location


def empirical_bayes_state_return_means(
    returns: np.ndarray,
    posterior: np.ndarray,
    state_means: np.ndarray,
    state_covariances: np.ndarray,
) -> np.ndarray:
    """Shrink noisy state means using posterior effective sample sizes."""
    values = np.asarray(returns, dtype=float)
    responsibilities = np.maximum(np.asarray(posterior, dtype=float), 0.0)
    means = np.asarray(state_means, dtype=float)
    covariances = np.asarray(state_covariances, dtype=float)
    if responsibilities.shape != (len(values), len(means)):
        raise ValueError("Posterior responsibilities are not aligned to state means")
    if covariances.shape != (len(means), values.shape[1], values.shape[1]):
        raise ValueError("State covariances are not aligned to return means")
    mass = responsibilities.sum(axis=0)
    mass_weights = mass / max(float(mass.sum()), 1e-12)
    effective_n = np.square(mass) / np.maximum(
        np.square(responsibilities).sum(axis=0), 1e-12
    )
    unconditional = values.mean(axis=0)
    sampling_variance = np.diagonal(covariances, axis1=1, axis2=2) / np.maximum(
        effective_n[:, None], 1.0
    )
    observed_between = np.sum(
        mass_weights[:, None] * np.square(means - unconditional),
        axis=0,
    )
    average_sampling = np.sum(
        mass_weights[:, None] * sampling_variance,
        axis=0,
    )
    between_variance = np.maximum(observed_between - average_sampling, 0.0)
    shrinkage = between_variance / np.maximum(
        between_variance + sampling_variance,
        1e-18,
    )
    return unconditional + shrinkage * (means - unconditional)


def _parameter_count(
    states: int,
    dimensions: int,
    covariance_type: str = "diag",
) -> int:
    covariance_parameters = {
        "diag": states * dimensions,
        "tied": dimensions * (dimensions + 1) // 2,
    }.get(covariance_type)
    if covariance_parameters is None:
        raise ValueError(f"Unsupported HMM covariance type: {covariance_type}")
    return (
        (states - 1)
        + states * (states - 1)
        + states * dimensions
        + covariance_parameters
    )


def emission_marginal_variances(model: BaseHMM) -> np.ndarray:
    covariances = np.asarray(model.covars_, dtype=float)
    if covariances.ndim == 3:
        return np.diagonal(covariances, axis1=1, axis2=2)
    if covariances.ndim == 2:
        return covariances
    raise ValueError("HMM emission covariances have an unsupported shape")


def _fit_hmm(
    observations: np.ndarray,
    states: int,
    n_iter: int,
    restarts: int,
    random_seed: int,
    covariance_floor: float,
    emission_distribution: str = "gaussian",
    student_t_degrees_of_freedom: float = 5.0,
    sticky_transition_prior: float = 0.0,
    covariance_type: str = "diag",
) -> GaussianHMM:
    if sticky_transition_prior < 0.0:
        raise ValueError("Sticky transition prior cannot be negative")
    if covariance_type not in {"diag", "tied"}:
        raise ValueError(f"Unsupported HMM covariance type: {covariance_type}")
    if emission_distribution == "student_t" and covariance_type != "diag":
        raise ValueError("Student-t HMM currently supports diagonal covariance only")
    best: GaussianHMM | None = None
    best_score = -np.inf
    restart_diagnostics: list[dict[str, float | int]] = []
    best_restart = -1
    for restart in range(restarts):
        model_class = {
            "gaussian": GaussianHMM,
            "student_t": StudentTDiagHMM,
        }.get(emission_distribution)
        if model_class is None:
            raise ValueError(
                f"Unsupported HMM emission distribution: {emission_distribution}"
            )
        model_kwargs: dict[str, object] = {}
        if model_class is StudentTDiagHMM:
            model_kwargs["degrees_of_freedom"] = student_t_degrees_of_freedom
        if sticky_transition_prior > 0.0:
            model_kwargs["transmat_prior"] = (
                np.ones((states, states), dtype=float)
                + sticky_transition_prior * np.eye(states, dtype=float)
            )
        model = model_class(
            n_components=states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            min_covar=covariance_floor,
            random_state=random_seed + 1009 * restart + states,
            **model_kwargs,
        )
        hmmlearn_logger = logging.getLogger("hmmlearn.base")
        previous_level = hmmlearn_logger.level
        try:
            hmmlearn_logger.setLevel(logging.ERROR)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(observations)
                score = model.score(observations)
        finally:
            hmmlearn_logger.setLevel(previous_level)
        history = list(model.monitor_.history)
        final_delta = (
            float(history[-1] - history[-2]) if len(history) >= 2 else float("nan")
        )
        restart_diagnostics.append(
            {
                "restart": restart,
                "score": float(score),
                "iterations": int(model.monitor_.iter),
                "final_delta": final_delta,
            }
        )
        if np.isfinite(score) and score > best_score:
            best, best_score, best_restart = model, score, restart
    if best is None:
        raise RuntimeError(f"All {states}-state HMM fits failed")
    best.restart_diagnostics_ = restart_diagnostics
    best.best_restart_ = best_restart
    return best


def hmm_fit_diagnostics(model: BaseHMM) -> dict[str, float | int]:
    """Summarize restart and terminal EM behavior attached by `_fit_hmm`."""
    records = list(getattr(model, "restart_diagnostics_", []))
    if not records:
        return {
            "hmm_fit_iterations": int(model.monitor_.iter),
            "hmm_fit_final_delta": float("nan"),
            "hmm_fit_hit_iteration_limit": int(
                model.monitor_.iter >= model.monitor_.n_iter
            ),
            "hmm_nonmonotonic_restarts": 0,
            "hmm_restart_score_range": float("nan"),
            "hmm_best_restart_margin": float("nan"),
        }
    best_restart = int(getattr(model, "best_restart_"))
    best_record = next(
        record for record in records if int(record["restart"]) == best_restart
    )
    scores = np.asarray([float(record["score"]) for record in records])
    ordered = np.sort(scores)
    margin = float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else float("nan")
    return {
        "hmm_fit_iterations": int(best_record["iterations"]),
        "hmm_fit_final_delta": float(best_record["final_delta"]),
        "hmm_fit_hit_iteration_limit": int(
            int(best_record["iterations"]) >= model.monitor_.n_iter
        ),
        "hmm_nonmonotonic_restarts": int(
            sum(float(record["final_delta"]) < 0.0 for record in records)
        ),
        "hmm_restart_score_range": float(scores.max() - scores.min()),
        "hmm_best_restart_margin": margin,
    }


@dataclass
class RegimeFit:
    model: GaussianHMM
    scaler: FeatureScaler
    raw_means: np.ndarray
    raw_variances: np.ndarray
    state_return_means: np.ndarray
    state_return_covariances: np.ndarray
    training_features: pd.DataFrame
    feature_lower_bounds: pd.Series
    feature_upper_bounds: pd.Series
    order: int

    def transformed_features(
        self, features_through_signal: pd.DataFrame
    ) -> np.ndarray:
        clipped = clip_features(
            features_through_signal,
            self.feature_lower_bounds,
            self.feature_upper_bounds,
        )
        return self.scaler.transform(clipped)

    def filtered_probabilities(self, features_through_signal: pd.DataFrame) -> np.ndarray:
        transformed = self.transformed_features(features_through_signal)
        # At the final observation, smoothed and filtered probabilities coincide.
        posterior = self.model.predict_proba(transformed)
        probabilities = np.maximum(posterior[-1], 0.0)
        return probabilities / probabilities.sum()

    def filter_diagnostics(
        self,
        features_through_signal: pd.DataFrame,
        probabilities: np.ndarray | None = None,
    ) -> dict[str, float]:
        transformed = self.transformed_features(features_through_signal)
        selected = (
            self.filtered_probabilities(features_through_signal)
            if probabilities is None
            else np.asarray(probabilities, dtype=float)
        )
        predictive_log_likelihood = float("nan")
        if len(transformed) >= 2:
            predictive_log_likelihood = float(
                self.model.score(transformed)
                - self.model.score(transformed[:-1])
            )
        return {
            "state_entropy": normalized_probability_entropy(selected),
            "state_probability_margin": probability_margin(selected),
            "one_step_predictive_log_likelihood": predictive_log_likelihood,
        }


class CausalHMMEstimator:
    def __init__(self, config: dict[str, object]):
        self.config = config

    def select_order(self, features: pd.DataFrame) -> tuple[int, dict[int, float]]:
        validation_days = int(self.config["validation_days"])
        if len(features) <= validation_days + 100:
            raise ValueError("Insufficient training history for predictive order selection")
        raw_train = features.iloc[:-validation_days]
        raw_validation = features.iloc[-validation_days:]
        lower, upper = feature_clip_bounds(
            raw_train,
            float(self.config.get("feature_clip_quantile", 0.0)),
        )
        clipped_train = clip_features(raw_train, lower, upper)
        scaler = make_feature_scaler(
            str(self.config.get("feature_scaler", "standard"))
        ).fit(clipped_train)
        train = scaler.transform(clipped_train)
        validation = scaler.transform(
            clip_features(raw_validation, lower, upper)
        )
        scores: dict[int, float] = {}
        for states in self.config["candidate_states"]:  # type: ignore[union-attr]
            states = int(states)
            model = _fit_hmm(
                train,
                states,
                int(self.config["n_iter"]),
                int(self.config["restarts"]),
                int(self.config["random_seed"]),
                float(self.config["covariance_floor"]),
                str(self.config.get("emission_distribution", "gaussian")),
                float(self.config.get("student_t_degrees_of_freedom", 5.0)),
                float(self.config.get("sticky_transition_prior", 0.0)),
                str(self.config.get("covariance_type", "diag")),
            )
            predictive = validation_predictive_score(
                model,
                train,
                validation,
                bool(self.config.get("condition_order_validation_on_train", False)),
            )
            penalty = (
                float(self.config["complexity_penalty"])
                * _parameter_count(
                    states,
                    train.shape[1],
                    str(self.config.get("covariance_type", "diag")),
                )
                / len(train)
            )
            scores[states] = float(predictive - penalty)
        return max(scores, key=scores.get), scores

    def fit(
        self,
        features: pd.DataFrame,
        outcome_returns: pd.DataFrame,
        order: int,
    ) -> RegimeFit:
        aligned_returns = outcome_returns.reindex(features.index)
        valid = aligned_returns.notna().all(axis=1)
        features = features.loc[valid]
        aligned_returns = aligned_returns.loc[valid]
        lower, upper = feature_clip_bounds(
            features,
            float(self.config.get("feature_clip_quantile", 0.0)),
        )
        clipped_features = clip_features(features, lower, upper)
        scaler = make_feature_scaler(
            str(self.config.get("feature_scaler", "standard"))
        ).fit(clipped_features)
        observations = scaler.transform(clipped_features)
        model = _fit_hmm(
            observations,
            order,
            int(self.config["n_iter"]),
            int(self.config["restarts"]),
            int(self.config["random_seed"]),
            float(self.config["covariance_floor"]),
            str(self.config.get("emission_distribution", "gaussian")),
            float(self.config.get("student_t_degrees_of_freedom", 5.0)),
            float(self.config.get("sticky_transition_prior", 0.0)),
            str(self.config.get("covariance_type", "diag")),
        )
        posterior_mode = str(
            self.config.get("return_moment_posterior", "smoothed")
        )
        if posterior_mode == "filtered":
            posterior = filtered_state_posteriors(model, observations)
        elif posterior_mode == "smoothed":
            posterior = model.predict_proba(observations)
        else:
            raise ValueError(
                f"Unsupported return-moment posterior mode: {posterior_mode}"
            )
        return_means: list[np.ndarray] = []
        return_covariances: list[np.ndarray] = []
        values = aligned_returns.to_numpy()
        for state in range(order):
            mean, covariance = _weighted_return_moments(
                values,
                posterior[:, state],
                float(self.config["covariance_floor"]),
                str(self.config.get("return_location_estimator", "mean")),
            )
            return_means.append(mean)
            return_covariances.append(covariance)

        state_return_means = np.asarray(return_means)
        state_return_covariances = np.asarray(return_covariances)
        return_mean_mode = str(
            self.config.get("state_return_mean_shrinkage", "none")
        )
        if return_mean_mode == "empirical_bayes":
            state_return_means = empirical_bayes_state_return_means(
                values,
                posterior,
                state_return_means,
                state_return_covariances,
            )
        elif return_mean_mode != "none":
            raise ValueError(
                f"Unsupported state return mean shrinkage: {return_mean_mode}"
            )

        raw_means, raw_variances = inverse_emission_marginals(
            scaler,
            model.means_,
            emission_marginal_variances(model),
        )
        return RegimeFit(
            model=model,
            scaler=scaler,
            raw_means=raw_means,
            raw_variances=raw_variances,
            state_return_means=state_return_means,
            state_return_covariances=state_return_covariances,
            training_features=clipped_features,
            feature_lower_bounds=lower,
            feature_upper_bounds=upper,
            order=order,
        )


@dataclass
class RegimeTemplate:
    feature_mean: np.ndarray
    feature_variance: np.ndarray
    return_mean: np.ndarray
    return_covariance: np.ndarray


class WassersteinTemplateTracker:
    def __init__(
        self,
        template_count: int,
        smoothing: float,
        assignment_mode: str = "nearest",
        update_geometry: str = "euclidean_variance",
    ):
        self.template_count = template_count
        self.smoothing = smoothing
        self.assignment_mode = assignment_mode
        if update_geometry not in {"euclidean_variance", "wasserstein_diag"}:
            raise ValueError(f"Unsupported template update geometry: {update_geometry}")
        self.update_geometry = update_geometry
        self.templates: list[RegimeTemplate] = []
        self.last_distance_matrix = np.empty((0, 0), dtype=float)

    def initialize(self, calibration: RegimeFit) -> None:
        # Low-to-high equity-lag mean produces deterministic initial identities.
        order = np.argsort(calibration.raw_means[:, 0])
        self.templates = [
            RegimeTemplate(
                calibration.raw_means[state].copy(),
                calibration.raw_variances[state].copy(),
                calibration.state_return_means[state].copy(),
                calibration.state_return_covariances[state].copy(),
            )
            for state in order[: self.template_count]
        ]

    def map_and_update(
        self,
        fit: RegimeFit,
        state_probabilities: np.ndarray,
        update: bool = True,
    ) -> tuple[np.ndarray, list[int]]:
        if not self.templates:
            self.initialize(fit)
        distance_rows: list[list[float]] = []
        for state in range(fit.order):
            distance_rows.append(
                [
                gaussian_w2_diag(
                    template.feature_mean,
                    template.feature_variance,
                    fit.raw_means[state],
                    fit.raw_variances[state],
                )
                for template in self.templates
                ]
            )
        self.last_distance_matrix = np.asarray(distance_rows, dtype=float)
        mapping = assign_templates(
            self.last_distance_matrix,
            self.assignment_mode,
        )

        template_probabilities = np.zeros(len(self.templates))
        for state, template_index in enumerate(mapping):
            template_probabilities[template_index] += state_probabilities[state]

        if not update:
            return template_probabilities, mapping

        eta = self.smoothing
        for template_index, template in enumerate(self.templates):
            assigned = [i for i, target in enumerate(mapping) if target == template_index]
            if not assigned:
                continue
            local_weights = state_probabilities[assigned]
            if local_weights.sum() <= 1e-12:
                local_weights = np.ones(len(assigned))
            local_weights = local_weights / local_weights.sum()
            feature_mean = np.average(fit.raw_means[assigned], axis=0, weights=local_weights)
            feature_variance = np.average(
                fit.raw_variances[assigned], axis=0, weights=local_weights
            )
            return_mean = np.average(
                fit.state_return_means[assigned], axis=0, weights=local_weights
            )
            return_covariance = np.average(
                fit.state_return_covariances[assigned], axis=0, weights=local_weights
            )
            template.feature_mean = (1 - eta) * template.feature_mean + eta * feature_mean
            if self.update_geometry == "wasserstein_diag":
                old_scale = np.sqrt(np.maximum(template.feature_variance, 0.0))
                new_scale = np.sqrt(np.maximum(feature_variance, 0.0))
                template.feature_variance = (
                    (1 - eta) * old_scale + eta * new_scale
                ) ** 2
            else:
                template.feature_variance = (
                    (1 - eta) * template.feature_variance + eta * feature_variance
                )
            template.return_mean = (1 - eta) * template.return_mean + eta * return_mean
            template.return_covariance = _nearest_psd(
                (1 - eta) * template.return_covariance + eta * return_covariance
            )
        return template_probabilities, mapping

    def distance_diagnostics(
        self,
        state_probabilities: np.ndarray,
    ) -> dict[str, float]:
        if self.last_distance_matrix.size == 0:
            return {
                "template_expected_nearest_distance": float("nan"),
                "template_expected_distance_margin": float("nan"),
            }
        probabilities = np.maximum(
            np.asarray(state_probabilities, dtype=float), 0.0
        )
        probabilities /= probabilities.sum()
        ordered = np.sort(self.last_distance_matrix, axis=1)
        nearest = ordered[:, 0]
        margin = (
            ordered[:, 1] - ordered[:, 0]
            if ordered.shape[1] >= 2
            else np.full(len(ordered), np.nan)
        )
        return {
            "template_expected_nearest_distance": float(probabilities @ nearest),
            "template_expected_distance_margin": float(probabilities @ margin),
        }

    def conditional_moments(
        self, template_probabilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.templates:
            raise RuntimeError("Templates are not initialized")
        probabilities = np.asarray(template_probabilities, dtype=float)
        if probabilities.sum() <= 1e-12:
            probabilities = np.ones(len(self.templates))
        probabilities = probabilities / probabilities.sum()
        mean = sum(
            probability * template.return_mean
            for probability, template in zip(probabilities, self.templates, strict=True)
        )
        covariance = sum(
            probability * template.return_covariance
            for probability, template in zip(probabilities, self.templates, strict=True)
        )
        return np.asarray(mean), _nearest_psd(np.asarray(covariance))

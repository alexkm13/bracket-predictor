"""
Bayesian Hierarchical Measurement Error Model for NCAA Tournament Prediction.

Architecture:
    Seed priors -> Team strengths -> Observed ratings (sensor fusion) + Game margins

Each rating source is a noisy thermometer measuring team strength θ.
The model learns each source's bias (a[k]) and scale (b[k]), fuses all
readings with a structural seed prior, and validates against game margins.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Optional


def build_model(
    ratings: np.ndarray,
    seeds: np.ndarray,
    team_indices: np.ndarray,
    year_indices: np.ndarray,
    margins: Optional[np.ndarray] = None,
    team_i_game: Optional[np.ndarray] = None,
    team_j_game: Optional[np.ndarray] = None,
    n_teams: Optional[int] = None,
) -> pm.Model:
    """
    Build the full hierarchical model.

    Parameters
    ----------
    ratings : (N_teams, K_sources) array — z-scored ratings, NaN=missing
    seeds : (N_teams,) — seeds 1-16
    team_indices : (N_teams,) — unique index per team-season
    year_indices : (N_teams,) — year index (not used in v1)
    margins : (N_games,) or None — point margins for game layer
    team_i_game : (N_games,) or None — team indices for game layer
    team_j_game : (N_games,) or None — team indices for game layer
    """
    if n_teams is None:
        n_teams = ratings.shape[0]

    n_sources = ratings.shape[1]
    has_games = margins is not None and len(margins) > 0

    # Precompute observation masks (avoid NaN in likelihood)
    obs_masks = []
    obs_values = []
    obs_team_idx = []
    for k in range(n_sources):
        mask = ~np.isnan(ratings[:, k])
        obs_masks.append(mask)
        obs_values.append(ratings[mask, k].astype(np.float64))
        obs_team_idx.append(np.where(mask)[0].astype(np.int64))

    with pm.Model() as model:

        # =============================================================
        # Level 1: Seed Structure
        # =============================================================
        alpha = pm.Normal("alpha", mu=0, sigma=50)
        beta = pm.Normal("beta", mu=-2, sigma=2)
        sigma_seed = pm.HalfNormal("sigma_seed", sigma=5)

        seed_vals = pt.arange(1, 17).astype("float64")
        mu_seed_linear = alpha + beta * seed_vals
        mu_seed = pm.Normal("mu_seed", mu=mu_seed_linear, sigma=sigma_seed, shape=16)

        # Map each team to its seed mean (seeds are 1-indexed)
        team_seed_mu = mu_seed[seeds - 1]

        # =============================================================
        # Level 2: Team Strengths
        # =============================================================
        sigma_team = pm.HalfNormal("sigma_team", sigma=5)
        nu_team = 7  # Fixed

        theta = pm.StudentT("theta", nu=nu_team, mu=team_seed_mu,
                            sigma=sigma_team, shape=n_teams)

        # =============================================================
        # Level 3: Observation Model (Sensor Fusion)
        # Source 0 is anchor: a=0, b=1
        # Sources 1..K-1: learn a[k], b[k] = exp(eta[k])
        # =============================================================
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=5, shape=n_sources)

        if n_sources > 1:
            eta = pm.Normal("eta", mu=0, sigma=0.3, shape=n_sources - 1)
            b_nonanchor = pm.Deterministic("b_nonanchor", pt.exp(eta))
            a_nonanchor = pm.Normal("a_nonanchor", mu=0, sigma=10, shape=n_sources - 1)

        for k in range(n_sources):
            idx = obs_team_idx[k]
            y_obs = obs_values[k]

            if len(idx) == 0:
                continue

            theta_k = theta[idx]

            if k == 0:
                mu_k = theta_k
            else:
                mu_k = a_nonanchor[k - 1] + b_nonanchor[k - 1] * theta_k

            pm.Normal(f"obs_source_{k + 1}", mu=mu_k, sigma=sigma_obs[k],
                      observed=y_obs)

        # =============================================================
        # Level 4: Game Outcomes (2008-2019 only)
        # =============================================================
        if has_games:
            sigma_game = pm.HalfNormal("sigma_game", sigma=12)
            nu_game = 7

            delta = theta[team_i_game] - theta[team_j_game]
            pm.StudentT("game_margin", nu=nu_game, mu=delta,
                        sigma=sigma_game, observed=margins)

    return model


def build_prediction_model(
    new_ratings: np.ndarray,
    new_seeds: np.ndarray,
    trace_params: dict,
) -> dict:
    """
    Two-step out-of-sample prediction for new teams.

    For each posterior draw:
        1. Start with seed prior (mu_seed[s], sigma_team)
        2. Update with each observed rating via Gaussian conjugate update
        3. Draw theta from approximate posterior

    Parameters
    ----------
    new_ratings : (N_new, K) — z-scored ratings, NaN=missing
    new_seeds : (N_new,) — seeds 1-16
    trace_params : dict with posterior samples:
        'mu_seed' (D, 16), 'sigma_team' (D,), 'sigma_obs' (D, K),
        'a_nonanchor' (D, K-1), 'b_nonanchor' (D, K-1), 'sigma_game' (D,)

    Returns
    -------
    dict: 'theta' (n_draws, N_new), 'sigma_game' (n_draws,)
    """
    n_draws = trace_params["mu_seed"].shape[0]
    n_teams = new_ratings.shape[0]
    n_sources = new_ratings.shape[1]

    rng = np.random.default_rng(42)
    theta_samples = np.zeros((n_draws, n_teams))

    for d in range(n_draws):
        mu_seed_d = trace_params["mu_seed"][d]
        sigma_team_d = trace_params["sigma_team"][d]

        for i in range(n_teams):
            seed_idx = int(new_seeds[i]) - 1
            prior_mu = mu_seed_d[seed_idx]
            prior_var = sigma_team_d ** 2

            post_mu = prior_mu
            post_var = prior_var
            sigma_obs_d = trace_params["sigma_obs"][d]

            for k in range(n_sources):
                if np.isnan(new_ratings[i, k]):
                    continue

                y_ik = new_ratings[i, k]
                obs_var = sigma_obs_d[k] ** 2

                if k == 0:
                    likelihood_mu = y_ik
                    likelihood_var = obs_var
                else:
                    a_k = trace_params["a_nonanchor"][d, k - 1]
                    b_k = trace_params["b_nonanchor"][d, k - 1]
                    likelihood_mu = (y_ik - a_k) / b_k
                    likelihood_var = obs_var / (b_k ** 2)

                # Gaussian precision-weighted update
                post_prec = 1.0 / post_var + 1.0 / likelihood_var
                post_mu = (post_mu / post_var + likelihood_mu / likelihood_var) / post_prec
                post_var = 1.0 / post_prec

            theta_samples[d, i] = rng.normal(post_mu, np.sqrt(max(post_var, 1e-10)))

    return {
        "theta": theta_samples,
        "sigma_game": trace_params["sigma_game"],
    }

"""
Backtesting and Diagnostics.

Leave-one-year-out cross-validation + calibration + log score.

Usage:
    python diagnostics.py                              # Full CV (slow: fits 12 models)
    python diagnostics.py --trace output/trace.nc      # Quick diagnostics on existing trace
    python diagnostics.py --chains 2 --draws 500       # Fast CV for debugging
"""

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import t as student_t

from model import build_model, build_prediction_model
from simulate import extract_trace_params, win_probability


# ===========================================================================
# Leave-One-Year-Out CV
# ===========================================================================

def leave_one_year_out_cv(
    data_dir: str = "data/processed",
    chains: int = 2,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.95,
) -> pd.DataFrame:
    """
    For each year Y in 2008-2019:
        1. Fit model on all data EXCEPT year Y's games
        2. Use year Y's ratings through observation model (no leakage)
        3. Compute win probabilities for year Y's games

    Returns DataFrame: season, team_i, team_j, seed_i, seed_j, p_i, margin, correct
    """
    data_dir = Path(data_dir)

    ratings_df = pd.read_csv(data_dir / "ratings_matrix_standardized.csv")
    games_df = pd.read_csv(data_dir / "tournament_games.csv")
    source_cols = [c for c in ratings_df.columns if c.startswith("source_")]

    game_years = sorted(y for y in games_df["season"].unique() if 2008 <= y <= 2019)
    all_predictions = []

    for holdout_year in game_years:
        print(f"\n{'='*60}")
        print(f"Holdout year: {holdout_year} ({game_years.index(holdout_year)+1}/{len(game_years)})")
        print(f"{'='*60}")

        # Global team index (ALL years feed observation model)
        team_keys = list(zip(ratings_df["season"].astype(int), ratings_df["team"]))
        key_to_idx = {k: i for i, k in enumerate(team_keys)}

        ratings = ratings_df[source_cols].values.astype(np.float64)
        seeds = ratings_df["seed"].values.astype(np.int64)

        # Training games: EXCLUDE holdout year
        train_games = games_df[
            (games_df["season"] != holdout_year) &
            (games_df["season"].between(2008, 2019))
        ].copy()

        margins_list, ti_list, tj_list = [], [], []
        for _, row in train_games.iterrows():
            ki = (int(row["season"]), str(row["team_i"]).strip())
            kj = (int(row["season"]), str(row["team_j"]).strip())
            if ki in key_to_idx and kj in key_to_idx:
                ti_list.append(key_to_idx[ki])
                tj_list.append(key_to_idx[kj])
                margins_list.append(float(row["margin"]))

        margins = np.array(margins_list, dtype=np.float64)
        team_i_game = np.array(ti_list, dtype=np.int64)
        team_j_game = np.array(tj_list, dtype=np.int64)

        print(f"  Training games: {len(margins)} (excluding {holdout_year})")

        # Fit
        model = build_model(
            ratings=ratings,
            seeds=seeds,
            team_indices=np.arange(len(team_keys)),
            year_indices=np.zeros(len(team_keys), dtype=np.int64),
            margins=margins,
            team_i_game=team_i_game,
            team_j_game=team_j_game,
        )

        with model:
            trace = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept,
                random_seed=holdout_year,
                return_inferencedata=True,
                progressbar=True,
            )

        # Get theta and sigma_game from trace
        theta_post = trace.posterior["theta"].values
        theta_post = theta_post.reshape(-1, theta_post.shape[-1])
        sigma_game_post = trace.posterior["sigma_game"].values.flatten()

        # Predict holdout games
        holdout_games = games_df[games_df["season"] == holdout_year].copy()
        n_pred = 0

        for _, row in holdout_games.iterrows():
            ki = (int(row["season"]), str(row["team_i"]).strip())
            kj = (int(row["season"]), str(row["team_j"]).strip())

            if ki not in key_to_idx or kj not in key_to_idx:
                continue

            idx_i, idx_j = key_to_idx[ki], key_to_idx[kj]

            # Average win probability across posterior draws
            n_draws_use = min(theta_post.shape[0], 2000)  # cap for speed
            draw_indices = np.random.choice(theta_post.shape[0], n_draws_use, replace=False)
            
            probs = np.array([
                win_probability(theta_post[d, idx_i], theta_post[d, idx_j], sigma_game_post[d])
                for d in draw_indices
            ])
            p_i_mean = float(probs.mean())
            actual_win = 1 if row["margin"] > 0 else 0

            all_predictions.append({
                "season": int(row["season"]),
                "team_i": row["team_i"],
                "team_j": row["team_j"],
                "seed_i": int(row["seed_i"]),
                "seed_j": int(row["seed_j"]),
                "p_i": p_i_mean,
                "margin": float(row["margin"]),
                "correct": int((p_i_mean > 0.5) == (actual_win == 1)),
            })
            n_pred += 1

        print(f"  Holdout predictions: {n_pred} games")

    return pd.DataFrame(all_predictions)


# ===========================================================================
# Calibration
# ===========================================================================

def calibration_plot(predictions, n_bins=8, output_path="output/calibration.png"):
    """
    Plot predicted win probability vs actual win rate.
    Perfect calibration = diagonal line.
    """
    p_values = predictions["p_i"].values
    outcomes = (predictions["margin"] > 0).astype(float).values

    bin_edges = np.linspace(0, 1, n_bins + 1)
    actual_rates = []
    predicted_means = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p_values >= bin_edges[i]) & (p_values <= bin_edges[i + 1])
        else:
            mask = (p_values >= bin_edges[i]) & (p_values < bin_edges[i + 1])

        n = mask.sum()
        if n > 0:
            actual_rates.append(outcomes[mask].mean())
            predicted_means.append(p_values[mask].mean())
            bin_counts.append(n)
        else:
            actual_rates.append(np.nan)
            predicted_means.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    # ECE (Expected Calibration Error)
    ece = 0.0
    total = sum(bin_counts)
    for i in range(n_bins):
        if bin_counts[i] > 0 and not np.isnan(actual_rates[i]):
            ece += bin_counts[i] / total * abs(actual_rates[i] - predicted_means[i])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    
    valid = [i for i in range(n_bins) if bin_counts[i] > 0 and not np.isnan(actual_rates[i])]
    ax.scatter(
        [predicted_means[i] for i in valid],
        [actual_rates[i] for i in valid],
        s=[bin_counts[i] * 3 for i in valid],
        c="steelblue", alpha=0.8, edgecolors="white", linewidths=1,
        label="Model",
    )

    # Error bars (approximate binomial CI)
    for i in valid:
        se = np.sqrt(actual_rates[i] * (1 - actual_rates[i]) / max(bin_counts[i], 1))
        ax.errorbar(predicted_means[i], actual_rates[i], yerr=1.96 * se,
                    color="steelblue", alpha=0.5, capsize=3)

    ax.set_xlabel("Predicted Win Probability", fontsize=12)
    ax.set_ylabel("Actual Win Rate", fontsize=12)
    ax.set_title(f"Calibration Plot (ECE = {ece:.4f})", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Annotate bin counts
    for i in valid:
        ax.annotate(f"n={bin_counts[i]}", (predicted_means[i], actual_rates[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to {output_path}")

    return {"ece": float(ece), "n_bins": n_bins, "total_games": int(total)}


# ===========================================================================
# Log Score
# ===========================================================================

def compute_log_score(predictions):
    """Compare model log score to naive seed baseline."""
    eps = 1e-6
    p_values = predictions["p_i"].values
    outcomes = (predictions["margin"] > 0).astype(float).values

    # Model log score
    p_clipped = np.clip(p_values, eps, 1 - eps)
    model_scores = outcomes * np.log(p_clipped) + (1 - outcomes) * np.log(1 - p_clipped)
    model_log_score = model_scores.mean()

    # Seed baseline: historical win rates by seed matchup
    seed_win_rates = {
        (1, 16): 0.99, (2, 15): 0.94, (3, 14): 0.85, (4, 13): 0.80,
        (5, 12): 0.64, (6, 11): 0.62, (7, 10): 0.61, (8, 9): 0.51,
    }

    baseline_scores = []
    for _, row in predictions.iterrows():
        s_i, s_j = int(row["seed_i"]), int(row["seed_j"])
        if s_i < s_j:
            p_base = seed_win_rates.get((s_i, s_j), 0.5 + 0.02 * (s_j - s_i))
        elif s_j < s_i:
            p_base = 1 - seed_win_rates.get((s_j, s_i), 0.5 + 0.02 * (s_i - s_j))
        else:
            p_base = 0.5

        p_b = np.clip(p_base, eps, 1 - eps)
        outcome = 1 if row["margin"] > 0 else 0
        baseline_scores.append(outcome * np.log(p_b) + (1 - outcome) * np.log(1 - p_b))

    baseline_log_score = np.mean(baseline_scores)
    model_accuracy = float(((p_values > 0.5) == outcomes).mean())

    results = {
        "model_log_score": float(model_log_score),
        "baseline_log_score": float(baseline_log_score),
        "improvement": float(model_log_score - baseline_log_score),
        "model_accuracy": float(model_accuracy),
        "n_games": int(len(predictions)),
    }

    print(f"\nLog Score Comparison:")
    print(f"  Model:    {model_log_score:.4f}")
    print(f"  Baseline: {baseline_log_score:.4f}")
    print(f"  Δ:        {model_log_score - baseline_log_score:+.4f} "
          f"({'better' if model_log_score > baseline_log_score else 'worse'})")
    print(f"  Accuracy: {model_accuracy:.1%} ({int(model_accuracy * len(predictions))}/{len(predictions)})")

    return results


# ===========================================================================
# Trace Diagnostics (quick, no re-fitting)
# ===========================================================================

def trace_diagnostics(trace_path, output_dir="output"):
    """Quick diagnostics on a pre-fitted trace."""
    print("Loading trace...")
    trace = az.from_netcdf(trace_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available_vars = list(trace.posterior.data_vars)
    key_vars = [v for v in ["alpha", "beta", "sigma_seed", "sigma_team",
                             "sigma_obs", "sigma_game", "mu_seed"] if v in available_vars]

    summary = az.summary(trace, var_names=key_vars)
    print("\nParameter Summary:")
    print(summary.to_string())

    # Trace plots
    plot_vars = [v for v in ["alpha", "beta", "sigma_seed", "sigma_team", "sigma_game"]
                 if v in available_vars]
    if plot_vars:
        az.plot_trace(trace, var_names=plot_vars, compact=True)
        plt.tight_layout()
        plt.savefig(output_dir / "trace_plots.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Trace plots saved to {output_dir / 'trace_plots.png'}")

    # Seed-strength relationship
    if "mu_seed" in available_vars:
        mu_seed = trace.posterior["mu_seed"].values.reshape(-1, 16)
        fig, ax = plt.subplots(figsize=(10, 6))
        seeds = np.arange(1, 17)
        ax.plot(seeds, mu_seed.mean(axis=0), "o-", color="steelblue", lw=2, label="Posterior mean")
        ax.fill_between(seeds,
                        np.percentile(mu_seed, 5, axis=0),
                        np.percentile(mu_seed, 95, axis=0),
                        alpha=0.2, color="steelblue", label="90% CI")
        ax.set_xlabel("Seed", fontsize=12)
        ax.set_ylabel("μ_seed (z-scored strength)", fontsize=12)
        ax.set_title("Seed → Strength Relationship (Posterior)", fontsize=14)
        ax.set_xticks(seeds)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "seed_strength.png", dpi=150)
        plt.close()
        print(f"Seed-strength plot saved to {output_dir / 'seed_strength.png'}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Model diagnostics and backtesting")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--trace", type=str, default=None,
                        help="Existing trace for quick diagnostics (skip CV)")
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.95)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick diagnostics mode
    if args.trace:
        trace_diagnostics(args.trace, args.output_dir)
        return

    # Full leave-one-year-out CV
    print("=" * 60)
    print("Leave-One-Year-Out Cross-Validation")
    print("=" * 60)
    print(f"Fits 12 models. Settings: {args.chains} chains × {args.draws} draws")
    print(f"Estimated time: {args.chains * args.draws * 12 / 500:.0f}+ minutes\n")

    predictions = leave_one_year_out_cv(
        data_dir=args.data_dir,
        chains=args.chains,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
    )

    predictions.to_csv(output_dir / "cv_predictions.csv", index=False)
    print(f"\nCV predictions saved to {output_dir / 'cv_predictions.csv'}")

    # Calibration
    print("\n" + "=" * 60)
    print("Calibration Analysis")
    print("=" * 60)
    cal = calibration_plot(predictions, n_bins=8,
                           output_path=str(output_dir / "calibration.png"))
    print(f"ECE: {cal['ece']:.4f}")

    # Log Score
    print("\n" + "=" * 60)
    print("Log Score Analysis")
    print("=" * 60)
    scores = compute_log_score(predictions)

    # Save everything
    all_results = {"calibration": cal, "log_score": scores}
    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")
    print("✓ Backtesting complete.")


if __name__ == "__main__":
    main()

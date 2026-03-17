"""
Fit the hierarchical measurement error model via MCMC.

Usage (run from project root, not etl/):
    python model/fit.py --data-dir etl/data/processed --output-dir output

Or from etl/:
    python fit.py --data-dir data/processed --output-dir ../output
"""

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from model import build_model


def load_data(data_dir: str = "data/processed") -> tuple:
    """Load and prepare all data for model fitting."""
    data_dir = Path(data_dir)

    # --- Ratings matrix ---
    ratings_path = data_dir / "ratings_matrix_standardized.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings matrix not found at {ratings_path}\n"
            f"Run preprocess.py first to generate it."
        )

    ratings_df = pd.read_csv(ratings_path)
    source_cols = [c for c in ratings_df.columns if c.startswith("source_")]

    # Build global team-season index
    team_keys = list(zip(ratings_df["season"].astype(int), ratings_df["team"]))
    key_to_idx = {k: i for i, k in enumerate(team_keys)}

    ratings = ratings_df[source_cols].values.astype(np.float64)
    seeds = ratings_df["seed"].values.astype(np.int64)
    seasons = ratings_df["season"].values.astype(np.int64)

    print(f"Loaded {len(team_keys)} team-seasons across {len(set(seasons))} years")
    print(f"Sources: {source_cols}")
    for i, col in enumerate(source_cols):
        n = (~np.isnan(ratings[:, i])).sum()
        print(f"  {col}: {n}/{len(team_keys)} ({n/len(team_keys)*100:.1f}%)")

    # --- Game outcomes (2008-2019) ---
    games_path = data_dir / "tournament_games.csv"
    if not games_path.exists():
        print(f"WARNING: No game results found at {games_path}")
        print(f"Model will fit observation layer only (no game layer)")
        margins = np.array([], dtype=np.float64)
        team_i_game = np.array([], dtype=np.int64)
        team_j_game = np.array([], dtype=np.int64)
        game_years = np.array([], dtype=np.int64)
    else:
        games_df = pd.read_csv(games_path)
        games_df = games_df[games_df["season"].between(2008, 2019)].copy()

        margins_list, ti_list, tj_list, gy_list = [], [], [], []
        skipped = 0

        for _, row in games_df.iterrows():
            key_i = (int(row["season"]), str(row["team_i"]).strip())
            key_j = (int(row["season"]), str(row["team_j"]).strip())

            if key_i not in key_to_idx or key_j not in key_to_idx:
                skipped += 1
                continue

            ti_list.append(key_to_idx[key_i])
            tj_list.append(key_to_idx[key_j])
            margins_list.append(float(row["margin"]))
            gy_list.append(int(row["season"]))

        if skipped > 0:
            print(f"\nWARNING: {skipped}/{len(games_df)} games skipped (team name mismatch)")
            # Show some examples
            skip_examples = []
            for _, row in games_df.iterrows():
                key_i = (int(row["season"]), str(row["team_i"]).strip())
                key_j = (int(row["season"]), str(row["team_j"]).strip())
                if key_i not in key_to_idx:
                    skip_examples.append(f"  {key_i[0]} {key_i[1]}")
                if key_j not in key_to_idx:
                    skip_examples.append(f"  {key_j[0]} {key_j[1]}")
            for ex in sorted(set(skip_examples))[:10]:
                print(ex)

        margins = np.array(margins_list, dtype=np.float64)
        team_i_game = np.array(ti_list, dtype=np.int64)
        team_j_game = np.array(tj_list, dtype=np.int64)
        game_years = np.array(gy_list, dtype=np.int64)

        print(f"\nLoaded {len(margins)} games for game layer (2008-2019)")

    return (ratings, seeds, seasons, team_keys, key_to_idx,
            margins, team_i_game, team_j_game, game_years)


def print_diagnostics(trace: az.InferenceData) -> dict:
    """Print and return MCMC diagnostics."""
    diagnostics = {}

    # Divergences
    if hasattr(trace, "sample_stats"):
        divs = int(trace.sample_stats.diverging.values.sum())
        diagnostics["divergences"] = divs
        print(f"\nDivergences: {divs}")
        if divs > 0:
            print("  ⚠ Consider increasing target_accept or reparameterizing")

    # Summary for key parameters
    var_names = ["alpha", "beta", "sigma_seed", "sigma_team", "sigma_game", "sigma_obs", "mu_seed"]
    # Only include vars that exist in trace
    available = [v for v in var_names if v in trace.posterior]
    
    summary = az.summary(trace, var_names=available, round_to=4)
    print("\nParameter Summary:")
    print(summary.to_string())

    rhat_cols = [c for c in summary.columns if "r_hat" in c.lower()]
    if rhat_cols:
        max_rhat = summary[rhat_cols[0]].max()
        diagnostics["max_rhat"] = float(max_rhat)
        status = "✓" if max_rhat <= 1.01 else "⚠"
        print(f"\n  {status} Max R-hat = {max_rhat:.4f}")

    ess_cols = [c for c in summary.columns if "ess_bulk" in c.lower()]
    if ess_cols:
        min_ess = summary[ess_cols[0]].min()
        diagnostics["min_ess_bulk"] = float(min_ess)
        print(f"  Min ESS (bulk) = {min_ess:.0f}")

    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="Fit NCAA tournament model")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    (ratings, seeds, seasons, team_keys, key_to_idx,
     margins, team_i_game, team_j_game, game_years) = load_data(args.data_dir)

    unique_years = sorted(set(seasons))
    year_to_idx = {y: i for i, y in enumerate(unique_years)}
    year_indices = np.array([year_to_idx[y] for y in seasons])

    # Build model
    print("\n" + "=" * 60)
    print("Building model...")
    print("=" * 60)
    model = build_model(
        ratings=ratings,
        seeds=seeds,
        team_indices=np.arange(len(team_keys)),
        year_indices=year_indices,
        margins=margins if len(margins) > 0 else None,
        team_i_game=team_i_game if len(margins) > 0 else None,
        team_j_game=team_j_game if len(margins) > 0 else None,
    )

    print(f"Free RVs: {[v.name for v in model.free_RVs]}")
    print(f"Observed RVs: {[v.name for v in model.observed_RVs]}")
    print(f"Total parameters: ~{sum(v.eval().size for v in model.free_RVs)}")

    # Sample
    print("\n" + "=" * 60)
    print(f"Sampling: {args.chains} chains × {args.draws} draws "
          f"({args.tune} tuning, target_accept={args.target_accept})")
    print("=" * 60)
    print("This will take a while with 1600+ team-seasons...")

    with model:
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=42,
            return_inferencedata=True,
            progressbar=True,
        )

    # Diagnostics
    print("\n" + "=" * 60)
    print("Diagnostics")
    print("=" * 60)
    diag = print_diagnostics(trace)

    # Save trace
    trace_path = output_dir / "trace.nc"
    az.to_netcdf(trace, str(trace_path))
    print(f"\nTrace saved to {trace_path}")

    # Save team mapping
    mapping = {
        "team_keys": [(int(s), t) for s, t in team_keys],
        "key_to_idx": {f"{s}_{t}": i for (s, t), i in key_to_idx.items()},
        "unique_years": [int(y) for y in unique_years],
        "source_cols": [c for c in pd.read_csv(Path(args.data_dir) / "ratings_matrix_standardized.csv").columns if c.startswith("source_")],
    }
    with open(output_dir / "team_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    print("\n✓ Fitting complete.")
    print(f"  Trace: {trace_path}")
    print(f"  Next: python simulate.py --year 2025 --trace {trace_path}")


if __name__ == "__main__":
    main()

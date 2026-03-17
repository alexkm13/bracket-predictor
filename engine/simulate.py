"""
Tournament Simulation Engine.

Usage:
    python simulate.py --year 2025 --trace output/trace.nc
    python simulate.py --year 2025 --trace output/trace.nc --n-sims 50000
"""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from model import build_prediction_model


# Standard first-round seed matchups within each region
# Listed in bracket order (winners play each other)
BRACKET_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]


def extract_trace_params(trace: az.InferenceData) -> dict:
    """Extract posterior samples, stacking chains into single draw dimension."""
    posterior = trace.posterior

    def stack(name):
        arr = posterior[name].values
        return arr.reshape(-1, *arr.shape[2:])

    params = {
        "mu_seed": stack("mu_seed"),
        "sigma_team": stack("sigma_team"),
        "sigma_obs": stack("sigma_obs"),
        "sigma_game": stack("sigma_game"),
    }

    if "a_nonanchor" in posterior:
        params["a_nonanchor"] = stack("a_nonanchor")
        params["b_nonanchor"] = stack("b_nonanchor")
    else:
        n_draws = params["sigma_game"].shape[0]
        params["a_nonanchor"] = np.zeros((n_draws, 0))
        params["b_nonanchor"] = np.ones((n_draws, 0))

    print(f"Extracted {params['mu_seed'].shape[0]} posterior draws")
    return params


def win_probability(theta_i: float, theta_j: float, sigma_game: float, df: int = 7) -> float:
    """
    P(team_i wins) using Student-t CDF.
    margin ~ Student-t(df, theta_i - theta_j, sigma_game)
    P(margin > 0) = 1 - CDF(0)
    """
    return 1.0 - student_t.cdf(0, df=df, loc=theta_i - theta_j, scale=sigma_game)


def simulate_bracket(thetas, sigma_game, seeds, region_teams, rng):
    """
    Simulate a single 64-team bracket.

    Parameters
    ----------
    thetas : (N,) team strengths
    sigma_game : scalar
    seeds : (N,) seeds
    region_teams : dict {region_name: list of team indices, ordered by seed}
    rng : numpy Generator

    Returns
    -------
    dict with round-by-round winners and champion
    """
    round_winners = {r: [] for r in range(7)}  # 0=R64, 1=R32, 2=S16, 3=E8, 4=F4, 5=CG

    # --- Regional rounds (R64 through E8) ---
    regional_winners = []
    regions = sorted(region_teams.keys())

    for region in regions:
        team_idx_list = region_teams[region]

        # Build seed-to-index mapping for this region
        # Resolve duplicate seeds (play-in games)
        # If multiple teams share a seed, keep the one with higher theta
        seed_counts = {}
        for tidx in team_idx_list:
            s = int(seeds[tidx])
            if s not in seed_counts:
                seed_counts[s] = []
            seed_counts[s].append(tidx)
        
        seed_to_team = {}
        for s, team_list in seed_counts.items():
            if len(team_list) == 1:
                seed_to_team[s] = team_list[0]
            else:
                # Simulate play-in: higher theta wins
                best = max(team_list, key=lambda t: thetas[t])
                # Actually simulate it probabilistically
                if len(team_list) == 2:
                    i, j = team_list
                    p_i = win_probability(thetas[i], thetas[j], sigma_game)
                    seed_to_team[s] = i if rng.random() < p_i else j
                else:
                    seed_to_team[s] = best
        

        # First round: use BRACKET_MATCHUPS for proper seed pairing
        current_round = []
        for seed_a, seed_b in BRACKET_MATCHUPS:
            # Handle play-in games: if a seed has multiple teams, just take first
            if seed_a not in seed_to_team or seed_b not in seed_to_team:
                # Skip this matchup if teams missing
                continue

            i = seed_to_team[seed_a]
            j = seed_to_team[seed_b]

            p_i = win_probability(thetas[i], thetas[j], sigma_game)
            winner = i if rng.random() < p_i else j
            current_round.append(winner)
            round_winners[0].append(winner)

        # Subsequent rounds: winners play adjacent winners
        round_num = 1
        while len(current_round) > 1:
            next_round = []
            for g in range(0, len(current_round), 2):
                if g + 1 >= len(current_round):
                    next_round.append(current_round[g])
                    continue
                i, j = current_round[g], current_round[g + 1]
                p_i = win_probability(thetas[i], thetas[j], sigma_game)
                winner = i if rng.random() < p_i else j
                next_round.append(winner)
                round_winners[round_num].append(winner)
            current_round = next_round
            round_num += 1

        regional_winners.append(current_round[0])

    # --- Final Four ---
    # Convention: region 0 vs 1, region 2 vs 3
    ff_winners = []
    for g in range(0, len(regional_winners), 2):
        if g + 1 >= len(regional_winners):
            ff_winners.append(regional_winners[g])
            continue
        i, j = regional_winners[g], regional_winners[g + 1]
        p_i = win_probability(thetas[i], thetas[j], sigma_game)
        winner = i if rng.random() < p_i else j
        ff_winners.append(winner)
        round_winners[4].append(winner)

    # --- Championship ---
    if len(ff_winners) >= 2:
        i, j = ff_winners[0], ff_winners[1]
        p_i = win_probability(thetas[i], thetas[j], sigma_game)
        champion = i if rng.random() < p_i else j
        round_winners[5].append(champion)
    else:
        champion = ff_winners[0] if ff_winners else -1

    return {
        "champion": champion,
        "final_four": regional_winners,
        "round_winners": round_winners,
    }


def run_simulation(theta_samples, sigma_game_samples, seeds, region_teams,
                    team_labels, n_sims=10_000, seed=42):
    """
    Run n_sims bracket simulations from posterior draws.

    Returns advancement probabilities and champion probabilities.
    """
    n_draws = theta_samples.shape[0]
    n_teams = theta_samples.shape[1]
    rng = np.random.default_rng(seed)

    # Track: how many times each team reaches each round
    # Rounds: 0=win R64, 1=win R32, 2=win S16, 3=win E8, 4=win F4, 5=win CG
    n_rounds = 6
    advancement = np.zeros((n_teams, n_rounds), dtype=np.int32)
    champion_counts = np.zeros(n_teams, dtype=np.int32)

    print(f"Running {n_sims:,} simulations...")

    for sim in range(n_sims):
        if (sim + 1) % 2000 == 0:
            print(f"  {sim + 1:,}/{n_sims:,}")

        draw_idx = rng.integers(0, n_draws)
        thetas = theta_samples[draw_idx]
        sigma_game = sigma_game_samples[draw_idx]

        result = simulate_bracket(thetas, sigma_game, seeds, region_teams, rng)

        # Track advancement
        for round_num, winners in result["round_winners"].items():
            if round_num < n_rounds:
                for w in winners:
                    advancement[w, round_num] += 1

        if result["champion"] >= 0:
            champion_counts[result["champion"]] += 1

    return {
        "advancement_probs": advancement / n_sims,
        "champion_probs": champion_counts / n_sims,
        "n_sims": n_sims,
    }


def load_year_data(year, data_dir="data/processed"):
    """
    Load tournament data for a specific year.

    Builds region assignments from the ratings matrix.
    Teams within each region are identified by grouping —
    the standardized CSV should have teams ordered by region/seed.
    """
    data_dir = Path(data_dir)
    ratings_df = pd.read_csv(data_dir / "ratings_matrix_standardized.csv")
    year_df = ratings_df[ratings_df["season"] == year].copy().reset_index(drop=True)

    if len(year_df) == 0:
        raise ValueError(f"No data for year {year}")

    source_cols = [c for c in year_df.columns if c.startswith("source_")]
    ratings = year_df[source_cols].values.astype(np.float64)
    seeds = year_df["seed"].values.astype(np.int64)
    team_labels = year_df["team"].values.tolist()

    print(f"Loaded {len(team_labels)} teams for {year}")

    # Build region assignments
    # Try to load from a bracket file first
    bracket_path = data_dir / f"bracket_{year}.json"
    if bracket_path.exists():
        with open(bracket_path) as f:
            bracket_data = json.load(f)
        region_teams = {}
        for region, teams in bracket_data["regions"].items():
            region_teams[region] = [team_labels.index(t) for t in teams if t in team_labels]
        print(f"Loaded bracket from {bracket_path}")
    else:
        # Default: assign teams to 4 balanced regions
        # Each region gets one team per seed (1-16)
        # Without a bracket file, we randomly distribute same-seeded teams
        region_names = ["East", "West", "South", "Midwest"]
        region_teams = {r: [] for r in region_names}
        
        # Group teams by seed
        seed_groups = {}
        for idx, s in enumerate(seeds):
            seed_groups.setdefault(int(s), []).append(idx)
        
        # For each seed, distribute teams across regions
        assign_rng = np.random.default_rng(year)  # deterministic per year
        for s in sorted(seed_groups.keys()):
            teams_with_seed = seed_groups[s]
            assign_rng.shuffle(teams_with_seed)
            for r_idx, tidx in enumerate(teams_with_seed[:4]):
                region_teams[region_names[r_idx]].append(tidx)
        
        print(f"WARNING: Using synthetic region assignments (no bracket_{year}.json found)")
        print(f"  Each region has seeds: ", end="")
        for rname in region_names:
            s_list = sorted([int(seeds[t]) for t in region_teams[rname]])
            print(f"{rname}={s_list[:4]}... ", end="")
        print()
    return ratings, seeds, team_labels, region_teams


def format_results(sim_results, team_labels, seeds, region_teams):
    """Format results as a DataFrame sorted by championship probability."""
    round_names = ["R64", "R32", "S16", "E8", "F4", "Champ"]
    n_rounds = sim_results["advancement_probs"].shape[1]

    # Build team-to-region lookup
    team_region = {}
    for reg, teams in region_teams.items():
        for t in teams:
            team_region[t] = reg

    rows = []
    for i, label in enumerate(team_labels):
        row = {
            "team": label,
            "seed": int(seeds[i]),
            "region": team_region.get(i, "?"),
        }
        for r in range(min(n_rounds, len(round_names))):
            row[round_names[r]] = sim_results["advancement_probs"][i, r]
        row["champ_pct"] = sim_results["champion_probs"][i]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("champ_pct", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Simulate NCAA tournament")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--trace", type=str, default="output/trace.nc")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trace
    print("Loading posterior trace...")
    trace = az.from_netcdf(args.trace)
    trace_params = extract_trace_params(trace)

    # Load year data
    print(f"\nLoading {args.year} tournament data...")
    ratings, seeds, team_labels, region_teams = load_year_data(args.year, args.data_dir)

    # Generate posterior theta for new teams
    print("\nGenerating posterior team strengths...")
    pred = build_prediction_model(
        new_ratings=ratings, new_seeds=seeds, trace_params=trace_params,
    )

    # Print top teams
    theta_mean = pred["theta"].mean(axis=0)
    theta_std = pred["theta"].std(axis=0)
    sorted_idx = np.argsort(-theta_mean)
    print(f"\nTop 15 teams by posterior mean θ:")
    for rank, idx in enumerate(sorted_idx[:15]):
        print(f"  {rank+1:>2}. [{seeds[idx]:>2}] {team_labels[idx]:<25s} "
              f"θ = {theta_mean[idx]:+.3f} ± {theta_std[idx]:.3f}")

    # Simulate
    print()
    sim_results = run_simulation(
        theta_samples=pred["theta"],
        sigma_game_samples=pred["sigma_game"],
        seeds=seeds,
        region_teams=region_teams,
        team_labels=team_labels,
        n_sims=args.n_sims,
    )

    # Format and display
    results_df = format_results(sim_results, team_labels, seeds, region_teams)

    print("\n" + "=" * 80)
    print(f"Tournament Simulation Results ({args.n_sims:,} simulations)")
    print("=" * 80)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(results_df.head(25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Save
    results_df.to_csv(output_dir / f"simulation_{args.year}.csv", index=False)

    # Save summary JSON
    summary = {
        "year": args.year,
        "n_sims": args.n_sims,
        "champion_favorites": [
            {"team": team_labels[sorted_idx[i]], "seed": int(seeds[sorted_idx[i]]),
             "champ_pct": float(sim_results["champion_probs"][sorted_idx[i]])}
            for i in range(min(10, len(sorted_idx)))
        ],
    }
    with open(output_dir / f"bracket_summary_{args.year}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print("✓ Simulation complete.")


if __name__ == "__main__":
    main()

"""
Export frontend-ready bracket data for the React UI.

Usage:
    python export_frontend_data.py --year 2026 --trace output/trace.nc --data-dir ../etl/data/processed
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np
from scipy.stats import t as student_t

from model import build_prediction_model
from simulate import BRACKET_MATCHUPS, extract_trace_params, load_year_data


def pairwise_win_matrix(theta_samples, sigma_game_samples, max_draws=1200, draw_seed=42):
    """
    Compute P(team_i beats team_j) matrix averaged over posterior draws.
    """
    n_draws = theta_samples.shape[0]
    rng = np.random.default_rng(draw_seed)

    if n_draws > max_draws:
        use_idx = rng.choice(n_draws, size=max_draws, replace=False)
        theta = theta_samples[use_idx]
        sigma = sigma_game_samples[use_idx]
    else:
        theta = theta_samples
        sigma = sigma_game_samples

    # delta[d, i, j] = theta[d, i] - theta[d, j]
    delta = theta[:, :, None] - theta[:, None, :]
    probs = 1.0 - student_t.cdf(0, df=7, loc=delta, scale=sigma[:, None, None])
    matrix = probs.mean(axis=0)

    # Numerically enforce anti-symmetry and diagonal baseline.
    matrix = 0.5 * (matrix + (1.0 - matrix.T))
    np.fill_diagonal(matrix, 0.5)
    return matrix


def build_region_seed_slots(region_teams, seeds):
    """
    Convert region->team indices into region->seed->team indices.
    Preserves order from bracket source to keep play-in ordering stable.
    """
    region_seed_slots = {}
    for region, team_indices in region_teams.items():
        seed_slots = {str(s): [] for s in range(1, 17)}
        for idx in team_indices:
            seed_slots[str(int(seeds[idx]))].append(int(idx))
        region_seed_slots[region] = seed_slots
    return region_seed_slots


def build_final_four_pairs(region_names):
    """
    Use canonical NCAA layout when standard region names are present.
    """
    expected = {"East", "West", "South", "Midwest"}
    if expected.issubset(set(region_names)):
        return [["East", "West"], ["South", "Midwest"]]

    pairs = []
    for i in range(0, len(region_names), 2):
        if i + 1 < len(region_names):
            pairs.append([region_names[i], region_names[i + 1]])
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Export frontend bracket data")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--trace", type=str, default="output/trace.nc")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="../frontend/public/data/bracket_data.json")
    parser.add_argument("--max-draws", type=int, default=1200)
    parser.add_argument("--draw-seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading posterior trace...")
    trace = az.from_netcdf(args.trace)
    trace_params = extract_trace_params(trace)

    print(f"Loading {args.year} tournament data...")
    ratings, seeds, team_labels, region_teams = load_year_data(args.year, args.data_dir)

    print("Generating posterior team strengths...")
    pred = build_prediction_model(
        new_ratings=ratings,
        new_seeds=seeds,
        trace_params=trace_params,
    )
    theta_samples = pred["theta"]
    sigma_game_samples = pred["sigma_game"]

    print("Computing pairwise win probability matrix...")
    win_matrix = pairwise_win_matrix(
        theta_samples=theta_samples,
        sigma_game_samples=sigma_game_samples,
        max_draws=args.max_draws,
        draw_seed=args.draw_seed,
    )

    region_seed_slots = build_region_seed_slots(region_teams, seeds)
    region_names = list(region_teams.keys())
    final_four_pairs = build_final_four_pairs(region_names)

    team_region = {}
    for region, tlist in region_teams.items():
        for tid in tlist:
            team_region[int(tid)] = region

    payload = {
        "year": args.year,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matchups": {
            "round_of_64_seed_pairs": [[a, b] for a, b in BRACKET_MATCHUPS],
            "final_four_pairs": final_four_pairs,
        },
        "teams": [
            {
                "id": int(i),
                "name": team_labels[i],
                "seed": int(seeds[i]),
                "region": team_region.get(int(i), None),
            }
            for i in range(len(team_labels))
        ],
        "regions": {
            region: {"seed_slots": slots}
            for region, slots in region_seed_slots.items()
        },
        "win_prob_matrix": np.round(win_matrix, 6).tolist(),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved frontend data to {output_path}")


if __name__ == "__main__":
    main()

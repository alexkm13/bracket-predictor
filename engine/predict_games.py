"""
Game-by-game predictions for a specific tournament year.

Usage:
    python predict_games.py --year 2026 --trace output/trace.nc --data-dir ../etl/data/processed
"""

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from model import build_prediction_model
from simulate import extract_trace_params, win_probability, BRACKET_MATCHUPS


def predict_round(matchups, thetas, sigma_game, team_labels, seeds):
    """
    Given a list of (idx_i, idx_j) matchups, compute win probs and spreads.
    Returns list of dicts with predictions.
    """
    results = []
    for idx_i, idx_j in matchups:
        # Compute across posterior draws
        n_draws = thetas.shape[0]
        use = min(n_draws, 2000)
        draw_idx = np.random.choice(n_draws, use, replace=False)

        probs = np.array([
            win_probability(thetas[d, idx_i], thetas[d, idx_j], sigma_game[d])
            for d in draw_idx
        ])
        spreads = thetas[draw_idx, idx_i] - thetas[draw_idx, idx_j]

        p_i = float(probs.mean())
        spread_mean = float(spreads.mean())
        spread_std = float(spreads.std())

        fav_idx = idx_i if p_i >= 0.5 else idx_j
        dog_idx = idx_j if p_i >= 0.5 else idx_i
        fav_prob = p_i if p_i >= 0.5 else 1 - p_i

        results.append({
            "favorite": team_labels[fav_idx],
            "fav_seed": int(seeds[fav_idx]),
            "underdog": team_labels[dog_idx],
            "dog_seed": int(seeds[dog_idx]),
            "fav_win_pct": fav_prob,
            "spread": spread_mean,  # positive = team_i favored
            "spread_std": spread_std,
            "team_i": team_labels[idx_i],
            "team_j": team_labels[idx_j],
            "p_i": p_i,
        })
    return results


def build_r64_matchups(region_teams, seeds, thetas, rng):
    """
    Build R64 matchups from bracket structure.
    Resolves play-in duplicates probabilistically.
    """
    matchups_by_region = {}

    for region, team_idx_list in region_teams.items():
        # Resolve duplicate seeds (play-in)
        seed_counts = {}
        for tidx in team_idx_list:
            s = int(seeds[tidx])
            seed_counts.setdefault(s, []).append(tidx)

        seed_to_team = {}
        playin_games = []
        for s, team_list in seed_counts.items():
            if len(team_list) == 1:
                seed_to_team[s] = team_list[0]
            else:
                # Record play-in game, pick winner by mean theta
                i, j = team_list[0], team_list[1]
                playin_games.append((i, j))
                # Use mean theta for deterministic play-in resolution
                seed_to_team[s] = i if thetas[:, i].mean() > thetas[:, j].mean() else j

        region_matchups = []
        for seed_a, seed_b in BRACKET_MATCHUPS:
            if seed_a in seed_to_team and seed_b in seed_to_team:
                region_matchups.append((seed_to_team[seed_a], seed_to_team[seed_b]))

        matchups_by_region[region] = {
            "r64": region_matchups,
            "playin": playin_games,
        }

    return matchups_by_region


def main():
    parser = argparse.ArgumentParser(description="Game-by-game predictions")
    parser.add_argument("--year", type=int, required=True)
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
    from simulate import load_year_data
    ratings, seeds, team_labels, region_teams = load_year_data(args.year, args.data_dir)

    # Generate posterior thetas
    print("Generating posterior team strengths...")
    pred = build_prediction_model(
        new_ratings=ratings, new_seeds=seeds, trace_params=trace_params,
    )
    thetas = pred["theta"]       # (n_draws, n_teams)
    sigma_game = pred["sigma_game"]  # (n_draws,)

    rng = np.random.default_rng(42)

    # Build matchups
    matchups_by_region = build_r64_matchups(region_teams, seeds, thetas, rng)

    # === PLAY-IN GAMES ===
    has_playins = any(len(m["playin"]) > 0 for m in matchups_by_region.values())
    if has_playins:
        print("\n" + "=" * 70)
        print("PLAY-IN GAMES")
        print("=" * 70)
        for region in sorted(matchups_by_region.keys()):
            playins = matchups_by_region[region]["playin"]
            if playins:
                preds = predict_round(playins, thetas, sigma_game, team_labels, seeds)
                for g in preds:
                    print(f"  [{g['fav_seed']:>2}] {g['favorite']:<22s} vs [{g['dog_seed']:>2}] {g['underdog']:<22s}"
                          f"  →  {g['favorite']} {g['fav_win_pct']*100:5.1f}%  (spread {g['spread']:+.1f})")

    # === ROUND OF 64 ===
    print("\n" + "=" * 70)
    print("ROUND OF 64")
    print("=" * 70)

    all_r64_preds = []
    for region in sorted(matchups_by_region.keys()):
        print(f"\n  --- {region} Region ---")
        r64 = matchups_by_region[region]["r64"]
        preds = predict_round(r64, thetas, sigma_game, team_labels, seeds)
        for g in preds:
            print(f"  [{g['fav_seed']:>2}] {g['favorite']:<22s} vs [{g['dog_seed']:>2}] {g['underdog']:<22s}"
                  f"  →  {g['favorite']} {g['fav_win_pct']*100:5.1f}%  (spread {g['spread']:+.1f})")
            all_r64_preds.append({**g, "region": region, "round": "R64"})

    # Summary stats
    print("\n" + "=" * 70)
    print("UPSET WATCH (favorites < 75% win probability)")
    print("=" * 70)
    upsets = [g for g in all_r64_preds if g["fav_win_pct"] < 0.75]
    upsets.sort(key=lambda x: x["fav_win_pct"])
    for g in upsets:
        print(f"  [{g['fav_seed']:>2}] {g['favorite']:<22s} vs [{g['dog_seed']:>2}] {g['underdog']:<22s}"
              f"  →  {g['fav_win_pct']*100:5.1f}%  ({g['region']})")

    # Save
    pd.DataFrame(all_r64_preds).to_csv(output_dir / f"r64_predictions_{args.year}.csv", index=False)
    print(f"\nSaved to {output_dir}/r64_predictions_{args.year}.csv")

def predict_full_bracket(args):
    """Predict every round by advancing favorites."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading posterior trace...")
    trace = az.from_netcdf(args.trace)
    trace_params = extract_trace_params(trace)

    from simulate import load_year_data
    ratings, seeds, team_labels, region_teams = load_year_data(args.year, args.data_dir)

    print("Generating posterior team strengths...")
    pred = build_prediction_model(
        new_ratings=ratings, new_seeds=seeds, trace_params=trace_params,
    )
    thetas = pred["theta"]
    sigma_game = pred["sigma_game"]
    rng = np.random.default_rng(42)

    matchups_by_region = build_r64_matchups(region_teams, seeds, thetas, rng)

    round_names = ["PLAY-IN", "ROUND OF 64", "ROUND OF 32", "SWEET 16", "ELITE 8", "FINAL FOUR", "CHAMPIONSHIP"]
    all_preds = []

    # === PLAY-IN ===
    has_playins = any(len(m["playin"]) > 0 for m in matchups_by_region.values())
    if has_playins:
        print(f"\n{'='*70}")
        print("PLAY-IN GAMES")
        print(f"{'='*70}")
        for region in sorted(matchups_by_region.keys()):
            playins = matchups_by_region[region]["playin"]
            if playins:
                preds = predict_round(playins, thetas, sigma_game, team_labels, seeds)
                for g in preds:
                    print(f"  [{g['fav_seed']:>2}] {g['favorite']:<22s} vs [{g['dog_seed']:>2}] {g['underdog']:<22s}"
                          f"  →  {g['favorite']} {g['fav_win_pct']*100:5.1f}%  (spread {g['spread']:+.1f})")
                    all_preds.append({**g, "region": region, "round": "Play-In"})

    # === REGIONAL ROUNDS ===
    regional_champions = {}
    region_order = sorted(matchups_by_region.keys())

    for region in region_order:
        r64_matchups = matchups_by_region[region]["r64"]
        current_matchups = r64_matchups
        
        for round_idx, round_name in enumerate(["ROUND OF 64", "ROUND OF 32", "SWEET 16", "ELITE 8"]):
            if round_idx == 0:
                print(f"\n{'='*70}")
                print(f"{round_name}")
                print(f"{'='*70}")
            
            if round_idx > 0 and round_idx <= 1:
                print(f"\n{'='*70}")
                print(f"{round_name}")
                print(f"{'='*70}")
            elif round_idx > 1:
                print(f"\n{'='*70}")
                print(f"{round_name}")
                print(f"{'='*70}")

            print(f"\n  --- {region} Region ---")
            preds = predict_round(current_matchups, thetas, sigma_game, team_labels, seeds)

            winners = []
            for g in preds:
                # Advance the favorite
                fav_idx = team_labels.index(g["favorite"])
                winners.append(fav_idx)
                print(f"  [{g['fav_seed']:>2}] {g['favorite']:<22s} vs [{g['dog_seed']:>2}] {g['underdog']:<22s}"
                      f"  →  {g['favorite']} {g['fav_win_pct']*100:5.1f}%  (spread {g['spread']:+.1f})")
                all_preds.append({**g, "region": region, "round": round_name.title()})

            # Build next round matchups: adjacent winners play each other
            current_matchups = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    current_matchups.append((winners[i], winners[i + 1]))

            if len(winners) == 1:
                regional_champions[region] = winners[0]
                print(f"\n  ★ {region} Champion: [{int(seeds[winners[0]]):>2}] {team_labels[winners[0]]}")

    # === FINAL FOUR ===
    print(f"\n{'='*70}")
    print("FINAL FOUR")
    print(f"{'='*70}")

    ff_regions = region_order
    # Convention: region 0 vs 1, region 2 vs 3
    ff_matchups = [
        (regional_champions[ff_regions[0]], regional_champions[ff_regions[1]]),
        (regional_champions[ff_regions[2]], regional_champions[ff_regions[3]]),
    ]
    preds = predict_round(ff_matchups, thetas, sigma_game, team_labels, seeds)
    ff_winners = []
    for g in preds:
        fav_idx = team_labels.index(g["favorite"])
        ff_winners.append(fav_idx)
        print(f"  [{g['fav_seed']:>2}] {g['favorite']:<22s} vs [{g['dog_seed']:>2}] {g['underdog']:<22s}"
              f"  →  {g['favorite']} {g['fav_win_pct']*100:5.1f}%  (spread {g['spread']:+.1f})")
        all_preds.append({**g, "region": "Final Four", "round": "Final Four"})

    # === CHAMPIONSHIP ===
    print(f"\n{'='*70}")
    print("CHAMPIONSHIP")
    print(f"{'='*70}")

    champ_matchup = [(ff_winners[0], ff_winners[1])]
    preds = predict_round(champ_matchup, thetas, sigma_game, team_labels, seeds)
    for g in preds:
        print(f"  [{g['fav_seed']:>2}] {g['favorite']:<22s} vs [{g['dog_seed']:>2}] {g['underdog']:<22s}"
              f"  →  {g['favorite']} {g['fav_win_pct']*100:5.1f}%  (spread {g['spread']:+.1f})")
        all_preds.append({**g, "region": "Championship", "round": "Championship"})
        print(f"\n  ★★★ PREDICTED CHAMPION: [{g['fav_seed']:>2}] {g['favorite']} ★★★")

    # Save all predictions
    pd.DataFrame(all_preds).to_csv(output_dir / f"full_bracket_{args.year}.csv", index=False)
    print(f"\nSaved to {output_dir}/full_bracket_{args.year}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game-by-game predictions")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--trace", type=str, default="output/trace.nc")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--full", action="store_true", help="Predict full bracket (all rounds)")
    args = parser.parse_args()

    if args.full:
        predict_full_bracket(args)
    else:
        main()

"""
KenPom Data Extraction

Processes the pre-tournament KenPom summary CSV into the format
needed by the model: one row per tournament team per year with
AdjOE, AdjDE, AdjEM, and seed.

Input:  INT___KenPom___Summary__Pre-Tournament_.csv
Output: Cleaned DataFrame with columns:
        [season, team, seed, kenpom_adj_oe, kenpom_adj_de, kenpom_adj_em]
"""

import pandas as pd
import numpy as np
from pathlib import Path


def extract_kenpom(filepath: str) -> pd.DataFrame:
    """
    Extract pre-tournament KenPom ratings for tournament teams.
    
    Parameters:
        filepath: path to INT___KenPom___Summary__Pre-Tournament_.csv
    
    Returns:
        DataFrame with tournament teams only, cleaned and typed
    """
    df = pd.read_csv(filepath)
    
    # Rename columns to our convention
    df = df.rename(columns={
        "Season": "season",
        "TeamName": "team",
        "AdjOE": "kenpom_adj_oe",
        "AdjDE": "kenpom_adj_de",
        "AdjEM": "kenpom_adj_em",
        "Seed": "seed",
    })
    
    # Keep only tournament teams (those with a seed)
    df = df[df["seed"].notna()].copy()
    df["seed"] = df["seed"].astype(int)
    
    # Keep only the columns we need
    df = df[["season", "team", "seed", "kenpom_adj_oe", "kenpom_adj_de", "kenpom_adj_em"]]
    
    # Convert to numeric (handle any parsing issues)
    for col in ["kenpom_adj_oe", "kenpom_adj_de", "kenpom_adj_em"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing AdjEM (shouldn't happen but be safe)
    df = df.dropna(subset=["kenpom_adj_em"])
    
    # Sort for readability
    df = df.sort_values(["season", "seed", "team"]).reset_index(drop=True)
    
    return df


def summarize_kenpom(df: pd.DataFrame) -> None:
    """Print summary statistics for validation."""
    print(f"Total team-seasons: {len(df)}")
    print(f"Seasons: {df['season'].min()} - {df['season'].max()}")
    print(f"Teams per season:")
    print(df.groupby("season").size().to_string())
    print(f"\nAdjEM summary:")
    print(df["kenpom_adj_em"].describe())
    print(f"\nAdjEM by seed (all years):")
    seed_means = df.groupby("seed")["kenpom_adj_em"].agg(["mean", "std", "count"])
    print(seed_means.to_string())


if __name__ == "__main__":
    filepath = "data/raw/kenpom/INT _ KenPom _ Summary (Pre-Tournament).csv"
    df = extract_kenpom(filepath)
    summarize_kenpom(df)
    
    # Save processed output
    output_path = "data/processed/kenpom_ratings.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

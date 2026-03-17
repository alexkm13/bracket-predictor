"""
Tournament Game Results Extraction

Processes the Big Dance CSV into the format needed by the model's
game outcome layer (Level 4). Each row is one tournament game with
two teams, their seeds, scores, and the margin.

This data is used for:
    1. Fitting σ_game (how noisy single games are)
    2. Backtesting calibration (do our win probabilities match reality?)
    3. Log score evaluation (how good are our predictions?)

Input:  Big_Dance_CSV.csv
        Format: Year, Round, Region Number, Region Name, Seed, Score, Team, Team, Score, Seed

Output: Cleaned DataFrame with columns:
        [season, round, team_i, seed_i, score_i, team_j, seed_j, score_j, margin]
        where margin = score_i - score_j (positive means team_i won)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Big Dance CSV uses its own naming convention
# These map to canonical tournament reference names
BIGDANCE_TO_CANONICAL = {
    "CONN": "UConn",
    "Connecticut": "UConn",
    "UConn": "UConn",
    "Boise St": "Boise State",
    "Colorado St": "Colorado State",
    "Iowa St": "Iowa State",
    "Michigan St": "Michigan State",
    "Mississippi St": "Mississippi State",
    "Montana St": "Montana State",
    "Utah St": "Utah State",
    "Washington St": "Washington State",
    "San Diego St": "San Diego State",
    "South Dakota St": "South Dakota State",
    "Wichita St": "Wichita State",
    "Wright St": "Wright State",
    "Murray St": "Murray State",
    "Ohio St": "Ohio State",
    "Kansas St": "Kansas State",
    "Penn St": "Penn State",
    "Oklahoma St": "Oklahoma State",
    "Oregon St": "Oregon State",
    "Arizona St": "Arizona State",
    "Fresno St": "Fresno State",
    "Norfolk St": "Norfolk State",
    "Morehead St": "Morehead State",
    "Jacksonville St": "Jacksonville State",
    "Kennesaw St": "Kennesaw State",
    "Weber St": "Weber State",
    "Illinois St": "Illinois State",
    "Indiana St": "Indiana State",
    "Missouri St": "Missouri State",
    "New Mexico St": "New Mexico State",
    "Long Beach St": "Long Beach State",
    "N Carolina St": "NC State",
    "NC State": "NC State",
    "North Carolina St": "NC State",
    "N.C. State": "NC State",
    "Ole Miss": "Ole Miss",
    "Miss": "Ole Miss",
    "Mississippi": "Ole Miss",
    "St Johns": "St. John's",
    "St. Johns": "St. John's",
    "St John's": "St. John's",
    "St Marys": "Saint Mary's",
    "Saint Marys": "Saint Mary's",
    "St. Mary's": "Saint Mary's",
    "Saint Mary's": "Saint Mary's",
    "St Peters": "Saint Peter's",
    "Saint Peters": "Saint Peter's",
    "St. Peter's": "Saint Peter's",
    "Saint Peter's": "Saint Peter's",
    "Miami FL": "Miami",
    "Miami (FL)": "Miami",
    "Miami OH": "Miami (OH)",
    "Miami (OH)": "Miami (OH)",
    "USC": "Southern California",
    "So California": "Southern California",
    "Southern Cal": "Southern California",
    "VCU": "VCU",
    "UCF": "UCF",
    "SMU": "SMU",
    "BYU": "BYU",
    "UNLV": "UNLV",
    "LSU": "LSU",
    "UAB": "UAB",
    "UTEP": "UTEP",
    "LIU": "LIU",
    "LIU Brooklyn": "LIU",
    "FDU": "FDU",
    "Fairleigh Dickinson": "FDU",
    "UMBC": "UMBC",
    "UNC Greensboro": "UNC Greensboro",
    "UNC Asheville": "UNC Asheville",
    "UNC Wilmington": "UNC Wilmington",
    "ETSU": "East Tennessee State",
    "E Tennessee St": "East Tennessee State",
    "East Tennessee St": "East Tennessee State",
    "Loyola Chicago": "Loyola Chicago",
    "Loyola-Chicago": "Loyola Chicago",
    "Loyola IL": "Loyola Chicago",
    "Detroit": "Detroit Mercy",
    "Detroit Mercy": "Detroit Mercy",
    "Cal St Fullerton": "Cal State Fullerton",
    "CS Fullerton": "Cal State Fullerton",
    "Cal St Bakersfield": "Cal State Bakersfield",
    "CS Bakersfield": "Cal State Bakersfield",
    "Texas A&M CC": "Texas A&M-CC",
    "Texas A&M-Corpus Christi": "Texas A&M-CC",
    "TX A&M-CC": "Texas A&M-CC",
    "Mount St Marys": "Mount St. Mary's",
    "Mt St Marys": "Mount St. Mary's",
    "Mount St. Mary's": "Mount St. Mary's",
    "Ark Little Rock": "Little Rock",
    "Arkansas-Little Rock": "Little Rock",
    "Little Rock": "Little Rock",
    "Sam Houston St": "Sam Houston",
    "Sam Houston": "Sam Houston",
    "App State": "Appalachian State",
    "Appalachian St": "Appalachian State",
    "Florida St": "Florida State",
    "N Dakota St": "North Dakota State",
    "North Dakota St": "North Dakota State",
    "S Dakota St": "South Dakota State",
    "South Dakota St": "South Dakota State",
    "SE Missouri St": "Southeast Missouri State",
    "Kent St": "Kent State",
    "Kent State": "Kent State",
    "Grambling St": "Grambling",
    "Grambling": "Grambling",
    "McNeese St": "McNeese",
    "McNeese": "McNeese",
    "Northwestern St": "Northwestern State",
    "NW State": "Northwestern State",
    "Ga Southern": "Georgia Southern",
    "Georgia Southern": "Georgia Southern",
    "Georgia St": "Georgia State",
    "Georgia State": "Georgia State",
    "Albany": "UAlbany",
    "UAlbany": "UAlbany",
    "Morgan St": "Morgan State",
    "Miss Valley St": "Mississippi Valley State",
    "Mississippi Valley St": "Mississippi Valley State",
    "Prairie View": "Prairie View A&M",
    "S Carolina St": "South Carolina State",
    "South Carolina St": "South Carolina State",
    "Jackson St": "Jackson State",
    "Alcorn St": "Alcorn State",
    "Coppin St": "Coppin State",
    "Delaware St": "Delaware State",
    "Alabama St": "Alabama State",
    "Tennessee St": "Tennessee State",
    "UIC": "UIC",
    "Ill Chicago": "UIC",
    "Penn": "Pennsylvania",
    "Pennsylvania": "Pennsylvania",
    "Gardner Webb": "Gardner-Webb",
    "Gardner-Webb": "Gardner-Webb",
    "Ark Pine Bluff": "Arkansas-Pine Bluff",
    "Arkansas-Pine Bluff": "Arkansas-Pine Bluff",
    "SC Upstate": "South Carolina Upstate",
    "South Carolina Upstate": "South Carolina Upstate",
    "Southern Miss": "Southern Miss",
    "So Miss": "Southern Miss",
    "N Kentucky": "Northern Kentucky",
    "Northern Kentucky": "Northern Kentucky",
    "Charleston": "Charleston",
    "Col of Charleston": "Charleston",
    "College of Charleston": "Charleston",
    "Geo Washington": "George Washington",
    "George Washington": "George Washington",
    "Geo Mason": "George Mason",
    "George Mason": "George Mason",
    "St Josephs": "Saint Joseph's",
    "Saint Josephs": "Saint Joseph's",
    "St. Joseph's": "Saint Joseph's",
    "SIU Edwardsville": "SIU Edwardsville",
    "SIU-Edwardsville": "SIU Edwardsville",
}


ROUND_MAP = {
    1: "R64",
    2: "R32",
    3: "S16",
    4: "E8",
    5: "F4",
    6: "CG",
}


def extract_game_results(filepath: str, min_season: int = 2008) -> pd.DataFrame:
    """
    Extract tournament game results from the Big Dance CSV.
    
    Parameters:
        filepath: path to Big_Dance_CSV.csv
        min_season: earliest season to include (default 2008 to align with other sources)
    
    Returns DataFrame with columns:
        [season, round, team_i, seed_i, score_i, team_j, seed_j, score_j, margin]
    """
    df = pd.read_csv(filepath, encoding="latin-1")
    
    # Assign clean column names (the CSV has duplicate "Team" and "Score" columns)
    df.columns = [
        "season", "round_num", "region_num", "region",
        "seed_i", "score_i", "team_i",
        "team_j", "score_j", "seed_j",
    ]
    
    # Clean whitespace from string columns
    for col in ["team_i", "team_j", "region"]:
        df[col] = df[col].astype(str).str.strip()
    
    # Convert types
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["score_i"] = pd.to_numeric(df["score_i"], errors="coerce").astype(int)
    df["score_j"] = pd.to_numeric(df["score_j"], errors="coerce").astype(int)
    df["seed_i"] = pd.to_numeric(df["seed_i"], errors="coerce").astype(int)
    df["seed_j"] = pd.to_numeric(df["seed_j"], errors="coerce").astype(int)
    
    # Compute margin (positive = team_i won)
    df["margin"] = df["score_i"] - df["score_j"]
    
    # Map round numbers to names
    df["round"] = df["round_num"].map(ROUND_MAP)
    
    # Standardize team names
    df["team_i"] = df["team_i"].replace(BIGDANCE_TO_CANONICAL)
    df["team_j"] = df["team_j"].replace(BIGDANCE_TO_CANONICAL)
    
    # Filter to desired seasons
    df = df[df["season"] >= min_season].copy()
    
    # Select output columns
    df = df[[
        "season", "round", "region",
        "team_i", "seed_i", "score_i",
        "team_j", "seed_j", "score_j",
        "margin",
    ]]
    
    return df.sort_values(["season", "round", "region"]).reset_index(drop=True)


def summarize_game_results(df: pd.DataFrame) -> None:
    """Print summary statistics for validation."""
    print(f"Total games: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Games per season: {df.groupby('season').size().unique()}")
    
    print(f"\nGames per round:")
    for rnd in ["R64", "R32", "S16", "E8", "F4", "CG"]:
        n = len(df[df["round"] == rnd])
        print(f"  {rnd}: {n}")
    
    print(f"\nMargin statistics:")
    print(f"  Mean:   {df['margin'].mean():+.1f}")
    print(f"  Std:    {df['margin'].std():.1f}")
    print(f"  Median: {df['margin'].median():+.1f}")
    print(f"  Min:    {df['margin'].min()}")
    print(f"  Max:    {df['margin'].max()}")
    
    # Upset rate by round (lower seed winning = negative margin when higher seed is team_i)
    print(f"\nHigher seed win rate by round:")
    for rnd in ["R64", "R32", "S16", "E8", "F4", "CG"]:
        games = df[df["round"] == rnd]
        # team_i is typically the higher seed (lower number)
        higher_seed_wins = (games["seed_i"] < games["seed_j"]) & (games["margin"] > 0)
        lower_seed_wins = (games["seed_i"] > games["seed_j"]) & (games["margin"] > 0)
        same_seed = games["seed_i"] == games["seed_j"]
        
        # Count games where the better seed (lower number) won
        better_seed_won = ((games["seed_i"] < games["seed_j"]) & (games["margin"] > 0)) | \
                          ((games["seed_i"] > games["seed_j"]) & (games["margin"] < 0))
        
        if len(games) > 0:
            rate = better_seed_won.sum() / len(games)
            print(f"  {rnd}: {rate:.1%} ({better_seed_won.sum()}/{len(games)})")


if __name__ == "__main__":
    filepath = "data/raw/game_results/big_dance.csv"
    
    if not Path(filepath).exists():
        # Try alternative locations
        for alt in ["../data/raw/Big_Dance_CSV.csv", "Big_Dance_CSV.csv"]:
            if Path(alt).exists():
                filepath = alt
                break
    
    if Path(filepath).exists():
        print(f"Loading from {filepath}")
        df = extract_game_results(filepath)
        summarize_game_results(df)
        
        output_path = "data/processed/tournament_games.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
    else:
        print("Big Dance CSV not found. Place it at data/raw/Big_Dance_CSV.csv")

"""
Sports Reference SRS Data Extraction

Processes scraped SRS CSV files into the format needed by the model.
SRS data comes from the scraper output (scrapers.py --source srs).

Sports Reference uses full school names, sometimes with "NCAA" marker
for tournament teams. Names are generally close to canonical but
need some standardization.

Input:  Scraped SRS CSV (srs_all.csv or per-season srs_{YYYY}.csv files)
Output: Cleaned DataFrame with columns:
        [season, team, srs]
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Sports Reference name -> canonical tournament reference name
SRS_TO_CANONICAL = {
    # Sports Reference uses full names, mostly matching canonical
    # but with some differences
    "Connecticut": "UConn",
    "Miami (FL)": "Miami",
    "Miami FL": "Miami",
    "Miami (OH)": "Miami (OH)",
    "North Carolina State": "NC State",
    "Mississippi": "Ole Miss",
    "Southern California": "Southern California",
    "Louisiana State": "LSU",
    "Brigham Young": "BYU",
    "Virginia Commonwealth": "VCU",
    "Southern Methodist": "SMU",
    "Central Florida": "UCF",
    "Nevada-Las Vegas": "UNLV",
    "Texas-El Paso": "UTEP",
    "Alabama-Birmingham": "UAB",
    "East Tennessee State": "East Tennessee State",
    "SIU Edwardsville": "SIU Edwardsville",
    "LIU": "LIU",
    "LIU Brooklyn": "LIU",
    "Long Island University": "LIU",
    "Fairleigh Dickinson": "FDU",
    "Detroit Mercy": "Detroit Mercy",
    "Detroit": "Detroit Mercy",
    "Loyola (IL)": "Loyola Chicago",
    "Loyola Chicago": "Loyola Chicago",
    "Loyola (MD)": "Loyola Maryland",
    "Saint Mary's (CA)": "Saint Mary's",
    "Saint Mary's": "Saint Mary's",
    "Saint Peter's": "Saint Peter's",
    "St. John's (NY)": "St. John's",
    "St. John's": "St. John's",
    "St. Bonaventure": "St. Bonaventure",
    "Cal State Fullerton": "Cal State Fullerton",
    "Cal State Bakersfield": "Cal State Bakersfield",
    "Texas A&M-Corpus Christi": "Texas A&M-CC",
    "UT Arlington": "UT Arlington",
    "UT San Antonio": "UT San Antonio",
    "UT Rio Grande Valley": "UTRGV",
    "South Carolina Upstate": "South Carolina Upstate",
    "UMass Lowell": "UMass Lowell",
    "UNC Greensboro": "UNC Greensboro",
    "UNC Asheville": "UNC Asheville",
    "UNC Wilmington": "UNC Wilmington",
    "Albany (NY)": "UAlbany",
    "UMBC": "UMBC",
    "Mississippi Valley State": "Mississippi Valley State",
    "Grambling": "Grambling",
    "Grambling State": "Grambling",
    "McNeese State": "McNeese",
    "McNeese": "McNeese",
    "Northwestern State": "Northwestern State",
    "Southeast Missouri State": "Southeast Missouri State",
    "Little Rock": "Little Rock",
    "Arkansas-Little Rock": "Little Rock",
    "Sam Houston": "Sam Houston",
    "Sam Houston State": "Sam Houston",
    "App State": "Appalachian State",
    "Appalachian State": "Appalachian State",
    "Florida State": "Florida State",
    "Penn": "Pennsylvania",
    "Pennsylvania": "Pennsylvania",
    "Gardner-Webb": "Gardner-Webb",
    "Morgan State": "Morgan State",
    "Arkansas-Pine Bluff": "Arkansas-Pine Bluff",
    "Prairie View": "Prairie View A&M",
    "Prairie View A&M": "Prairie View A&M",
    "South Carolina State": "South Carolina State",
    "Jackson State": "Jackson State",
    "Alcorn State": "Alcorn State",
    "Coppin State": "Coppin State",
    "Delaware State": "Delaware State",
    "Norfolk State": "Norfolk State",
    "Alabama State": "Alabama State",
    "Tennessee State": "Tennessee State",
}


def extract_srs(input_path: str) -> pd.DataFrame:
    """
    Extract SRS ratings from scraped data.
    
    Parameters:
        input_path: path to srs_all.csv or directory of srs_{YYYY}.csv files
    
    Returns DataFrame with columns:
        [season, team, srs]
    """
    path = Path(input_path)
    
    if path.is_file():
        df = pd.read_csv(path)
    elif path.is_dir():
        all_data = []
        for csv_file in sorted(path.glob("srs_*.csv")):
            if "all" in csv_file.stem:
                continue
            all_data.append(pd.read_csv(csv_file))
        if not all_data:
            raise FileNotFoundError(f"No SRS CSVs found in {path}")
        df = pd.concat(all_data, ignore_index=True)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Ensure required columns
    required = ["team", "srs", "season"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {df.columns.tolist()}")
    
    # Clean team names
    df["team"] = df["team"].astype(str).str.strip()
    df["team"] = df["team"].replace(SRS_TO_CANONICAL)
    
    # Ensure numeric
    df["srs"] = pd.to_numeric(df["srs"], errors="coerce")
    df = df.dropna(subset=["srs"])
    
    df = df[["season", "team", "srs"]]
    
    return df.sort_values(["season", "team"]).reset_index(drop=True)


def summarize_srs(df: pd.DataFrame) -> None:
    """Print summary statistics for validation."""
    print(f"Total team-seasons: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Teams per season: {df.groupby('season').size().mean():.0f} avg")
    print(f"\nSRS summary:")
    print(df["srs"].describe())


if __name__ == "__main__":
    for path in ["data/raw/srs/srs_all.csv", "data/raw/srs/"]:
        if Path(path).exists():
            print(f"Loading from {path}")
            df = extract_srs(path)
            summarize_srs(df)
            
            output_path = "data/processed/srs_ratings.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
            break
    else:
        print("No SRS data found. Run: python scrapers.py --source srs")

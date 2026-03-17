"""
TeamRankings Predictive Rating Data Extraction

Processes scraped TeamRankings CSV files into the format needed by the model.
TeamRankings data comes from the scraper output (scrapers.py --source teamrankings).

TeamRankings uses short team names that are generally close to canonical
but need some standardization.

Input:  Scraped TeamRankings CSV (tr_all.csv or per-season tr_{YYYY}.csv files)
Output: Cleaned DataFrame with columns:
        [season, team, tr_predictive]
"""

import pandas as pd
import numpy as np
from pathlib import Path


# TeamRankings name -> canonical tournament reference name
TR_TO_CANONICAL = {
    # TeamRankings generally uses short, clean names
    # These are the known exceptions
    "Connecticut": "UConn",
    "Conn": "UConn",
    "UCONN": "UConn",
    "Miami FL": "Miami",
    "Miami (FL)": "Miami",
    "Miami OH": "Miami (OH)",
    "Miami (OH)": "Miami (OH)",
    "N.C. State": "NC State",
    "NC State": "NC State",
    "North Carolina St": "NC State",
    "N Carolina St": "NC State",
    "Ole Miss": "Ole Miss",
    "Mississippi": "Ole Miss",
    "So. California": "Southern California",
    "Southern Cal": "Southern California",
    "S. California": "Southern California",
    "Brigham Young": "BYU",
    "Virginia Commonwealth": "VCU",
    "So. Methodist": "SMU",
    "Southern Methodist": "SMU",
    "Central Florida": "UCF",
    "Nev.-Las Vegas": "UNLV",
    "Nevada-Las Vegas": "UNLV",
    "TX-El Paso": "UTEP",
    "Texas-El Paso": "UTEP",
    "AL-Birmingham": "UAB",
    "Alabama-Birmingham": "UAB",
    "E. Tennessee St": "East Tennessee State",
    "East Tenn St": "East Tennessee State",
    "ETSU": "East Tennessee State",
    "SIU-Edwardsville": "SIU Edwardsville",
    "SIU Edwardsville": "SIU Edwardsville",
    "Long Island": "LIU",
    "LIU Brooklyn": "LIU",
    "LIU": "LIU",
    "Fairleigh Dickinson": "FDU",
    "Fair. Dickinson": "FDU",
    "FDU": "FDU",
    "Detroit": "Detroit Mercy",
    "Detroit Mercy": "Detroit Mercy",
    "Loyola-Chicago": "Loyola Chicago",
    "Loyola Chicago": "Loyola Chicago",
    "Loyola IL": "Loyola Chicago",
    "Loyola Maryland": "Loyola Maryland",
    "Loyola MD": "Loyola Maryland",
    "St. Mary's": "Saint Mary's",
    "Saint Mary's": "Saint Mary's",
    "St. Peter's": "Saint Peter's",
    "Saint Peter's": "Saint Peter's",
    "St. John's": "St. John's",
    "St. Bonaventure": "St. Bonaventure",
    "Cal St. Fullerton": "Cal State Fullerton",
    "CS Fullerton": "Cal State Fullerton",
    "Cal St Fullerton": "Cal State Fullerton",
    "Cal St. Bakersfield": "Cal State Bakersfield",
    "CS Bakersfield": "Cal State Bakersfield",
    "TX A&M-Corpus Christi": "Texas A&M-CC",
    "Texas A&M-CC": "Texas A&M-CC",
    "Texas A&M-Corpus Christi": "Texas A&M-CC",
    "UT-Arlington": "UT Arlington",
    "UT Arlington": "UT Arlington",
    "UT-San Antonio": "UT San Antonio",
    "UTSA": "UT San Antonio",
    "UT Rio Grande Valley": "UTRGV",
    "UTRGV": "UTRGV",
    "SC Upstate": "South Carolina Upstate",
    "S Carolina Upstate": "South Carolina Upstate",
    "UMass-Lowell": "UMass Lowell",
    "UMass Lowell": "UMass Lowell",
    "Albany": "UAlbany",
    "Albany NY": "UAlbany",
    "UAlbany": "UAlbany",
    "UMBC": "UMBC",
    "Grambling St": "Grambling",
    "Grambling State": "Grambling",
    "Grambling": "Grambling",
    "McNeese St": "McNeese",
    "McNeese State": "McNeese",
    "McNeese": "McNeese",
    "Northwestern St": "Northwestern State",
    "NW State": "Northwestern State",
    "Ark-Little Rock": "Little Rock",
    "Little Rock": "Little Rock",
    "Arkansas-Little Rock": "Little Rock",
    "Sam Houston St": "Sam Houston",
    "Sam Houston State": "Sam Houston",
    "Sam Houston": "Sam Houston",
    "Appalachian St": "Appalachian State",
    "App State": "Appalachian State",
    "Appalachian State": "Appalachian State",
    "Fla. State": "Florida State",
    "Florida St": "Florida State",
    "Florida State": "Florida State",
    "Penn": "Pennsylvania",
    "Pennsylvania": "Pennsylvania",
    "Prairie View": "Prairie View A&M",
    "Prairie View A&M": "Prairie View A&M",
    "S Carolina St": "South Carolina State",
    "South Carolina St": "South Carolina State",
    "Miss Valley St": "Mississippi Valley State",
    "Mississippi Valley St": "Mississippi Valley State",
    "N Dakota St": "North Dakota State",
    "North Dakota St": "North Dakota State",
    "S Dakota St": "South Dakota State",
    "South Dakota St": "South Dakota State",
    "SE Missouri St": "Southeast Missouri State",
    "SE Missouri State": "Southeast Missouri State",
    "Kent St": "Kent State",
    "Kent State": "Kent State",
    "Boise St": "Boise State",
    "Boise State": "Boise State",
    "Colorado St": "Colorado State",
    "Colorado State": "Colorado State",
    "Iowa St": "Iowa State",
    "Iowa State": "Iowa State",
    "Michigan St": "Michigan State",
    "Michigan State": "Michigan State",
    "Mississippi St": "Mississippi State",
    "Mississippi State": "Mississippi State",
    "Montana St": "Montana State",
    "Montana State": "Montana State",
    "Utah St": "Utah State",
    "Utah State": "Utah State",
    "Washington St": "Washington State",
    "Washington State": "Washington State",
    "San Diego St": "San Diego State",
    "San Diego State": "San Diego State",
    "Wichita St": "Wichita State",
    "Wichita State": "Wichita State",
    "Wright St": "Wright State",
    "Wright State": "Wright State",
    "Murray St": "Murray State",
    "Murray State": "Murray State",
    "Ohio St": "Ohio State",
    "Ohio State": "Ohio State",
    "Kansas St": "Kansas State",
    "Kansas State": "Kansas State",
    "Penn St": "Penn State",
    "Penn State": "Penn State",
    "Oklahoma St": "Oklahoma State",
    "Oklahoma State": "Oklahoma State",
    "Oregon St": "Oregon State",
    "Oregon State": "Oregon State",
    "Arizona St": "Arizona State",
    "Arizona State": "Arizona State",
    "Fresno St": "Fresno State",
    "Fresno State": "Fresno State",
    "Norfolk St": "Norfolk State",
    "Norfolk State": "Norfolk State",
    "Morehead St": "Morehead State",
    "Morehead State": "Morehead State",
    "Jacksonville St": "Jacksonville State",
    "Jacksonville State": "Jacksonville State",
    "Kennesaw St": "Kennesaw State",
    "Kennesaw State": "Kennesaw State",
    "Weber St": "Weber State",
    "Weber State": "Weber State",
    "Illinois St": "Illinois State",
    "Illinois State": "Illinois State",
    "Indiana St": "Indiana State",
    "Indiana State": "Indiana State",
    "Missouri St": "Missouri State",
    "Missouri State": "Missouri State",
    "New Mexico St": "New Mexico State",
    "New Mexico State": "New Mexico State",
    "Long Beach St": "Long Beach State",
    "Long Beach State": "Long Beach State",
    "Texas St": "Texas State",
    "Texas State": "Texas State",
}


def extract_teamrankings(input_path: str) -> pd.DataFrame:
    """
    Extract TeamRankings predictive ratings from scraped data.
    
    Parameters:
        input_path: path to tr_all.csv or directory of tr_{YYYY}.csv files
    
    Returns DataFrame with columns:
        [season, team, tr_predictive]
    """
    path = Path(input_path)
    
    if path.is_file():
        df = pd.read_csv(path)
    elif path.is_dir():
        all_data = []
        for csv_file in sorted(path.glob("tr_*.csv")):
            if "all" in csv_file.stem:
                continue
            all_data.append(pd.read_csv(csv_file))
        if not all_data:
            raise FileNotFoundError(f"No TeamRankings CSVs found in {path}")
        df = pd.concat(all_data, ignore_index=True)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Ensure required columns
    required = ["team", "tr_predictive", "season"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {df.columns.tolist()}")
    
    # Clean and standardize team names
    df["team"] = df["team"].astype(str).str.strip()
    df["team"] = df["team"].replace(TR_TO_CANONICAL)
    
    # Ensure numeric
    df["tr_predictive"] = pd.to_numeric(df["tr_predictive"], errors="coerce")
    df = df.dropna(subset=["tr_predictive"])
    
    df = df[["season", "team", "tr_predictive"]]
    
    return df.sort_values(["season", "team"]).reset_index(drop=True)


def summarize_teamrankings(df: pd.DataFrame) -> None:
    """Print summary statistics for validation."""
    print(f"Total team-seasons: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Teams per season: {df.groupby('season').size().mean():.0f} avg")
    print(f"\nPredictive rating summary:")
    print(df["tr_predictive"].describe())


if __name__ == "__main__":
    for path in ["data/raw/teamrankings/tr_all.csv", "data/raw/teamrankings/"]:
        if Path(path).exists():
            print(f"Loading from {path}")
            df = extract_teamrankings(path)
            summarize_teamrankings(df)
            
            output_path = "data/processed/teamrankings_ratings.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
            break
    else:
        print("No TeamRankings data found. Run: python scrapers.py --source teamrankings")

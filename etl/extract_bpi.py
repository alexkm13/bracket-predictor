"""
ESPN BPI Data Extraction

Processes scraped BPI CSV files into the format needed by the model.
BPI data comes from the scraper output (scrapers.py --source bpi).

ESPN uses full mascot names like "Houston Cougars" or "UConn Huskies".
We need to strip the mascot and map to canonical names.

Input:  Scraped BPI CSV (bpi_all.csv or per-season bpi_{YYYY}.csv files)
Output: Cleaned DataFrame with columns:
        [season, team, bpi, bpi_off, bpi_def]
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


# ESPN BPI uses full names with mascots: "Duke Blue Devils", "UConn Huskies"
# This maps the ESPN full name to canonical tournament reference name.
# Most can be handled by stripping the mascot, but some need explicit mapping.
BPI_TO_CANONICAL = {
    # Schools where simple mascot stripping doesn't work
    "UConn Huskies": "UConn",
    "UConn": "UConn",
    "UCONN Huskies": "UConn",
    "LSU Tigers": "LSU",
    "LSU": "LSU",
    "SMU Mustangs": "SMU",
    "SMU": "SMU",
    "UCF Knights": "UCF",
    "UCF": "UCF",
    "VCU Rams": "VCU",
    "VCU": "VCU",
    "BYU Cougars": "BYU",
    "BYU": "BYU",
    "UNLV Rebels": "UNLV",
    "UNLV": "UNLV",
    "USC Trojans": "Southern California",
    "USC": "Southern California",
    "UTEP Miners": "UTEP",
    "UAB Blazers": "UAB",
    "UNC Greensboro Spartans": "UNC Greensboro",
    "UNC Asheville Bulldogs": "UNC Asheville",
    "UNC Wilmington Seahawks": "UNC Wilmington",
    "Miami Hurricanes": "Miami",
    "Miami (OH) RedHawks": "Miami (OH)",
    "NC State Wolfpack": "NC State",
    "Ole Miss Rebels": "Ole Miss",
    "Saint Mary's Gaels": "Saint Mary's",
    "Saint Peter's Peacocks": "Saint Peter's",
    "St. John's Red Storm": "St. John's",
    "St. Bonaventure Bonnies": "St. Bonaventure",
    "Loyola Chicago Ramblers": "Loyola Chicago",
    "LIU Sharks": "LIU",
    "FDU Knights": "FDU",
    "ETSU Buccaneers": "East Tennessee State",
    "SIU Edwardsville Cougars": "SIU Edwardsville",
    "UT Arlington Mavericks": "UT Arlington",
    "UT San Antonio Roadrunners": "UT San Antonio",
    "Texas A&M-CC Islanders": "Texas A&M-CC",
    "Detroit Mercy Titans": "Detroit Mercy",
    "Cal State Fullerton Titans": "Cal State Fullerton",
    "Cal State Bakersfield Roadrunners": "Cal State Bakersfield",
}

# Known mascot words to strip (order matters — check multi-word mascots first)
MASCOTS = [
    "Crimson Tide", "Blue Devils", "Tar Heels", "Golden Eagles", "Red Storm",
    "Scarlet Knights", "Yellow Jackets", "Orange", "Mountaineers", "Boilermakers",
    "Wolverines", "Buckeyes", "Spartans", "Hawkeyes", "Cyclones", "Jayhawks",
    "Wildcats", "Tigers", "Bulldogs", "Bears", "Eagles", "Cougars", "Huskies",
    "Knights", "Rams", "Rebels", "Trojans", "Gators", "Seminoles", "Volunteers",
    "Aggies", "Longhorns", "Sooners", "Cowboys", "Bruins", "Ducks", "Beavers",
    "Cardinals", "Hoosiers", "Badgers", "Hokies", "Cavaliers", "Demon Deacons",
    "Terrapins", "Nittany Lions", "Cornhuskers", "Illini", "Fighting Illini",
    "Bluejays", "Musketeers", "Friars", "Red Raiders", "Horned Frogs",
    "Billikens", "Flyers", "Peacocks", "Gaels", "Bonnies", "Ramblers",
    "Zags", "Toreros", "Waves", "Lions", "Owls", "Hawks", "Falcons",
    "Panthers", "Bearcats", "Shockers", "Phoenix", "Racers", "Governors",
    "Sharks", "Seahawks", "Chanticleers", "Paladins", "Catamounts",
    "Retrievers", "Great Danes", "Seawolves", "Thunderbirds", "Anteaters",
    "Gauchos", "Matadors", "Roadrunners", "Miners", "Blazers", "Jaguars",
    "Mean Green", "Red Wolves", "Ragin' Cajuns", "Bobcats", "Penguins",
    "Bison", "Leathernecks", "Salukis", "Redbirds", "Sycamores", "Braves",
    "Flames", "Titans", "49ers", "Highlanders", "Aggies", "Islanders",
    "Buccaneers", "RedHawks", "Wolfpack", "Wolf Pack",
]


def strip_mascot(name: str) -> str:
    """Remove mascot from ESPN team name to get school name."""
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    
    # Check explicit mapping first
    if name in BPI_TO_CANONICAL:
        return BPI_TO_CANONICAL[name]
    
    # Try stripping known mascots
    for mascot in MASCOTS:
        if name.endswith(f" {mascot}"):
            stripped = name[: -(len(mascot) + 1)].strip()
            if len(stripped) > 1:
                return stripped
    
    # If nothing matched, return as-is
    return name


def extract_bpi(input_path: str) -> pd.DataFrame:
    """
    Extract BPI ratings from scraped data.
    
    Parameters:
        input_path: path to bpi_all.csv or directory of bpi_{YYYY}.csv files
    
    Returns DataFrame with columns:
        [season, team, bpi, bpi_off, bpi_def]
    """
    path = Path(input_path)
    
    if path.is_file():
        df = pd.read_csv(path)
    elif path.is_dir():
        all_data = []
        for csv_file in sorted(path.glob("bpi_*.csv")):
            if "all" in csv_file.stem:
                continue
            all_data.append(pd.read_csv(csv_file))
        if not all_data:
            raise FileNotFoundError(f"No BPI CSVs found in {path}")
        df = pd.concat(all_data, ignore_index=True)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Ensure required columns exist
    required = ["team", "bpi", "season"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found: {df.columns.tolist()}")
    
    # Clean and standardize team names
    df["team"] = df["team"].apply(strip_mascot)
    
    # Ensure numeric
    df["bpi"] = pd.to_numeric(df["bpi"], errors="coerce")
    if "bpi_off" in df.columns:
        df["bpi_off"] = pd.to_numeric(df["bpi_off"], errors="coerce")
    else:
        df["bpi_off"] = np.nan
    if "bpi_def" in df.columns:
        df["bpi_def"] = pd.to_numeric(df["bpi_def"], errors="coerce")
    else:
        df["bpi_def"] = np.nan
    
    df = df.dropna(subset=["bpi"])
    df = df[["season", "team", "bpi", "bpi_off", "bpi_def"]]
    
    return df.sort_values(["season", "team"]).reset_index(drop=True)


def summarize_bpi(df: pd.DataFrame) -> None:
    """Print summary statistics for validation."""
    print(f"Total team-seasons: {len(df)}")
    print(f"Seasons: {sorted(df['season'].unique())}")
    print(f"Teams per season: {df.groupby('season').size().mean():.0f} avg")
    print(f"\nBPI summary:")
    print(df["bpi"].describe())


if __name__ == "__main__":
    for path in ["data/raw/bpi/bpi_all.csv", "data/raw/bpi/"]:
        if Path(path).exists():
            print(f"Loading from {path}")
            df = extract_bpi(path)
            summarize_bpi(df)
            
            output_path = "data/processed/bpi_ratings.csv"
            df.to_csv(output_path, index=False)
            print(f"\nSaved to {output_path}")
            break
    else:
        print("No BPI data found. Run: python scrapers.py --source bpi")

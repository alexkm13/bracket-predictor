"""
Team Name Standardization Module

Maps team names from different sources to a single canonical name.
The canonical name is the tournament reference name (REF___Post-Season_Tournament_Teams.csv)
since that's the name we'll display in the frontend.

Each source has its own quirks:
- KenPom: uses "St." abbreviations, "Connecticut" instead of "UConn"
- Barttorvik: similar to KenPom
- ESPN BPI: uses full mascot names like "UConn Huskies"  
- Sports Reference SRS: uses full names like "Connecticut Huskies"
- TeamRankings: uses short names like "UConn"

Strategy: build a lookup dict from each source's naming convention to the canonical name.
"""

import pandas as pd
from typing import Dict, Optional


# Known mappings from KenPom names to canonical (tournament reference) names.
# Pattern: KenPom uses "St." where reference uses "State" or drops "St." entirely.
KENPOM_TO_CANONICAL = {
    # St. -> State
    "Boise St.": "Boise State",
    "Colorado St.": "Colorado State",
    "Iowa St.": "Iowa State",
    "Michigan St.": "Michigan State",
    "Mississippi St.": "Mississippi State",
    "Montana St.": "Montana State",
    "Utah St.": "Utah State",
    "Washington St.": "Washington State",
    "San Diego St.": "San Diego State",
    "South Dakota St.": "South Dakota State",
    "Morehead St.": "Morehead State",
    "Long Beach St.": "Long Beach State",
    "McNeese St.": "McNeese",
    "Grambling St.": "Grambling",
    "Sacramento St.": "Sacramento State",
    "Kennesaw St.": "Kennesaw State",
    "Norfolk St.": "Norfolk State",
    "Cleveland St.": "Cleveland State",
    "Wichita St.": "Wichita State",
    "Wright St.": "Wright State",
    "Weber St.": "Weber State",
    "Portland St.": "Portland State",
    "Fresno St.": "Fresno State",
    "Oregon St.": "Oregon State",
    "Kansas St.": "Kansas State",
    "Penn St.": "Penn State",
    "Ohio St.": "Ohio State",
    "Georgia St.": "Georgia State",
    "Arizona St.": "Arizona State",
    "Murray St.": "Murray State",
    "Jacksonville St.": "Jacksonville State",
    "Alabama St.": "Alabama State",
    "Alcorn St.": "Alcorn State",
    "Appalachian St.": "Appalachian State",
    "Ball St.": "Ball State",
    "Coppin St.": "Coppin State",
    "Delaware St.": "Delaware State",
    "Illinois St.": "Illinois State",
    "Indiana St.": "Indiana State",
    "Missouri St.": "Missouri State",
    "New Mexico St.": "New Mexico State",
    "North Carolina St.": "NC State",
    "North Dakota St.": "North Dakota State",
    "Oklahoma St.": "Oklahoma State",
    "Sam Houston St.": "Sam Houston State",
    "San Jose St.": "San Jose State",
    "Texas St.": "Texas State",
    "Youngstown St.": "Youngstown State",

    # Other known differences
    "Connecticut": "UConn",
    "Central Connecticut": "Central Connecticut State",
    "Southern Miss.": "Southern Miss",
    "Detroit": "Detroit Mercy",
    "Cal St. Fullerton": "Cal State Fullerton",
    "Cal St. Bakersfield": "Cal State Bakersfield",
    "Cal St. Northridge": "Cal State Northridge",
    "Loyola Chicago": "Loyola Chicago",
    "Saint Mary's": "Saint Mary's",
    "St. Peter's": "Saint Peter's",
    "St. John's": "St. John's",
    "St. Bonaventure": "St. Bonaventure",
    "ETSU": "East Tennessee State",
    "UTSA": "UT San Antonio",
    "LIU Brooklyn": "LIU",
    "USC Upstate": "South Carolina Upstate",
    "UMass Lowell": "UMass Lowell",
    "UT Rio Grande Valley": "UTRGV",
    "Texas A&M Corpus Chris": "Texas A&M-CC",
    "Miami FL": "Miami",
    "Miami OH": "Miami (OH)",
    "USC": "Southern California",
    "UCF": "UCF",
    "VCU": "VCU",
    "SMU": "SMU",
    "BYU": "BYU",
    "UNLV": "UNLV",
    "UNC Greensboro": "UNC Greensboro",
    "UNC Asheville": "UNC Asheville",
    "UNC Wilmington": "UNC Wilmington",
}


def build_kenpom_name_map(kenpom_names: list, canonical_names: list) -> Dict[str, str]:
    """
    Build a complete mapping from KenPom names to canonical names.
    
    Uses the known mappings first, then attempts fuzzy matching for
    any remaining unmatched names.
    
    Parameters:
        kenpom_names: list of team names from KenPom data
        canonical_names: list of team names from tournament reference
    
    Returns:
        Dict mapping KenPom name -> canonical name
    """
    name_map = {}
    canonical_set = set(canonical_names)
    
    for name in kenpom_names:
        # Check if name already matches canonical
        if name in canonical_set:
            name_map[name] = name
        # Check known mappings
        elif name in KENPOM_TO_CANONICAL:
            mapped = KENPOM_TO_CANONICAL[name]
            if mapped in canonical_set:
                name_map[name] = mapped
            else:
                # Mapping exists but target not in canonical set
                # (team might not be in tournament that year)
                name_map[name] = mapped
        else:
            # Try automatic St. -> State conversion
            if "St." in name:
                expanded = name.replace(" St.", " State").replace("St. ", "Saint ")
                if expanded in canonical_set:
                    name_map[name] = expanded
                    continue
            
            # No match found - flag for manual review
            name_map[name] = name  # identity mapping, will need manual fix
    
    return name_map


def standardize_kenpom_names(df: pd.DataFrame, 
                              name_col: str = "TeamName",
                              canonical_names: Optional[list] = None) -> pd.DataFrame:
    """
    Standardize team names in a KenPom dataframe to canonical names.
    
    Parameters:
        df: DataFrame with KenPom data
        name_col: column containing team names
        canonical_names: optional list of canonical names to validate against
    
    Returns:
        DataFrame with standardized names
    """
    df = df.copy()
    
    if canonical_names is not None:
        name_map = build_kenpom_name_map(df[name_col].unique().tolist(), canonical_names)
    else:
        # Just apply known mappings
        name_map = KENPOM_TO_CANONICAL
    
    df[name_col] = df[name_col].map(lambda x: name_map.get(x, x))
    
    return df


def validate_name_matching(source_names: list, 
                            canonical_names: list,
                            source_label: str = "source") -> dict:
    """
    Check how well source names match canonical names.
    
    Returns dict with match stats and lists of unmatched names.
    """
    source_set = set(source_names)
    canonical_set = set(canonical_names)
    
    matched = source_set & canonical_set
    in_source_only = source_set - canonical_set
    in_canonical_only = canonical_set - source_set
    
    result = {
        "matched": len(matched),
        "total_source": len(source_set),
        "total_canonical": len(canonical_set),
        "match_rate": len(matched) / len(source_set) if source_set else 0,
        "unmatched_in_source": sorted(in_source_only),
        "unmatched_in_canonical": sorted(in_canonical_only),
    }
    
    if in_source_only:
        print(f"\n⚠️  {len(in_source_only)} {source_label} names not found in canonical:")
        for name in sorted(in_source_only):
            print(f"    {name}")
    
    if in_canonical_only:
        print(f"\n⚠️  {len(in_canonical_only)} canonical names not found in {source_label}:")
        for name in sorted(in_canonical_only):
            print(f"    {name}")
    
    return result


if __name__ == "__main__":
    # Quick test: load KenPom and tournament reference, check matching
    kenpom = pd.read_csv("data/raw/kenpom/pre_tournament_summary.csv")
    tourney = pd.read_csv("data/reference/tournament_teams.csv")
    
    # Test for 2024
    kenpom_2024 = kenpom[kenpom["Season"] == 2024]
    tourney_2024 = tourney[(tourney["Season"] == 2024) & 
                           (tourney["Post-Season Tournament"] == "March Madness")]
    
    seeded = kenpom_2024[kenpom_2024["Seed"].notna()]
    
    print("Before standardization:")
    validate_name_matching(
        seeded["TeamName"].tolist(),
        tourney_2024["Team Name"].tolist(),
        "KenPom"
    )
    
    standardized = standardize_kenpom_names(seeded, 
                                             canonical_names=tourney_2024["Team Name"].tolist())
    
    print("\nAfter standardization:")
    validate_name_matching(
        standardized["TeamName"].tolist(),
        tourney_2024["Team Name"].tolist(),
        "KenPom"
    )

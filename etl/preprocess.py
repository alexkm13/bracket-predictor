"""
Preprocessing Pipeline - Final Version

Handles all known data quality issues:
    - TeamRankings: strips (W-L) records from team names
    - Barttorvik: filters out conference abbreviation rows
    - BPI: handles abbreviated team names from scraper
    - SRS: maps full "State" names to KenPom convention
    - KenPom: maps "St." abbreviations to canonical names
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Tuple

from standardize_names import KENPOM_TO_CANONICAL


# ==============================================================
# MASTER NAME MAPPING
# ==============================================================
# All sources get mapped to KenPom-style names first,
# then KenPom names get mapped to canonical via KENPOM_TO_CANONICAL.
# This two-step approach means we only maintain one canonical mapping.

# SRS uses full names — map to KenPom style
# ==============================================================
# NAME MAPPING DICTIONARIES
# ==============================================================
# All three dictionaries map source names -> KenPom-style names.
# Then KENPOM_TO_CANONICAL (from standardize_names.py) maps to
# the final canonical form used in the ratings matrix.
#
# Two-step chain: Source raw name -> KenPom name -> Canonical name
#
# IMPORTANT: No duplicate keys. Each key appears exactly once.
# ==============================================================


SRS_TO_KENPOM = {
    # Sports Reference uses full "State" names, formal school names
    "Alabama State": "Alabama St.",
    "Albany (NY)": "Albany",
    "Alcorn State": "Alcorn St.",
    "Appalachian State": "Appalachian St.",
    "Arizona State": "Arizona St.",
    "Arkansas State": "Arkansas St.",
    "Arkansas-Little Rock": "Arkansas Little Rock",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "Ball State": "Ball St.",
    "Boise State": "Boise St.",
    "Bowling Green State": "Bowling Green",
    "Brigham Young": "BYU",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Cal State Northridge": "Cal St. Northridge",
    "California Baptist": "Cal Baptist",
    "Centenary (LA)": "Centenary",
    "Central Connecticut State": "Central Connecticut",
    "Central Florida": "UCF",
    "Charleston Southern": "Charleston So.",
    "Cleveland State": "Cleveland St.",
    "Colorado State": "Colorado St.",
    "Coppin State": "Coppin St.",
    "Delaware State": "Delaware St.",
    "East Tennessee State": "ETSU",
    "Fairleigh Dickinson": "Fairleigh Dickinson",
    "Florida Atlantic": "Florida Atlantic",
    "Florida State": "Florida St.",
    "Fresno State": "Fresno St.",
    "Gardner-Webb": "Gardner Webb",
    "Georgia State": "Georgia St.",
    "Grambling": "Grambling St.",
    "Grambling State": "Grambling St.",
    "Idaho State": "Idaho St.",
    "Illinois State": "Illinois St.",
    "Indiana State": "Indiana St.",
    "Iowa State": "Iowa St.",
    "Jackson State": "Jackson St.",
    "Jacksonville State": "Jacksonville St.",
    "Kansas State": "Kansas St.",
    "Kennesaw State": "Kennesaw St.",
    "Kent State": "Kent St.",
    "Little Rock": "Arkansas Little Rock",
    "Long Beach State": "Long Beach St.",
    "Long Island University": "LIU Brooklyn",
    "Louisiana State": "LSU",
    "Louisiana-Lafayette": "Louisiana Lafayette",
    "Louisiana-Monroe": "UL Monroe",
    "Loyola (IL)": "Loyola Chicago",
    "Loyola (MD)": "Loyola MD",
    "McNeese State": "McNeese St.",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Michigan State": "Michigan St.",
    "Middle Tennessee": "Middle Tennessee",
    "Mississippi": "Ole Miss",
    "Mississippi State": "Mississippi St.",
    "Mississippi Valley State": "Mississippi Valley St.",
    "Missouri State": "Missouri St.",
    "Montana State": "Montana St.",
    "Morehead State": "Morehead St.",
    "Morgan State": "Morgan St.",
    "Murray State": "Murray St.",
    "Nevada-Las Vegas": "UNLV",
    "New Mexico State": "New Mexico St.",
    "Norfolk State": "Norfolk St.",
    "North Carolina State": "North Carolina St.",
    "North Dakota State": "North Dakota St.",
    "Northwestern State": "Northwestern St.",
    "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.",
    "Oregon State": "Oregon St.",
    "Penn": "Penn",
    "Penn State": "Penn St.",
    "Portland State": "Portland St.",
    "Prairie View": "Prairie View",
    "Prairie View A&M": "Prairie View",
    "Sacramento State": "Sacramento St.",
    "Saint Joseph's": "St. Joseph's",
    "Saint Mary's (CA)": "Saint Mary's",
    "Saint Peter's": "St. Peter's",
    "Sam Houston": "Sam Houston St.",
    "Sam Houston State": "Sam Houston St.",
    "San Diego State": "San Diego St.",
    "San Jose State": "San Jose St.",
    "South Carolina State": "South Carolina St.",
    "South Carolina Upstate": "USC Upstate",
    "South Dakota State": "South Dakota St.",
    "Southeast Missouri State": "Southeast Missouri St.",
    "Southeastern Louisiana": "SE Louisiana",
    "Southern California": "USC",
    "Southern Illinois": "Southern Illinois",
    "Southern Methodist": "SMU",
    "Stephen F. Austin": "Stephen F. Austin",
    "Tennessee State": "Tennessee St.",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
    "Texas State": "Texas St.",
    "Troy": "Troy",
    "UT Arlington": "UT Arlington",
    "UT Rio Grande Valley": "UT Rio Grande Valley",
    "UT San Antonio": "UTSA",
    "Utah State": "Utah St.",
    "Virginia Commonwealth": "VCU",
    "Virginia Military Institute": "VMI",
    "Washington State": "Washington St.",
    "Weber State": "Weber St.",
    "Western Kentucky": "Western Kentucky",
    "Wichita State": "Wichita St.",
    "Wright State": "Wright St.",
    "Youngstown State": "Youngstown St.",
}


BPI_ABBREV_TO_KENPOM = {
    # ESPN BPI uses short all-caps abbreviations
    "AAMU": "Alabama A&M",
    "ACU": "Abilene Christian",
    "AKR": "Akron",
    "ALA": "Alabama",
    "ALCN": "Alcorn St.",
    "ALST": "Alabama St.",
    "AMCC": "Texas A&M Corpus Chris",
    "AMER": "American",
    "APP": "Appalachian St.",
    "APSU": "Austin Peay",
    "ARIZ": "Arizona",
    "ARK": "Arkansas",
    "ARKLR": "Arkansas Little Rock",
    "ARMY": "Army",
    "ARST": "Arkansas St.",
    "ASU": "Arizona St.",
    "AUB": "Auburn",
    "BALL": "Ball St.",
    "BAY": "Baylor",
    "BC": "Boston College",
    "BCU": "Bethune Cookman",
    "BEL": "Belmont",
    "BELL": "Bellarmine",
    "BGSU": "Bowling Green",
    "BING": "Binghamton",
    "BOIS": "Boise St.",
    "BRAD": "Bradley",
    "BRWN": "Brown",
    "BRY": "Bryant",
    "BUCK": "Bucknell",
    "BUF": "Buffalo",
    "BUT": "Butler",
    "BYU": "BYU",
    "CAL": "California",
    "CAM": "Campbell",
    "CAN": "Canisius",
    "CARK": "Central Arkansas",
    "CBAK": "Cal St. Bakersfield",
    "CBU": "Cal Baptist",
    "CCSU": "Central Connecticut",
    "CCU": "Coastal Carolina",
    "CHAR": "Charlotte",
    "CHAT": "Chattanooga",
    "CHS": "Charleston So.",
    "CHSO": "Charleston So.",
    "CHST": "Chicago St.",
    "CIN": "Cincinnati",
    "CIT": "The Citadel",
    "CLE": "Cleveland St.",
    "CLEM": "Clemson",
    "CLT": "Charlotte",
    "CLMB": "Columbia",
    "CMU": "Central Michigan",
    "COFC": "Charleston",
    "COLG": "Colgate",
    "COLO": "Colorado",
    "COLU": "Columbia",
    "CONN": "Connecticut",
    "COOK": "Bethune Cookman",
    "COPP": "Coppin St.",
    "COR": "Cornell",
    "CP": "Cal Poly",
    "CREI": "Creighton",
    "CSF": "Cal St. Fullerton",
    "CSU": "Charleston So.",
    "CSUB": "Cal St. Bakersfield",
    "CSUF": "Cal St. Fullerton",
    "CSUN": "Cal St. Northridge",
    "DART": "Dartmouth",
    "DAV": "Davidson",
    "DAY": "Dayton",
    "DEL": "Delaware",
    "DEN": "Denver",
    "DEP": "DePaul",
    "DET": "Detroit",
    "DETM": "Detroit",
    "DREX": "Drexel",
    "DRKE": "Drake",
    "DSU": "Delaware St.",
    "DUK": "Duke",
    "DUKE": "Duke",
    "DUQ": "Duquesne",
    "ECU": "East Carolina",
    "EIU": "Eastern Illinois",
    "EKU": "Eastern Kentucky",
    "ELON": "Elon",
    "EMU": "Eastern Michigan",
    "ETAM": "East Texas A&M",
    "ETSU": "ETSU",
    "EVAN": "Evansville",
    "EWU": "Eastern Washington",
    "FAIR": "Fairfield",
    "FAMU": "Florida A&M",
    "FAU": "Florida Atlantic",
    "FDU": "Fairleigh Dickinson",
    "FGCU": "Florida Gulf Coast",
    "FIU": "FIU",
    "FLA": "Florida",
    "FOR": "Fordham",
    "FRES": "Fresno St.",
    "FSU": "Florida St.",
    "FUR": "Furman",
    "GASO": "Georgia Southern",
    "GAST": "Georgia St.",
    "GCU": "Grand Canyon",
    "GMU": "George Mason",
    "GONZ": "Gonzaga",
    "GRAM": "Grambling St.",
    "GRBY": "Green Bay",
    "GTWN": "Georgetown",
    "GWEB": "Gardner Webb",
    "GWU": "George Washington",
    "HALL": "Seton Hall",
    "HAMP": "Hampton",
    "HART": "Hartford",
    "HARV": "Harvard",
    "HAW": "Hawaii",
    "HBU": "Houston Baptist",
    "HCU": "Houston Christian",
    "HOF": "Hofstra",
    "HOL": "Holy Cross",
    "HOU": "Houston",
    "HOW": "Howard",
    "HP": "High Point",
    "HPU": "High Point",
    "IDHO": "Idaho",
    "IDST": "Idaho St.",
    "ILL": "Illinois",
    "ILST": "Illinois St.",
    "IND": "Indiana",
    "INST": "Indiana St.",
    "IONA": "Iona",
    "IOWA": "Iowa",
    "ISU": "Iowa St.",
    "IUIN": "IU Indianapolis",
    "IUPU": "IUPUI",
    "JKST": "Jacksonville St.",
    "JMU": "James Madison",
    "JOES": "St. Joseph's",
    "JSU": "Jackson St.",
    "JXST": "Jackson St.",
    "JAX": "Jacksonville",
    "KAN": "Kansas",
    "KU": "Kansas",
    "KENN": "Kennesaw St.",
    "KENT": "Kent St.",
    "KSU": "Kansas St.",
    "LAF": "Lafayette",
    "LAM": "Lamar",
    "LAS": "La Salle",
    "LBSU": "Long Beach St.",
    "LEH": "Lehigh",
    "LEM": "Le Moyne",
    "LIB": "Liberty",
    "LIN": "Lindenwood",
    "LIP": "Lipscomb",
    "LIU": "LIU Brooklyn",
    "L-MD": "Loyola MD",
    "LMU": "Loyola Marymount",
    "LONG": "Longwood",
    "LOU": "Louisville",
    "LOY": "Loyola Chicago",
    "LOYM": "Loyola MD",
    "LSU": "LSU",
    "LT": "Louisiana Tech",
    "LTLR": "Arkansas Little Rock",
    "LUC": "Loyola Chicago",
    "M-OH": "Miami OH",
    "MAN": "Manhattan",
    "MARQ": "Marquette",
    "MARS": "Marshall",
    "MASS": "UMass",
    "MCN": "McNeese St.",
    "MCNS": "McNeese St.",
    "MD": "Maryland",
    "MEM": "Memphis",
    "MER": "Mercer",
    "MERC": "Mercer",
    "MIA": "Miami FL",
    "MICH": "Michigan",
    "MILW": "Milwaukee",
    "MINN": "Minnesota",
    "MIOH": "Miami OH",
    "MISS": "Ole Miss",
    "MIZ": "Missouri",
    "MOHO": "Morehead St.",
    "MONM": "Monmouth",
    "MONT": "Montana",
    "MORE": "Morehead St.",
    "MORG": "Morgan St.",
    "MOST": "Missouri St.",
    "MRSH": "Marshall",
    "MRMK": "Merrimack",
    "MRST": "Marist",
    "MSM": "Mount St. Mary's",
    "MSST": "Mississippi St.",
    "MSU": "Michigan St.",
    "MTST": "Montana St.",
    "MTSU": "Middle Tennessee",
    "MTU": "Middle Tennessee",
    "MUR": "Murray St.",
    "MURR": "Murray St.",
    "MVSU": "Mississippi Valley St.",
    "NAU": "Northern Arizona",
    "NAV": "Navy",
    "NAVY": "Navy",
    "NCAT": "North Carolina A&T",
    "NCCU": "North Carolina Central",
    "NCST": "North Carolina St.",
    "NCSU": "North Carolina St.",
    "ND": "Notre Dame",
    "NDSU": "North Dakota St.",
    "NE": "Northeastern",
    "NEB": "Nebraska",
    "NEV": "Nevada",
    "NHVN": "New Haven",
    "NIA": "Niagara",
    "NIAG": "Niagara",
    "NICH": "Nicholls St.",
    "NIU": "Northern Illinois",
    "NJIT": "NJIT",
    "NKU": "Northern Kentucky",
    "NM": "New Mexico",
    "NMSU": "New Mexico St.",
    "NORF": "Norfolk St.",
    "NOVA": "Villanova",
    "NW": "Northwestern",
    "NWST": "Northwestern St.",
    "OAK": "Oakland",
    "ODU": "Old Dominion",
    "OHIO": "Ohio",
    "OKLA": "Oklahoma",
    "OKST": "Oklahoma St.",
    "OMA": "Omaha",
    "ORE": "Oregon",
    "ORST": "Oregon St.",
    "ORU": "Oral Roberts",
    "OSU": "Ohio St.",
    "PAC": "Pacific",
    "PEAY": "Austin Peay",
    "PENN": "Penn",
    "PEPP": "Pepperdine",
    "PFW": "Purdue Fort Wayne",
    "PITT": "Pittsburgh",
    "PORT": "Portland",
    "PRE": "Presbyterian",
    "PRES": "Presbyterian",
    "PRIN": "Princeton",
    "PROV": "Providence",
    "PRST": "Portland St.",
    "PSU": "Penn St.",
    "PUR": "Purdue",
    "PV": "Prairie View",
    "PVAM": "Prairie View",
    "QUC": "Queens University",
    "QUIN": "Quinnipiac",
    "RAD": "Radford",
    "RGV": "UT Rio Grande Valley",
    "RICE": "Rice",
    "RICH": "Richmond",
    "RID": "Rider",
    "RMU": "Robert Morris",
    "RUTG": "Rutgers",
    "SAC": "Sacramento St.",
    "SAM": "Samford",
    "SAU": "South Alabama",
    "SBON": "St. Bonaventure",
    "SBU": "St. Bonaventure",
    "SC": "South Carolina",
    "SCST": "South Carolina St.",
    "SCUP": "USC Upstate",
    "SCU": "Santa Clara",
    "SDAK": "South Dakota",
    "SDST": "South Dakota St.",
    "SDSU": "San Diego St.",
    "SEA": "Seattle",
    "SELA": "SE Louisiana",
    "SEMO": "Southeast Missouri St.",
    "SF": "San Francisco",
    "SFA": "Stephen F. Austin",
    "SFBK": "St. Francis Brooklyn",
    "SFPA": "St. Francis PA",
    "SHU": "Sacred Heart",
    "SHSU": "Sam Houston St.",
    "SIE": "Siena",
    "SIU": "Southern Illinois",
    "SIUE": "SIU Edwardsville",
    "SJSU": "San Jose St.",
    "SJU": "St. John's",
    "SLU": "Saint Louis",
    "SMC": "Saint Mary's",
    "SMU": "SMU",
    "SOU": "Southern",
    "SPU": "St. Peter's",
    "STAN": "Stanford",
    "STBK": "Stony Brook",
    "STET": "Stetson",
    "STMN": "St. Thomas",
    "STO": "Stetson",
    "STON": "Stony Brook",
    "SUU": "Southern Utah",
    "SYR": "Syracuse",
    "TA&M": "Texas A&M",
    "TAMU": "Texas A&M",
    "TAR": "Tarleton St.",
    "TCU": "TCU",
    "TEM": "Temple",
    "TENN": "Tennessee",
    "TEX": "Texas",
    "TLSA": "Tulsa",
    "TNST": "Tennessee St.",
    "TNTC": "Tennessee Tech",
    "TNTH": "Tennessee Tech",
    "TOL": "Toledo",
    "TOW": "Towson",
    "TOWS": "Towson",
    "TROY": "Troy",
    "TTU": "Texas Tech",
    "TULN": "Tulane",
    "TXSO": "Texas Southern",
    "TXST": "Texas St.",
    "UAB": "UAB",
    "UAPB": "Arkansas Pine Bluff",
    "UALB": "Albany",
    "UCD": "UC Davis",
    "UCF": "UCF",
    "UCI": "UC Irvine",
    "UCLA": "UCLA",
    "UCR": "UC Riverside",
    "UCSB": "UC Santa Barbara",
    "UCSD": "UC San Diego",
    "UGA": "Georgia",
    "UIC": "Illinois Chicago",
    "UIW": "Incarnate Word",
    "UK": "Kentucky",
    "ULL": "Louisiana Lafayette",
    "ULM": "UL Monroe",
    "UMBC": "UMBC",
    "UMD": "Maryland",
    "UMES": "Maryland Eastern Shore",
    "UMKC": "UMKC",
    "UML": "UMass Lowell",
    "UNA": "North Alabama",
    "UNC": "North Carolina",
    "UNCA": "UNC Asheville",
    "UNCG": "UNC Greensboro",
    "UNCO": "Northern Colorado",
    "UNCW": "UNC Wilmington",
    "UND": "North Dakota",
    "UNF": "North Florida",
    "UNH": "New Hampshire",
    "UNI": "Northern Iowa",
    "UNLV": "UNLV",
    "UNM": "New Mexico",
    "UNO": "New Orleans",
    "UNT": "North Texas",
    "UPST": "USC Upstate",
    "URI": "Rhode Island",
    "USA": "South Alabama",
    "USC": "USC",
    "USD": "San Diego",
    "USF": "South Florida",
    "USI": "Southern Indiana",
    "USM": "Southern Miss",
    "USU": "Utah St.",
    "UTA": "Utah",
    "UTAH": "Utah",
    "UTAM": "UT Arlington",
    "UTC": "Chattanooga",
    "UTEP": "UTEP",
    "UTM": "UT Martin",
    "UTRGV": "UT Rio Grande Valley",
    "UTSA": "UTSA",
    "UTU": "Utah Tech",
    "UVA": "Virginia",
    "UVM": "Vermont",
    "UVU": "Utah Valley",
    "VAL": "Valparaiso",
    "VAN": "Vanderbilt",
    "VCU": "VCU",
    "VILL": "Villanova",
    "VMI": "VMI",
    "VT": "Virginia Tech",
    "W&M": "William & Mary",
    "WAG": "Wagner",
    "WAKE": "Wake Forest",
    "WASH": "Washington",
    "WCU": "Western Carolina",
    "WEB": "Weber St.",
    "WEBU": "Weber St.",
    "WGA": "West Georgia",
    "WICH": "Wichita St.",
    "WIL": "William & Mary",
    "WIN": "Winthrop",
    "WIS": "Wisconsin",
    "WISC": "Wisconsin",
    "WIU": "Western Illinois",
    "WKU": "Western Kentucky",
    "WMU": "Western Michigan",
    "WOF": "Wofford",
    "WRST": "Wright St.",
    "WSU": "Washington St.",
    "WVU": "West Virginia",
    "WYO": "Wyoming",
    "XAV": "Xavier",
    "YALE": "Yale",
    "YSU": "Youngstown St.",
}


BARTTORVIK_CONFERENCE_ROWS = {
    "A10", "ACC", "AE", "ASun", "Amer", "B10", "B12", "BE", "BSky",
    "BSth", "BW", "CAA", "CUSA", "Horz", "Ivy", "MAAC", "MAC", "MEAC",
    "MVC", "MWC", "NEC", "OVC", "P12", "Pat", "SB", "SC", "SEC",
    "Slnd", "Sum", "SWAC", "WAC", "WCC",
}


TR_SHORT_TO_KENPOM = {
    # TeamRankings uses short names, often with (W-L) records stripped
    "Abl Christian": "Abilene Christian",
    "Ala A&M": "Alabama A&M",
    "Alcorn": "Alcorn St.",
    "App State": "Appalachian St.",
    "Arizona St": "Arizona St.",
    "Ark Little Rock": "Arkansas Little Rock",
    "Ark St": "Arkansas St.",
    "AR-Pine Bluff": "Arkansas Pine Bluff",
    "Austin Peay St": "Austin Peay",
    "Boise St": "Boise St.",
    "Boston U": "Boston University",
    "Bowling Grn": "Bowling Green",
    "BYU": "BYU",
    "Cal Baptist": "Cal Baptist",
    "Cent Arkansas": "Central Arkansas",
    "Cent Conn St": "Central Connecticut",
    "Cent Michigan": "Central Michigan",
    "Charl Southern": "Charleston So.",
    "Chicago St": "Chicago St.",
    "Cleveland St": "Cleveland St.",
    "Coastal Car": "Coastal Carolina",
    "Col Charleston": "Charleston",
    "Connecticut": "Connecticut",
    "Coppin St": "Coppin St.",
    "CS Bakersfield": "Cal St. Bakersfield",
    "CS Fullerton": "Cal St. Fullerton",
    "CS Northridge": "Cal St. Northridge",
    "Delaware St": "Delaware St.",
    "Detroit": "Detroit",
    "E Carolina": "East Carolina",
    "E Illinois": "Eastern Illinois",
    "E Kentucky": "Eastern Kentucky",
    "E Michigan": "Eastern Michigan",
    "E Tennessee St": "ETSU",
    "E Washington": "Eastern Washington",
    "F Dickinson": "Fairleigh Dickinson",
    "FL Gulf Coast": "Florida Gulf Coast",
    "Fla Atlantic": "Florida Atlantic",
    "Fla Gulf Cst": "Florida Gulf Coast",
    "Florida St": "Florida St.",
    "Ga Southern": "Georgia Southern",
    "Ga Tech": "Georgia Tech",
    "Geo Mason": "George Mason",
    "Geo Washington": "George Washington",
    "Georgia St": "Georgia St.",
    "Grambling St": "Grambling St.",
    "Grn Bay": "Green Bay",
    "Houston Bap": "Houston Baptist",
    "Idaho St": "Idaho St.",
    "Ill Chicago": "Illinois Chicago",
    "Illinois St": "Illinois St.",
    "Indiana St": "Indiana St.",
    "Iowa St": "Iowa St.",
    "Jackson St": "Jackson St.",
    "Jksonville St": "Jacksonville St.",
    "Kansas St": "Kansas St.",
    "Kennesaw St": "Kennesaw St.",
    "Kent St": "Kent St.",
    "L-Chicago": "Loyola Chicago",
    "LA Lafayette": "Louisiana Lafayette",
    "LIU": "LIU Brooklyn",
    "LIU-Brooklyn": "LIU Brooklyn",
    "Lngwd": "Longwood",
    "Long Beach St": "Long Beach St.",
    "Loyola MD": "Loyola MD",
    "McNeese St": "McNeese St.",
    "MD E Shore": "Maryland Eastern Shore",
    "Miami FL": "Miami FL",
    "Miami OH": "Miami OH",
    "Michigan St": "Michigan St.",
    "Mid Tennessee": "Middle Tennessee",
    "Middle Tenn": "Middle Tennessee",
    "Miss St": "Mississippi St.",
    "Miss Val St": "Mississippi Valley St.",
    "Miss Valley St": "Mississippi Valley St.",
    "Mississippi": "Ole Miss",
    "Mississippi St": "Mississippi St.",
    "Missouri St": "Missouri St.",
    "Montana St": "Montana St.",
    "Morehead St": "Morehead St.",
    "Morgan St": "Morgan St.",
    "Mt St Mary's": "Mount St. Mary's",
    "Mt St Marys": "Mount St. Mary's",
    "Murray St": "Murray St.",
    "N Alabama": "North Alabama",
    "N Arizona": "Northern Arizona",
    "N Carolina": "North Carolina",
    "N Colo": "Northern Colorado",
    "N Dakota": "North Dakota",
    "N Dakota St": "North Dakota St.",
    "N Florida": "North Florida",
    "N Hampshire": "New Hampshire",
    "N Illinois": "Northern Illinois",
    "N Iowa": "Northern Iowa",
    "N Kentucky": "Northern Kentucky",
    "N Mex St": "New Mexico St.",
    "N Texas": "North Texas",
    "NC A&T": "North Carolina A&T",
    "NC Asheville": "UNC Asheville",
    "NC Central": "North Carolina Central",
    "NC State": "North Carolina St.",
    "Neb Omaha": "Omaha",
    "New Mexico": "New Mexico",
    "New Mexico St": "New Mexico St.",
    "New Orleans": "New Orleans",
    "Nicholls St": "Nicholls St.",
    "Norfolk St": "Norfolk St.",
    "NW State": "Northwestern St.",
    "Ohio St": "Ohio St.",
    "Oklahoma St": "Oklahoma St.",
    "Ole Miss": "Ole Miss",
    "Oregon St": "Oregon St.",
    "Penn St": "Penn St.",
    "Portland St": "Portland St.",
    "Prairie View": "Prairie View",
    "S Alabama": "South Alabama",
    "S Carolina": "South Carolina",
    "S Carolina St": "South Carolina St.",
    "S Dakota": "South Dakota",
    "S Dakota St": "South Dakota St.",
    "S Florida": "South Florida",
    "S Illinois": "Southern Illinois",
    "S Miss": "Southern Miss",
    "S Utah": "Southern Utah",
    "Sac State": "Sacramento St.",
    "Sam Houston St": "Sam Houston St.",
    "San Diego St": "San Diego St.",
    "San Jose St": "San Jose St.",
    "SC Upstate": "USC Upstate",
    "SE Louisiana": "SE Louisiana",
    "SE Missouri St": "Southeast Missouri St.",
    "St Bonaventure": "St. Bonaventure",
    "St Francis PA": "St. Francis PA",
    "St John's": "St. John's",
    "St Johns": "St. John's",
    "St Josephs": "St. Joseph's",
    "St Marys": "Saint Mary's",
    "St Peters": "St. Peter's",
    "St Thomas MN": "St. Thomas",
    "Ste F Austin": "Stephen F. Austin",
    "Stony Brook": "Stony Brook",
    "Tenn Martin": "UT Martin",
    "Tenn St": "Tennessee St.",
    "Tenn Tech": "Tennessee Tech",
    "Texas So": "Texas Southern",
    "Texas St": "Texas St.",
    "Texas Tech": "Texas Tech",
    "TX A&M-CC": "Texas A&M Corpus Chris",
    "Texas A&M": "Texas A&M",
    "TX Southern": "Texas Southern",
    "U Mass": "UMass",
    "UAB": "UAB",
    "UC Davis": "UC Davis",
    "UC Irvine": "UC Irvine",
    "UC Riverside": "UC Riverside",
    "UC Sn Barb": "UC Santa Barbara",
    "UCF": "UCF",
    "UL Lafayette": "Louisiana Lafayette",
    "UL Monroe": "UL Monroe",
    "UMBC": "UMBC",
    "UMKC": "UMKC",
    "UNC Asheville": "UNC Asheville",
    "UNC Greensboro": "UNC Greensboro",
    "UNC Wilmington": "UNC Wilmington",
    "UNLV": "UNLV",
    "USC": "USC",
    "UT Arlington": "UT Arlington",
    "UT Pan Am": "UT Rio Grande Valley",
    "UT San Antonio": "UTSA",
    "Utah St": "Utah St.",
    "Utah Valley": "Utah Valley",
    "UTEP": "UTEP",
    "VCU": "VCU",
    "W Carolina": "Western Carolina",
    "W Illinois": "Western Illinois",
    "W Kentucky": "Western Kentucky",
    "W Michigan": "Western Michigan",
    "Wash St": "Washington St.",
    "Weber St": "Weber St.",
    "Wichita St": "Wichita St.",
    "Wm & Mary": "William & Mary",
    "Wright St": "Wright St.",
    "Youngstown St": "Youngstown St.",
}


# ==============================================================
# LOAD FUNCTIONS
# ==============================================================

def load_kenpom(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "Year" in df.columns:
        df = df.rename(columns={"Year": "season", "TeamName": "team", "AdjEM": "kenpom_adj_em", "seed": "seed"})
    else:
        df = df.rename(columns={"Season": "season", "TeamName": "team", "AdjEM": "kenpom_adj_em", "Seed": "seed"})
    df = df[df["seed"].notna()].copy()
    df["seed"] = df["seed"].astype(int)
    df["kenpom_adj_em"] = pd.to_numeric(df["kenpom_adj_em"], errors="coerce")
    df["team"] = df["team"].replace(KENPOM_TO_CANONICAL)
    # Fix 2026 seeds — FormulaBot data has wrong seeds
    seed_fix = {
        2026: {
            "Duke": 1, "Arizona": 1, "Michigan": 1, "Florida": 1,
            "Houston": 2, "Iowa State": 2, "Purdue": 2, "Connecticut": 2,
            "Illinois": 3, "Michigan State": 3, "Gonzaga": 3, "Virginia": 3,
            "Kansas": 4, "Nebraska": 4, "Arkansas": 4, "Alabama": 4,
            "St. John's": 5, "Vanderbilt": 5, "Wisconsin": 5, "Texas Tech": 5,
            "Tennessee": 6, "Louisville": 6, "BYU": 6, "North Carolina": 6,
            "Saint Mary's": 7, "UCLA": 7, "Kentucky": 7, "Miami": 7,
            "Ohio State": 8, "Clemson": 8, "Villanova": 8, "Georgia": 8,
            "TCU": 9, "Iowa": 9, "Utah State": 9, "Saint Louis": 9,
            "UCF": 10, "Texas A&M": 10, "Missouri": 10, "Santa Clara": 10,
            "South Florida": 11, "VCU": 11, "Texas": 11, "NC State": 11,
            "SMU": 11, "Miami OH": 11,
            "Northern Iowa": 12, "McNeese": 12, "Akron": 12, "High Point": 12,
            "California Baptist": 13, "Hofstra": 13, "Troy": 13, "Hawaii": 13,
            "North Dakota State": 14, "Pennsylvania": 14, "Kennesaw State": 14, "Wright State": 14,
            "Furman": 15, "Idaho": 15, "Queens University": 15, "Tennessee State": 15,
            "Siena": 16, "Prairie View A&M": 16, "Lehigh": 16, "Long Island University": 16,
            "UMBC": 16, "Howard": 16,
        }
    }
    for year, fixes in seed_fix.items():
        for team_name, correct_seed in fixes.items():
            mask = (df["season"] == year) & (df["team"] == team_name)
            df.loc[mask, "seed"] = correct_seed
    return df[["season", "team", "seed", "kenpom_adj_em"]].dropna(subset=["kenpom_adj_em"]).reset_index(drop=True)


def load_barttorvik(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if not {"season", "team", "barttorvik_adj_em"}.issubset(df.columns):
        raise ValueError(f"Expected [season, team, barttorvik_adj_em], got {df.columns.tolist()}")
    
    df["team"] = df["team"].astype(str).str.strip()
    
    # Filter out conference abbreviation rows
    df = df[~df["team"].isin(BARTTORVIK_CONFERENCE_ROWS)].copy()
    
    # Map to canonical via KenPom convention
    df["team"] = df["team"].replace(KENPOM_TO_CANONICAL)
    df["barttorvik_adj_em"] = pd.to_numeric(df["barttorvik_adj_em"], errors="coerce")
    
    return df[["season", "team", "barttorvik_adj_em"]].dropna(subset=["barttorvik_adj_em"]).reset_index(drop=True)


def load_bpi(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    
    if not {"season", "team", "bpi"}.issubset(set(df.columns)):
        raise ValueError(f"Expected [season, team, bpi], got {df.columns.tolist()}")
    
    df["team"] = df["team"].astype(str).str.strip()
    
    # Check if names are abbreviations (short, all caps) or full names
    sample = df["team"].head(20)
    is_abbreviated = sample.str.match(r'^[A-Z]{2,5}$').mean() > 0.5
    
    if is_abbreviated:
        df["team"] = df["team"].replace(BPI_ABBREV_TO_KENPOM)
    
    # Then map KenPom -> canonical
    df["team"] = df["team"].replace(KENPOM_TO_CANONICAL)
    df["bpi"] = pd.to_numeric(df["bpi"], errors="coerce")
    
    return df[["season", "team", "bpi"]].dropna(subset=["bpi"]).reset_index(drop=True)


def load_srs(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    
    if not {"season", "team", "srs"}.issubset(set(df.columns)):
        raise ValueError(f"Expected [season, team, srs], got {df.columns.tolist()}")
    
    df["team"] = df["team"].astype(str).str.strip()
    
    # SRS -> KenPom naming -> canonical
    df["team"] = df["team"].replace(SRS_TO_KENPOM)
    df["team"] = df["team"].replace(KENPOM_TO_CANONICAL)
    df["srs"] = pd.to_numeric(df["srs"], errors="coerce")
    
    return df[["season", "team", "srs"]].dropna(subset=["srs"]).reset_index(drop=True)


def load_teamrankings(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    
    if not {"season", "team", "tr_predictive"}.issubset(set(df.columns)):
        raise ValueError(f"Expected [season, team, tr_predictive], got {df.columns.tolist()}")
    
    df["team"] = df["team"].astype(str).str.strip()
    
    # Strip (W-L) records: "Duke (32-5)" -> "Duke"
    df["team"] = df["team"].str.replace(r'\s*\(\d+-\d+\)\s*$', '', regex=True)
    
    # TR short names -> KenPom -> canonical
    df["team"] = df["team"].replace(TR_SHORT_TO_KENPOM)
    df["team"] = df["team"].replace(KENPOM_TO_CANONICAL)
    df["tr_predictive"] = pd.to_numeric(df["tr_predictive"], errors="coerce")
    
    return df[["season", "team", "tr_predictive"]].dropna(subset=["tr_predictive"]).reset_index(drop=True)


def load_game_results(filepath: str, min_season: int = 2008) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="latin-1")
    
    if "margin" in df.columns:
        return df[df["season"] >= min_season].reset_index(drop=True)
    
    df.columns = ["season", "round_num", "region_num", "region", "seed_i", "score_i", "team_i", "team_j", "score_j", "seed_j"]
    for col in ["team_i", "team_j"]:
        df[col] = df[col].astype(str).str.strip()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["score_i"] = pd.to_numeric(df["score_i"], errors="coerce").astype(int)
    df["score_j"] = pd.to_numeric(df["score_j"], errors="coerce").astype(int)
    df["seed_i"] = pd.to_numeric(df["seed_i"], errors="coerce").astype(int)
    df["seed_j"] = pd.to_numeric(df["seed_j"], errors="coerce").astype(int)
    df["margin"] = df["score_i"] - df["score_j"]
    df["round"] = df["round_num"].map({1:"R64",2:"R32",3:"S16",4:"E8",5:"F4",6:"CG"})
    return df[df["season"] >= min_season].reset_index(drop=True)


# ==============================================================
# COMPOSITE + JOIN + STANDARDIZE
# ==============================================================

def composite_efficiency_sources(kenpom, barttorvik=None):
    if barttorvik is None:
        result = kenpom[["season", "team", "kenpom_adj_em"]].copy()
        return result.rename(columns={"kenpom_adj_em": "source_1"})
    merged = kenpom[["season", "team", "kenpom_adj_em"]].merge(barttorvik, on=["season", "team"], how="outer")
    merged["source_1"] = merged[["kenpom_adj_em", "barttorvik_adj_em"]].mean(axis=1)
    return merged[["season", "team", "source_1"]]


def build_ratings_matrix(tournament_teams, source_1, source_2=None, source_3=None, source_4=None):
    result = tournament_teams.copy()
    result = result.merge(source_1, on=["season", "team"], how="left")
    for col_name, src in {"source_2": source_2, "source_3": source_3, "source_4": source_4}.items():
        if src is not None:
            rcol = [c for c in src.columns if c not in ["season", "team"]][0]
            src = src.rename(columns={rcol: col_name})
            result = result.merge(src[["season", "team", col_name]], on=["season", "team"], how="left")
        else:
            result[col_name] = np.nan
    return result


def standardize_within_year(ratings, source_cols):
    standardized = ratings.copy()
    params_list = []
    for season in ratings["season"].unique():
        mask = ratings["season"] == season
        for col in source_cols:
            values = ratings.loc[mask, col]
            if values.notna().sum() < 3:
                continue
            m, s = float(values.mean()), float(values.std())
            if s > 0:
                standardized.loc[mask, col] = (values - m) / s
            params_list.append({"season": int(season), "source": col, "mean": m, "std": s, "n_teams": int(values.notna().sum())})
    return standardized, pd.DataFrame(params_list)


def validate_ratings_matrix(ratings, source_cols):
    print("=" * 60)
    print("RATINGS MATRIX VALIDATION")
    print("=" * 60)
    print(f"\nTotal team-seasons: {len(ratings)}")
    print(f"Seasons: {int(ratings['season'].min())} - {int(ratings['season'].max())}")
    print(f"Avg teams per season: {ratings.groupby('season').size().mean():.0f}")
    print(f"\nSource coverage:")
    for col in source_cols:
        n = ratings[col].notna().sum()
        print(f"  {col}: {n}/{len(ratings)} ({n/len(ratings)*100:.1f}%)")
    print(f"\nCoverage by season:")
    for season in sorted(ratings["season"].unique()):
        mask = ratings["season"] == season
        parts = [f"{col}={ratings.loc[mask, col].notna().sum()}/{int(mask.sum())}" for col in source_cols]
        print(f"  {int(season)}: {', '.join(parts)}")
    for col in source_cols:
        missing = ratings[ratings[col].isna()]
        if len(missing) > 0:
            print(f"\n  {col} missing ({len(missing)} team-seasons):")
            for _, row in missing.head(10).iterrows():
                print(f"    {int(row['season'])} {row['team']} (seed {int(row['seed'])})")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")


# ==============================================================
# MAIN
# ==============================================================

def build_model_dataset(kenpom_path, barttorvik_path=None, bpi_path=None,
                         srs_path=None, teamrankings_path=None,
                         game_results_path=None, output_dir="data/processed"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("Loading KenPom...")
    kenpom = load_kenpom(kenpom_path)
    tournament_teams = kenpom[["season", "team", "seed"]].copy()
    print(f"  {len(kenpom)} team-seasons ({int(kenpom['season'].min())}-{int(kenpom['season'].max())})")
    
    barttorvik = None
    if barttorvik_path and Path(barttorvik_path).exists():
        print("Loading Barttorvik...")
        barttorvik = load_barttorvik(barttorvik_path)
        print(f"  {len(barttorvik)} team-seasons")
    
    bpi_df = None
    if bpi_path and Path(bpi_path).exists():
        print("Loading BPI...")
        bpi_df = load_bpi(bpi_path)
        print(f"  {len(bpi_df)} team-seasons")
    
    srs_df = None
    if srs_path and Path(srs_path).exists():
        print("Loading SRS...")
        srs_df = load_srs(srs_path)
        print(f"  {len(srs_df)} team-seasons")
    
    tr_df = None
    if teamrankings_path and Path(teamrankings_path).exists():
        print("Loading TeamRankings...")
        tr_df = load_teamrankings(teamrankings_path)
        print(f"  {len(tr_df)} team-seasons")
    
    print("Building Source 1 (efficiency composite)...")
    source_1 = composite_efficiency_sources(kenpom, barttorvik)
    print(f"  {source_1['source_1'].notna().sum()} teams with source_1")
    
    print("Assembling ratings matrix...")
    ratings = build_ratings_matrix(tournament_teams, source_1, tr_df, bpi_df, srs_df)
    source_cols = ["source_1", "source_2", "source_3", "source_4"]
    
    validate_ratings_matrix(ratings, source_cols)
    
    print("\nSkipping standardization (model handles scale via observation layer)...")
    # Save raw ratings as both filenames so fit.py finds it
    ratings.to_csv(out / "ratings_matrix_raw.csv", index=False)
    ratings.to_csv(out / "ratings_matrix_standardized.csv", index=False)
    params = pd.DataFrame()  # empty placeholder
    params.to_csv(out / "standardization_params.csv", index=False)
    print(f"\nSaved to {out}/")
    
    if game_results_path and Path(game_results_path).exists():
        print("\nLoading game results...")
        games = load_game_results(game_results_path)
        games.to_csv(out / "tournament_games.csv", index=False)
        print(f"  {len(games)} games saved")
    
    return ratings, params


if __name__ == "__main__":
    ratings, params = build_model_dataset(
        kenpom_path="data/raw/kenpom/kenpom_march_madness_2000_2026.csv",
        barttorvik_path="data/raw/barttorvik/bt_all.csv",
        bpi_path="data/raw/bpi/bpi_all.csv",
        srs_path="data/raw/srs/srs_all.csv",
        teamrankings_path="data/raw/teamrankings/tr_all.csv",
        game_results_path="data/raw/game_results/big_dance.csv",
        output_dir="data/processed",
    )
    
    print("\n" + "=" * 60)
    print("DATASET READY FOR MODEL")
    print("=" * 60)
    print(f"Shape: {ratings.shape}")
    active = [c for c in ["source_1", "source_2", "source_3", "source_4"] if ratings[c].notna().any()]
    print(f"Active sources: {active}")
    print(f"\nNext step: python model.py")
"""
NBA Over/Under Predictor - Constants Module

This module contains all constant values used throughout the application,
including team mappings, season type definitions, and configuration values.
"""

# ==============================================================================
# TEAM MAPPINGS
# ==============================================================================

# Maps full team names to their NBA API team IDs
TEAM_ID_MAP = {
    "Atlanta Hawks": "1610612737",
    "Boston Celtics": "1610612738",
    "Cleveland Cavaliers": "1610612739",
    "New Orleans Pelicans": "1610612740",
    "Chicago Bulls": "1610612741",
    "Dallas Mavericks": "1610612742",
    "Denver Nuggets": "1610612743",
    "Golden State Warriors": "1610612744",
    "Houston Rockets": "1610612745",
    "Los Angeles Clippers": "1610612746",
    "Los Angeles Lakers": "1610612747",
    "Miami Heat": "1610612748",
    "Milwaukee Bucks": "1610612749",
    "Minnesota Timberwolves": "1610612750",
    "Brooklyn Nets": "1610612751",
    "New York Knicks": "1610612752",
    "Orlando Magic": "1610612753",
    "Indiana Pacers": "1610612754",
    "Philadelphia 76ers": "1610612755",
    "Phoenix Suns": "1610612756",
    "Portland Trail Blazers": "1610612757",
    "Sacramento Kings": "1610612758",
    "San Antonio Spurs": "1610612759",
    "Oklahoma City Thunder": "1610612760",
    "Toronto Raptors": "1610612761",
    "Utah Jazz": "1610612762",
    "Memphis Grizzlies": "1610612763",
    "Washington Wizards": "1610612764",
    "Detroit Pistons": "1610612765",
    "Charlotte Hornets": "1610612766",
}

# Maps various team name formats to standardized full names
# Handles abbreviations, historical names, and alternative spellings
TEAM_NAME_STANDARDIZATION = {
    # LA Clippers special cases
    "LA Clippers": "Los Angeles Clippers",
    "L.A. Clippers": "Los Angeles Clippers",
    # Historical team names
    "New Orleans/Oklahoma City Hornets": "New Orleans Hornets",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Jersey Nets": "Brooklyn Nets",
    "Seattle SuperSonics": "Oklahoma City Thunder",
    "New Orleans Hornets": "New Orleans Pelicans",
    # Atlanta Hawks
    "ATL Hawks": "Atlanta Hawks",
    "Atlanta Hawks": "Atlanta Hawks",
    # Boston Celtics
    "BOS Celtics": "Boston Celtics",
    "Boston Celtics": "Boston Celtics",
    # Brooklyn Nets
    "BKN Nets": "Brooklyn Nets",
    "Brooklyn Nets": "Brooklyn Nets",
    # Charlotte Hornets / Bobcats
    "CHA Hornets": "Charlotte Hornets",
    "Charlotte Hornets": "Charlotte Hornets",
    # Chicago Bulls
    "CHI Bulls": "Chicago Bulls",
    "Chicago Bulls": "Chicago Bulls",
    # Cleveland Cavaliers
    "CLE Cavaliers": "Cleveland Cavaliers",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    # Dallas Mavericks
    "DAL Mavericks": "Dallas Mavericks",
    "Dallas Mavericks": "Dallas Mavericks",
    # Denver Nuggets
    "DEN Nuggets": "Denver Nuggets",
    "Denver Nuggets": "Denver Nuggets",
    # Detroit Pistons
    "DET Pistons": "Detroit Pistons",
    "Detroit Pistons": "Detroit Pistons",
    # Golden State Warriors
    "GS Warriors": "Golden State Warriors",
    "Golden State Warriors": "Golden State Warriors",
    # Houston Rockets
    "HOU Rockets": "Houston Rockets",
    "Houston Rockets": "Houston Rockets",
    # Indiana Pacers
    "IND Pacers": "Indiana Pacers",
    "Indiana Pacers": "Indiana Pacers",
    # Los Angeles Clippers
    "Los Angeles Clippers": "Los Angeles Clippers",
    # Los Angeles Lakers
    "LA Lakers": "Los Angeles Lakers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    # Memphis Grizzlies
    "MEM Grizzlies": "Memphis Grizzlies",
    "Memphis Grizzlies": "Memphis Grizzlies",
    # Miami Heat
    "MIA Heat": "Miami Heat",
    "Miami Heat": "Miami Heat",
    # Milwaukee Bucks
    "MIL Bucks": "Milwaukee Bucks",
    "Milwaukee Bucks": "Milwaukee Bucks",
    # Minnesota Timberwolves
    "MIN Timberwolves": "Minnesota Timberwolves",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    # New Orleans Pelicans
    "NO Pelicans": "New Orleans Pelicans",
    "New Orleans Pelicans": "New Orleans Pelicans",
    # New York Knicks
    "NY Knicks": "New York Knicks",
    "New York Knicks": "New York Knicks",
    # Oklahoma City Thunder
    "OKC Thunder": "Oklahoma City Thunder",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    # Orlando Magic
    "ORL Magic": "Orlando Magic",
    "Orlando Magic": "Orlando Magic",
    # Philadelphia 76ers
    "PHI 76ers": "Philadelphia 76ers",
    "Philadelphia 76ers": "Philadelphia 76ers",
    # Phoenix Suns
    "PHO Suns": "Phoenix Suns",
    "Phoenix Suns": "Phoenix Suns",
    # Portland Trail Blazers
    "POR Trail Blazers": "Portland Trail Blazers",
    "Portland Trail Blazers": "Portland Trail Blazers",
    # Sacramento Kings
    "SAC Kings": "Sacramento Kings",
    "Sacramento Kings": "Sacramento Kings",
    # San Antonio Spurs
    "SA Spurs": "San Antonio Spurs",
    "San Antonio Spurs": "San Antonio Spurs",
    # Toronto Raptors
    "TOR Raptors": "Toronto Raptors",
    "Toronto Raptors": "Toronto Raptors",
    # Utah Jazz
    "UTA Jazz": "Utah Jazz",
    "Utah Jazz": "Utah Jazz",
    # Washington Wizards
    "WAS Wizards": "Washington Wizards",
    "Washington Wizards": "Washington Wizards",
    # Handle errors or irrelevant entries
    "Everton FC": None,  # Not an NBA team
}

# ==============================================================================
# SEASON TYPE MAPPINGS
# ==============================================================================

# Maps game ID prefixes to season types
SEASON_TYPE_MAP = {
    "001": "Preseason",
    "002": "Regular Season",
    "003": "All Star",
    "004": "Playoffs",
    "005": "Play-In Tournament",
    "006": "In-Season Final Game",
}

# ==============================================================================
# GAME CONFIGURATION
# ==============================================================================

# Standard NBA game duration in minutes (excluding overtime)
REGULATION_GAME_MINUTES = 240

# Minimum minutes threshold for overtime detection
OVERTIME_THRESHOLD_MINUTES = 259

# ==============================================================================
# STATISTICAL WINDOWS
# ==============================================================================

# Default rolling window sizes for various statistics
DEFAULT_ROLLING_WINDOW = 5
WEIGHTED_ROLLING_WINDOW = 10

# Weights for weighted moving average (most recent to oldest)
WEIGHTED_MA_WEIGHTS = [20, 15, 10, 8, 6, 5, 4, 3, 2, 1]

# Minimum minutes played threshold for player statistics
MIN_MINUTES_THRESHOLD = 15

# Default number of top players to analyze
DEFAULT_TOP_PLAYERS = 3

# ==============================================================================
# EXTERNAL API CONFIGURATION
# ==============================================================================

# NBA Injury Report URL
NBA_INJURY_REPORTS_URL = "https://official.nba.com/nba-injury-report-2025-26-season/"

# Sports data API sport ID for NBA
NBA_SPORT_ID = 4

# ==============================================================================
# BACKWARD COMPATIBILITY
# ==============================================================================

# Maintain old variable names for backward compatibility
TEAM_CONVERSION_DICT = TEAM_ID_MAP  # Legacy name
TEAM_NAME_EQUIVALENT_DICT = TEAM_NAME_STANDARDIZATION  # Legacy name
SEASON_TYPE_MAPPING = SEASON_TYPE_MAP  # Legacy name

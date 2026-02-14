"""
Mapping from BoxScoreSummaryV3 column names to legacy V2 column names
for referees and injuries data.

This maintains backward compatibility with existing database schemas
while using the newer V3 API endpoints.
"""

# Mapping for Officials/Referees data
V3_TO_V2_OFFICIALS_MAP = {
    # V3 -> V2 column names
    "gameId": "GAME_ID",
    "personId": "OFFICIAL_ID",  # V3 uses personId, not officialId
    "firstName": "FIRST_NAME",
    "familyName": "LAST_NAME",
    "jerseyNum": "JERSEY_NUM",
    # These don't exist in V3, will be filled separately
    # "SEASON_ID": None,  # Added programmatically
    # "GAME_DATE": None,  # Added programmatically
}

# Mapping for Inactive/Injured players data
V3_TO_V2_INJURIES_MAP = {
    # V3 -> V2 column names
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamCity": "TEAM_CITY",
    "teamName": "TEAM_NAME",
    "teamTricode": "TEAM_ABBREVIATION",
    "personId": "PLAYER_ID",
    "firstName": "FIRST_NAME",
    "familyName": "LAST_NAME",
    "jerseyNum": "JERSEY_NUM",
    # These don't exist in V3, will be filled separately
    # "SEASON_ID": None,  # Added programmatically
    # "GAME_DATE": None,  # Added programmatically
}

# V2 columns that must be present but don't exist in V3
# These will be added with None values or filled programmatically
V2_REQUIRED_OFFICIALS_COLUMNS = [
    "OFFICIAL_ID",
    "FIRST_NAME",
    "LAST_NAME",
    "JERSEY_NUM",
    "GAME_ID",
    "SEASON_ID",
    "GAME_DATE",
]

V2_REQUIRED_INJURIES_COLUMNS = [
    "PLAYER_ID",
    "FIRST_NAME",
    "LAST_NAME",
    "JERSEY_NUM",
    "TEAM_ID",
    "TEAM_CITY",
    "TEAM_NAME",
    "TEAM_ABBREVIATION",
    "GAME_ID",
    "SEASON_ID",
    "GAME_DATE",
]

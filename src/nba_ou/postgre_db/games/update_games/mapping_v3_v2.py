V3_TO_V2_TRADITIONAL_MAP = {
    # Game / Team identifiers
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamName": "TEAM_NAME",
    "teamTricode": "TEAM_ABBREVIATION",
    "teamCity": "TEAM_CITY",
    # Player identifiers / names
    "personId": "PLAYER_ID",
    "nameI": "PLAYER_NAME",  # closest single-field equivalent in V3
    "position": "START_POSITION",  # V3 has one position field; V2 distinguishes start position
    "comment": "COMMENT",
    # Minutes
    "minutes": "MIN",
    # Shooting
    "fieldGoalsMade": "FGM",
    "fieldGoalsAttempted": "FGA",
    "fieldGoalsPercentage": "FG_PCT",
    "threePointersMade": "FG3M",
    "threePointersAttempted": "FG3A",
    "threePointersPercentage": "FG3_PCT",
    "freeThrowsMade": "FTM",
    "freeThrowsAttempted": "FTA",
    "freeThrowsPercentage": "FT_PCT",
    # Rebounds
    "reboundsOffensive": "OREB",
    "reboundsDefensive": "DREB",
    "reboundsTotal": "REB",
    # Other counting stats
    "assists": "AST",
    "steals": "STL",
    "blocks": "BLK",
    "turnovers": "TOV",
    "foulsPersonal": "PF",
    "points": "PTS",
    # Plus minus
    "plusMinusPoints": "PLUS_MINUS",
}

V3_TO_V2_ADVANCED_PLAYER_MAP = {
    # IDs / team descriptors
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamName": "TEAM_NAME",

    "teamTricode": "TEAM_ABBREVIATION",
    "teamCity": "TEAM_CITY",
    # Player descriptors
    "personId": "PLAYER_ID",
    "nameI": "PLAYER_NAME",  # closest single-field equivalent
    "position": "START_POSITION",  # best approximation (V3 does not explicitly mark "start")
    "comment": "COMMENT",
    "minutes": "MIN",
    # Ratings
    "estimatedOffensiveRating": "E_OFF_RATING",
    "offensiveRating": "OFF_RATING",
    "estimatedDefensiveRating": "E_DEF_RATING",
    "defensiveRating": "DEF_RATING",
    "estimatedNetRating": "E_NET_RATING",
    "netRating": "NET_RATING",
    # Playmaking / usage
    "assistPercentage": "AST_PCT",
    "assistToTurnover": "AST_TOV",
    "assistRatio": "AST_RATIO",
    # Rebounding
    "offensiveReboundPercentage": "OREB_PCT",
    "defensiveReboundPercentage": "DREB_PCT",
    "reboundPercentage": "REB_PCT",
    # Turnovers
    "turnoverRatio": "TM_TOV_PCT",  # closest available in your V3 list for V2 TM_TOV_PCT
    # Shooting efficiency
    "effectiveFieldGoalPercentage": "EFG_PCT",
    "trueShootingPercentage": "TS_PCT",
    # Usage / pace / possessions
    "usagePercentage": "USG_PCT",
    "estimatedUsagePercentage": "E_USG_PCT",
    "estimatedPace": "E_PACE",
    "pace": "PACE",
    "pacePer40": "PACE_PER40",
    "possessions": "POSS",
    # Misc
    "PIE": "PIE",
}

V3_TO_V2_ADVANCED_TEAM_MAP = {
    # IDs / team descriptors
    "gameId": "GAME_ID",
    "teamId": "TEAM_ID",
    "teamName": "TEAM_NAME",
    "teamTricode": "TEAM_ABBREVIATION",
    "teamCity": "TEAM_CITY",
    "minutes": "MIN",
    # Ratings
    "estimatedOffensiveRating": "E_OFF_RATING",
    "offensiveRating": "OFF_RATING",
    "estimatedDefensiveRating": "E_DEF_RATING",
    "defensiveRating": "DEF_RATING",
    "estimatedNetRating": "E_NET_RATING",
    "netRating": "NET_RATING",
    # Playmaking / usage
    "assistPercentage": "AST_PCT",
    "assistToTurnover": "AST_TOV",
    "assistRatio": "AST_RATIO",
    # Rebounding
    "offensiveReboundPercentage": "OREB_PCT",
    "defensiveReboundPercentage": "DREB_PCT",
    "reboundPercentage": "REB_PCT",
    # Turnovers (Team schema has two in V2)
    "estimatedTeamTurnoverPercentage": "E_TM_TOV_PCT",
    "turnoverRatio": "TM_TOV_PCT",  # closest available in your V3 list for V2 TM_TOV_PCT
    # Shooting efficiency
    "effectiveFieldGoalPercentage": "EFG_PCT",
    "trueShootingPercentage": "TS_PCT",
    # Usage / pace / possessions
    "usagePercentage": "USG_PCT",
    "estimatedUsagePercentage": "E_USG_PCT",
    "estimatedPace": "E_PACE",
    "pace": "PACE",
    "pacePer40": "PACE_PER40",
    "possessions": "POSS",
    # Misc
    "PIE": "PIE",
}

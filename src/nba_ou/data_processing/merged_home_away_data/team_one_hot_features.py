import re

import pandas as pd
from nba_ou.config.constants import TEAM_ID_MAP

TEAM_ID_BY_NAME = {
    team_name: str(team_id) for team_name, team_id in TEAM_ID_MAP.items()
}
TEAM_NAME_BY_ID = {team_id: team_name for team_name, team_id in TEAM_ID_BY_NAME.items()}

CATEGORICAL_HOME_COLUMN = "TEAM_HOME_CATEGORY_BEFORE"
CATEGORICAL_AWAY_COLUMN = "TEAM_AWAY_CATEGORY_BEFORE"


def _team_slug(team_name: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "_", team_name.upper()).strip("_")
    return slug


def add_team_one_hot_features(
    df_merged: pd.DataFrame, categorical_team_encoding: bool = False
) -> pd.DataFrame:
    """Add team-identity features for home and away teams.

    categorical_team_encoding=False (default): adds 60 binary
        TEAM_HOME_<SLUG>_BEFORE / TEAM_AWAY_<SLUG>_BEFORE columns.
    categorical_team_encoding=True: adds two pandas categorical columns
        (TEAM_HOME_CATEGORY_BEFORE, TEAM_AWAY_CATEGORY_BEFORE) for native categorical
        handling by gradient-boosted models (e.g. XGBoost enable_categorical=True).
    """
    required_cols = {"TEAM_ID_TEAM_HOME", "TEAM_ID_TEAM_AWAY"}
    missing_cols = sorted(required_cols - set(df_merged.columns))
    if missing_cols:
        raise ValueError(f"df_merged is missing columns: {missing_cols}")

    out = df_merged.copy()
    home_team_ids = out["TEAM_ID_TEAM_HOME"].astype(str)
    away_team_ids = out["TEAM_ID_TEAM_AWAY"].astype(str)

    if not categorical_team_encoding:
        new_cols = {}
        for team_name, team_id in TEAM_ID_BY_NAME.items():
            slug = _team_slug(team_name)
            new_cols[f"TEAM_HOME_{slug}_BEFORE"] = home_team_ids.eq(team_id).astype(int)
            new_cols[f"TEAM_AWAY_{slug}_BEFORE"] = away_team_ids.eq(team_id).astype(int)
        return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    category_values = list(TEAM_ID_BY_NAME.keys())
    home_names = home_team_ids.map(TEAM_NAME_BY_ID)
    away_names = away_team_ids.map(TEAM_NAME_BY_ID)
    new_cols = {
        CATEGORICAL_HOME_COLUMN: pd.Categorical(home_names, categories=category_values),
        CATEGORICAL_AWAY_COLUMN: pd.Categorical(away_names, categories=category_values),
    }
    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

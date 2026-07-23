"""Leakage-safe team style and opponent-matchup features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.utils.general_utils import _with_before_suffix

STYLE_OFFENSIVE_RATE_COLUMNS = (
    "STYLE_FG3A_RATE",
    "STYLE_FTA_RATE",
    "STYLE_TOV_RATE",
    "STYLE_OREB_RATE",
    "STYLE_FGA_PER_POSS",
)

STYLE_ALLOWED_RATE_COLUMNS = (
    "STYLE_FG3A_RATE_ALLOWED",
    "STYLE_FTA_RATE_ALLOWED",
    "STYLE_TOV_FORCED_RATE",
    "STYLE_OREB_RATE_ALLOWED",
    "STYLE_FGA_PER_POSS_ALLOWED",
)

STYLE_SOURCE_COLUMNS = STYLE_OFFENSIVE_RATE_COLUMNS + STYLE_ALLOWED_RATE_COLUMNS

_RATE_INPUTS = {
    "STYLE_FG3A_RATE": ("FG3A", "FGA"),
    "STYLE_FTA_RATE": ("FTA", "FGA"),
    "STYLE_TOV_RATE": ("TOV", "POSS"),
    "STYLE_OREB_RATE": ("OREB", "POSS"),
    "STYLE_FGA_PER_POSS": ("FGA", "POSS"),
}

_ALLOWED_FROM_OFFENSIVE = {
    "STYLE_FG3A_RATE_ALLOWED": "STYLE_FG3A_RATE",
    "STYLE_FTA_RATE_ALLOWED": "STYLE_FTA_RATE",
    "STYLE_TOV_FORCED_RATE": "STYLE_TOV_RATE",
    "STYLE_OREB_RATE_ALLOWED": "STYLE_OREB_RATE",
    "STYLE_FGA_PER_POSS_ALLOWED": "STYLE_FGA_PER_POSS",
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def _opponent_value(
    df: pd.DataFrame,
    values: pd.Series,
    *,
    game_id_col: str,
    team_id_col: str,
) -> pd.Series:
    """Return the other team's value for valid two-team games.

    Missing scheduled-game box scores remain missing. They are never replaced
    by the opponent or by a zero, which ensures they cannot enter their own
    historical rolling features.
    """
    numeric_values = pd.to_numeric(values, errors="coerce")
    games = df[game_id_col]
    value_count = numeric_values.notna().groupby(games).transform("sum")
    value_sum = numeric_values.groupby(games).transform("sum")
    team_count = df[team_id_col].groupby(games).transform("nunique")

    opponent = value_sum - numeric_values
    return opponent.where((value_count == 2) & (team_count == 2))


def add_team_style_source_features(
    df_team: pd.DataFrame,
    *,
    game_id_col: str = "GAME_ID",
    team_id_col: str = "TEAM_ID",
) -> pd.DataFrame:
    """Add same-game style observations used only as lagged history sources.

    The offensive rate columns describe the team row. The allowed/forced
    columns contain the opponent's same-game rate and therefore describe what
    the team allowed defensively in that completed game. These columns are not
    prediction features themselves; ``compute_all_rolling_statistics`` shifts
    them before creating ``*_BEFORE`` features.
    """
    required_keys = {game_id_col, team_id_col}
    missing_keys = required_keys - set(df_team.columns)
    if missing_keys:
        raise ValueError(
            "add_team_style_source_features missing required columns: "
            f"{sorted(missing_keys)}"
        )

    out = df_team.copy()
    new_columns: dict[str, pd.Series] = {}

    for output_col, (numerator_col, denominator_col) in _RATE_INPUTS.items():
        if numerator_col in out.columns and denominator_col in out.columns:
            new_columns[output_col] = _safe_ratio(
                out[numerator_col], out[denominator_col]
            )

    if new_columns:
        out = pd.concat(
            [out, pd.DataFrame(new_columns, index=out.index)],
            axis=1,
        )

    allowed_columns: dict[str, pd.Series] = {}
    for allowed_col, offensive_col in _ALLOWED_FROM_OFFENSIVE.items():
        if offensive_col in out.columns:
            allowed_columns[allowed_col] = _opponent_value(
                out,
                out[offensive_col],
                game_id_col=game_id_col,
                team_id_col=team_id_col,
            )

    if allowed_columns:
        out = pd.concat(
            [out, pd.DataFrame(allowed_columns, index=out.index)],
            axis=1,
        )

    return out


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def _pair_mean(first: pd.Series | None, second: pd.Series | None) -> pd.Series | None:
    if first is None or second is None:
        return None
    return pd.concat([first, second], axis=1).mean(axis=1, skipna=True)


def add_style_matchup_features(
    df_games: pd.DataFrame,
    *,
    history_suffix: str = "SEASON_BEFORE_AVG",
    drop_source_history: bool = True,
) -> pd.DataFrame:
    """Add compact home-vs-away style interactions from lagged inputs only.

    The required team inputs all contain ``_BEFORE`` in ``history_suffix``.
    This function deliberately has no access to same-game FGA, FTA, TOV, OREB,
    or possessions.
    """
    if "BEFORE" not in history_suffix:
        raise ValueError("history_suffix must refer to leakage-safe BEFORE columns.")

    out = df_games.copy()
    expected_possessions = _numeric_column(out, "EXPECTED_POSS_FROM_PACE_BEFORE")
    new_columns: dict[str, pd.Series] = {}

    def side_column(source: str, side: str) -> pd.Series | None:
        return _numeric_column(out, f"{source}_{history_suffix}_TEAM_{side}")

    expected_rates: dict[str, tuple[pd.Series | None, pd.Series | None]] = {}
    rate_definitions = {
        "FG3A": ("STYLE_FG3A_RATE", "STYLE_FG3A_RATE_ALLOWED"),
        "FTA": ("STYLE_FTA_RATE", "STYLE_FTA_RATE_ALLOWED"),
        "TOV": ("STYLE_TOV_RATE", "STYLE_TOV_FORCED_RATE"),
        "OREB": ("STYLE_OREB_RATE", "STYLE_OREB_RATE_ALLOWED"),
    }

    for label, (offensive_source, defensive_source) in rate_definitions.items():
        expected_home = _pair_mean(
            side_column(offensive_source, "HOME"),
            side_column(defensive_source, "AWAY"),
        )
        expected_away = _pair_mean(
            side_column(offensive_source, "AWAY"),
            side_column(defensive_source, "HOME"),
        )
        expected_rates[label] = (expected_home, expected_away)

        if expected_home is not None and expected_away is not None:
            new_columns[
                _with_before_suffix(f"STYLE_EXPECTED_{label}_RATE_HOME")
            ] = expected_home
            new_columns[
                _with_before_suffix(f"STYLE_EXPECTED_{label}_RATE_AWAY")
            ] = expected_away

    expected_fga_per_poss_home = _pair_mean(
        side_column("STYLE_FGA_PER_POSS", "HOME"),
        side_column("STYLE_FGA_PER_POSS_ALLOWED", "AWAY"),
    )
    expected_fga_per_poss_away = _pair_mean(
        side_column("STYLE_FGA_PER_POSS", "AWAY"),
        side_column("STYLE_FGA_PER_POSS_ALLOWED", "HOME"),
    )

    if expected_possessions is not None:
        expected_fg3a_home, expected_fg3a_away = expected_rates["FG3A"]
        if (
            expected_fg3a_home is not None
            and expected_fg3a_away is not None
            and expected_fga_per_poss_home is not None
            and expected_fga_per_poss_away is not None
        ):
            new_columns[_with_before_suffix("STYLE_EXPECTED_TOTAL_FG3A")] = (
                expected_possessions
                * (
                    expected_fga_per_poss_home * expected_fg3a_home
                    + expected_fga_per_poss_away * expected_fg3a_away
                )
            )

        expected_fta_home, expected_fta_away = expected_rates["FTA"]
        if (
            expected_fta_home is not None
            and expected_fta_away is not None
            and expected_fga_per_poss_home is not None
            and expected_fga_per_poss_away is not None
        ):
            new_columns[_with_before_suffix("STYLE_EXPECTED_TOTAL_FTA")] = (
                expected_possessions
                * (
                    expected_fga_per_poss_home * expected_fta_home
                    + expected_fga_per_poss_away * expected_fta_away
                )
            )

        for label in ("TOV", "OREB"):
            expected_home, expected_away = expected_rates[label]
            if expected_home is not None and expected_away is not None:
                new_columns[
                    _with_before_suffix(f"STYLE_EXPECTED_TOTAL_{label}")
                ] = expected_possessions * (expected_home + expected_away)

    expected_fta_home, expected_fta_away = expected_rates["FTA"]
    referee_foul_effect = _numeric_column(out, "REF_AVG_TOTAL_PF_DIFF_BEFORE")
    if (
        expected_fta_home is not None
        and expected_fta_away is not None
        and referee_foul_effect is not None
    ):
        combined_fta_rate = (expected_fta_home + expected_fta_away) / 2.0
        new_columns[_with_before_suffix("STYLE_FTA_REFEREE_INTERACTION")] = (
            combined_fta_rate * referee_foul_effect
        )

    if new_columns:
        out = pd.concat(
            [out, pd.DataFrame(new_columns, index=out.index)],
            axis=1,
        )

    if drop_source_history:
        source_history_columns = [
            f"{source}_{history_suffix}_TEAM_{side}"
            for source in STYLE_SOURCE_COLUMNS
            for side in ("HOME", "AWAY")
        ]
        out = out.drop(columns=source_history_columns, errors="ignore")

    return out

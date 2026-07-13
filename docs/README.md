# Training Data Processing

This document describes how this project builds the training and prediction feature
set for NBA over/under models. It is written as an implementation reference for
future agents, so it focuses on the actual pipeline in the codebase rather than
only the modeling intent.

The main entry point is
`src/nba_ou/create_training_data/create_df_to_predict.py`, specifically
`create_df_to_predict()`. Despite the function name, it is used for both
historical training data and same-day prediction data.

## Entry Points

Historical training data is usually created by calling:

```python
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict

df_train = create_df_to_predict(
    todays_prediction=False,
    recent_limit_to_include="2026-03-04",
    older_season_limit=None,
)
```

Common scripts that call this path:

- `scripts/create_train_data/create_train_data.py`
- `scripts/create_train_data/create_and_upload_historical_train_data.py`
- `scripts/retrain_prediction_models.py`
- The `__main__` block in `create_df_to_predict.py`

Same-day prediction data is created by first collecting scheduled-game context
with `get_all_info_for_scheduled_games()`, then passing that dictionary into
`create_df_to_predict(todays_prediction=True, scheduled_data=...)`.

```python
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict

scheduled_data = get_all_info_for_scheduled_games(
    date_to_predict="2026-04-12",
    nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
)

df_to_predict = create_df_to_predict(
    todays_prediction=True,
    scheduled_data=scheduled_data,
    strict_mode=30,
)
```

## High-Level Pipeline

The pipeline builds one final row per NBA game. Most intermediate processing is
team-level, with two rows per game, then the home and away rows are merged into
one game-level row.

At a high level, `create_df_to_predict()` does this:

1. Determine the historical cutoff date.
2. Determine which seasons to load.
3. Optionally collect extra historical matchup games for same-day prediction.
4. Load team game logs and player boxscores from PostgreSQL.
5. Load and merge Yahoo plus Sportsbook Review odds.
6. Clean and enrich team-level rows.
7. Load injury data and attach player/injury features.
8. Merge home and away team rows into one game row.
9. Merge remaining game-level odds.
10. Add all-star voting, team identity, betting, market-regime, referee,
    odds-engineered, injury-effect, travel, and date features.
11. Select final training columns and return a DataFrame.

## Date And Season Selection

The cutoff is controlled by `recent_limit_to_include`.

- If `todays_prediction=False` and no cutoff is provided, the cutoff defaults to
  yesterday in US/Pacific time.
- If `todays_prediction=True` and no cutoff is provided, the cutoff is inferred
  as the day immediately before the scheduled games. This prevents current-day
  games from entering historical rolling statistics.

Season selection is based on the cutoff date:

- With `older_season_limit=None`, historical training defaults to all seasons
  from the 2017-18 season onward.
- With `older_season_limit=N`, the pipeline includes the current season and the
  previous `N - 1` seasons.
- For same-day prediction, the default is two seasons unless
  `older_season_limit` is provided.

Filtering is performed by
`filter_by_seasons_with_extra_game_ids()`, which keeps rows from the selected
season years and applies the date cutoff. Extra game IDs can be preserved even
when they fall outside the ordinary season filter, as long as they respect the
upper date cap.

## Historical Inputs

The historical path loads these main data sources:

- Team game logs from `load_all_nba_data_from_db()`.
- Player boxscores from `load_all_nba_data_from_db()`.
- Yahoo odds from `load_odds_yahoo_from_db()`.
- Sportsbook Review odds from `load_odds_sportsbook_from_db()`.
- Injury rows from `get_injury_data_from_db()`.
- Referee rows from `get_refs_data_from_db()`.
- All-Star voting rows from `load_all_star_voting_from_db()`.

The active sportsbook is configured through `nba_ou.config.odds_columns`.
`get_main_book()` returns `SETTINGS.main_sportsbook` when configured, otherwise
it falls back to `consensus_opener`. The configured book controls canonical
column names such as:

- `TOTAL_LINE_<book>`
- `SPREAD_<book>`
- `MONEYLINE_<book>`

## Same-Day Prediction Inputs

Same-day prediction requires a `scheduled_data` dictionary created by
`get_all_info_for_scheduled_games()`. It contains:

- `scheduled_games`: games from the NBA schedule endpoint.
- `df_referees_scheduled`: processed scheduled referee assignments.
- `injury_dict_scheduled`: game/team/player injury lookup for scheduled games.
- `df_odds_yahoo_scheduled`: Yahoo odds for the scheduled games.
- `df_odds_sportsbook_scheduled`: Sportsbook Review odds for the scheduled games.
- `games_not_updated`: scheduled games whose injury report status was not usable.

Scheduled team rows are appended to the historical team table by
`standardize_and_merge_scheduled_games_to_team_data()`. Scheduled player
placeholder rows are appended by
`standardize_and_merge_scheduled_games_to_players_data()` so roster and player
cumulative features can be computed consistently for future games.

The scheduled odds data is merged with historical odds by
`merge_and_validate_scheduled_odds()`. That function checks that scheduled odds
columns are compatible with historical odds columns, drops extra scheduled-only
columns, logs null-count diagnostics when `strict_mode >= 0`, then concatenates
historical and scheduled odds.

For scheduled games, the pipeline may also include extra historical game IDs for
the exact home/away matchups being predicted. These are discovered through
`get_historical_game_ids_for_home_away_matchups()` and help matchup-specific
features have more history.

## Team-Level Processing

Team-level processing is handled by `process_team_statistics_for_training()`.
At this point the data has one row per team per game.

The major steps are:

1. `clean_team_data()`
   - Converts `GAME_DATE` to datetime.
   - Drops rows with missing points.
   - Drops duplicate `(GAME_ID, TEAM_ID)` rows.
   - Removes zero-minute rows and rows with implausibly low points.
   - Converts `TEAM_ID` to string.
   - Attempts to fix home/away parsing issues from `MATCHUP`.

2. `adjust_overtime()`
   - Creates `IS_OVERTIME`.
   - Normalizes most numeric stats from overtime games to a 48-minute equivalent.
   - Leaves columns such as `MIN`, `PACE_PER40`, IDs, and `IS_OVERTIME` unchanged.

3. `merge_total_spread_moneyline_by_game_id()`
   - Merges the selected book's spread and moneyline by `GAME_ID`.
   - Merges total lines for all known total sources when `total_lines_mode="all"`.
   - Creates canonical columns such as `TOTAL_LINE_betmgm`, `TOTAL_LINE_bet365`,
     `SPREAD_<book>`, and `MONEYLINE_<book>`.
   - Assigns spread and moneyline from the current team's perspective.

4. `compute_total_points_features()`
   - Creates `TOTAL_POINTS` as home plus away points, repeated on both team rows.
   - For each `TOTAL_LINE_*` column, creates `DIFF_FROM_LINE_<book>` as actual
     total points minus that line.

5. `filter_valid_games()`
   - Keeps only games with exactly two team entries.
   - Classifies season type.
   - Drops preseason and All-Star games.
   - Adds integer `SEASON_YEAR`.

6. `add_last_season_playoff_games()`
   - Counts each team's playoff games from the previous season.

7. `add_team_record_before_game()`
   - Adds `GAME_NUMBER`, `WINS_BEFORE_THIS_GAME`, and
     `TEAM_RECORD_BEFORE_GAME`.

8. `compute_rest_days_before_match()`
   - Adds `REST_DAYS_BEFORE_MATCH` within team and season.

9. `merge_odds_percentages_and_prices_by_game_id()`
   - Merges public betting percentages, consensus percentages, and price columns
     before rolling features are computed.
   - Total-market prices are game-level.
   - Spread and moneyline prices are converted to the current team's side.

10. `compute_all_rolling_statistics()`
    - Adds the bulk of team-form, betting-form, trend, weighted-average, and
      season-to-date features.

## Rolling And Trend Features

Rolling features are implemented in `src/nba_ou/data_processing/team/rolling.py`
and `src/nba_ou/data_processing/statistics/statistics.py`.

Most rolling features are explicitly shifted by one game, so they represent
information available before the current game.

Core team stats rolled over recent games include:

- Points and total points.
- Offensive, defensive, and net rating.
- Effective field goal percentage and true shooting percentage.
- Pace, possessions, PIE.
- Field goals, threes, free throws, and personal fouls.

Betting-related rolling inputs include:

- Main and per-book total lines.
- Spread and moneyline columns.
- `DIFF_FROM_*` columns.
- Public betting percentages from Yahoo.
- Consensus percentages.
- Price columns for totals, spreads, and moneylines.

Feature patterns include:

- `*_LAST_ALL_5_MATCHES_BEFORE`: previous five games, any location.
- `*_LAST_HOME_AWAY_5_MATCHES_BEFORE`: same-location rolling average minus the
  all-games rolling average.
- `*_LAST_ALL_10_MATCHES_BEFORE`: previous ten games for selected features.
- `*_LAST_5_MINUS_LAST_10_MATCHES_BEFORE`: short-window minus longer-window form.
- `*_LAST_1_MATCHES_BEFORE`, `*_LAST_2_MATCHES_BEFORE`,
  `*_LAST_3_MATCHES_BEFORE`: short windows for line-difference features.
- `*_LAST_5_WMA_BEFORE`: weighted moving average with more weight on recent games.
- `*_SEASON_BEFORE_AVG`: season-to-date average before the current game, with a
  previous-season fallback.
- `*_SEASON_BEFORE_STD`: expanding season-to-date standard deviation before the
  current game, with a previous-season fallback.
- `*_TREND_SLOPE_LAST_5_GAMES_BEFORE`: linear trend slope over prior games.
- `*_TREND_SLOPE_LAST_10_GAMES_BEFORE` and
  `*_TREND_SLOPE_LAST_5_MINUS_LAST_10_GAMES_BEFORE`: trend comparison features.

`compute_all_rolling_statistics()` dynamically discovers new `TOTAL_LINE_*`,
`DIFF_FROM_*`, percentage, consensus, and price columns, so adding a new book can
automatically expand the rolling feature set when the column naming convention
matches the existing patterns.

## Player And Injury Features

Player processing is handled by `process_player_statistics_for_training()` and
`add_player_history_features()`.

The player table is first cleaned by `clear_player_statistics()`:

- Game dates and season fields are merged from the team table.
- `MIN` is parsed from `MM:SS` into decimal minutes.
- Duplicate rows are dropped.
- Rows are filtered to the selected seasons and cutoff date.

For scheduled prediction, synthetic player rows are appended from each player's
latest known row. This lets the feature code identify likely roster membership
for a future game without using future boxscore data.

The pipeline computes player features for these stats:

- `PTS`
- `PACE_PER40`
- `DEF_RATING`
- `OFF_RATING`
- `TS_PCT`
- `MIN`

For each stat, `precompute_cumulative_avg_stat()` creates a recency-weighted
player estimate using an EWMA with a 10-game halflife. It is shifted so the
current game's stat is excluded. The lookup is grouped by `SEASON_YEAR` and
`PLAYER_ID`.

For each team-game, the pipeline identifies active and injured players using:

- Historical injury database rows.
- Player boxscore comments containing injury wording.
- Scheduled injury-report data for same-day predictions.
- Roster membership inferred from each player's last team before the game date.

Feature families added at the team row level include:

- Top six active-player IDs, names, and prior averages by stat.
- Top four injured-player IDs, names, and prior averages by stat.
- Average top-injured-player value by stat.
- Total active-player and injured-player prior value by stat.
- Number of active and injured players.
- Injury streak features for injured players by points.
- Bench scoring and pace features from players averaging roughly 7 to 21 minutes.

After home/away merging, player columns are suffixed by side, for example:

- `TOP1_PLAYER_PTS_BEFORE_TEAM_HOME`
- `TOP1_INJURED_PLAYER_PTS_BEFORE_TEAM_AWAY`
- `TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME`
- `N_ACTIVE_PLAYERS_BEFORE_TEAM_AWAY`

## All-Star Voting Features

The pipeline adds all-star fan-vote features before the home/away merge. The
pipeline computes required All-Star voting season years from team `GAME_DATE`
values with `all_star_season_year_for_game_date()`, then loads those years from
PostgreSQL through `load_all_star_voting_from_db()`.

The season-year mapping is intentionally calendar-sensitive:

- Games before March 1 use the All-Star voting season from two NBA season starts
  earlier.
- Games on or after March 1 use the prior NBA season start year.

For example, February 20, 2026 maps to All-Star `season_year=2024`, while
March 2, 2026 maps to `season_year=2025`.

`add_all_star_voting_features()` combines:

- Team-game rows.
- Cleaned player rows, including roster movement inferred from latest team
  before the game date.
- All-Star voting rows with `season_year`, `player_id`, `team_name`,
  `fan_votes`, and optional `score`.
- The same injury dictionary used by player features.

The function validates that every required voting season has usable rows. If a
required season is missing or has zero total fan votes, the pipeline raises a
`ValueError` and asks for the Basketball Reference voting data to be scraped,
rebuilt, and uploaded.

Team-level all-star columns initially include:

- `ALL_STAR_FAN_VOTE_SHARE_BEFORE`
- `ALL_STAR_MIN_SCORE_BEFORE`
- `ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE`
- `ALL_STAR_MIN_INJURED_SCORE_BEFORE`
- `ALL_STAR_FAN_VOTES_BEFORE`
- `ALL_STAR_CANDIDATE_COUNT_BEFORE`
- `ALL_STAR_SEASON_YEAR_BEFORE`

After `merge_home_away_data()`, `create_df_to_predict()` drops the audit/count
columns and keeps only the side-specific predictive columns:

- `ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_HOME`
- `ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY`
- `ALL_STAR_MIN_SCORE_BEFORE_TEAM_HOME`
- `ALL_STAR_MIN_SCORE_BEFORE_TEAM_AWAY`
- `ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE_TEAM_HOME`
- `ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY`
- `ALL_STAR_MIN_INJURED_SCORE_BEFORE_TEAM_HOME`
- `ALL_STAR_MIN_INJURED_SCORE_BEFORE_TEAM_AWAY`

The all-star vote share denominator is league-wide total fan votes for the
selected All-Star voting season. Traded players are handled by checking the
player's last known team before the game date and by also adding current roster
players who appear in that season's All-Star voting rows.

## Home/Away Merge

`merge_home_away_data()` converts the two team rows per game into one row per
game.

It separates `HOME == True` and `HOME == False`, then merges on game-level keys:

- `SEASON_ID`
- `GAME_ID`
- `GAME_DATE`
- `SEASON_TYPE`
- `SEASON_YEAR`
- `IS_OVERTIME`
- `GAME_TIME` for same-day prediction, when available

Home-side columns get `_TEAM_HOME`. Away-side columns get `_TEAM_AWAY`.

During this stage, the pipeline also:

- Renames top-player and average-injured columns with `_BEFORE`.
- Creates star contribution features, such as
  `STAR_OFFENSIVE_RATIO_IMPROVEMENT_BEFORE` and
  `STAR_PTS_PERCENTAGE_BEFORE`.
- Deduplicates game-level `TOTAL_LINE_*` columns that were duplicated by the
  home/away merge.
- Creates game-level `TOTAL_POINTS` and `TOTAL_PF`.
- Adds `IS_PLAYOFF_GAME_BEFORE`.
- Computes home/away points-conceded features and historical matchup features.
- Creates `TEAMS_DIFFERENCE_OVER_UNDER_LINE_BEFORE` when both side-specific
  season average total-line columns are available.

Immediately after the home/away merge, `create_df_to_predict()` calls
`add_team_one_hot_features()`. By default it adds 60 binary identity features:

- `TEAM_HOME_<TEAM_SLUG>_BEFORE`
- `TEAM_AWAY_<TEAM_SLUG>_BEFORE`

There is one home and one away column for each team in `TEAM_ID_MAP`.

If `create_df_to_predict(categorical_team_encoding=True)` is used, the pipeline
adds two pandas categorical columns instead:

- `TEAM_HOME_CATEGORY_BEFORE`
- `TEAM_AWAY_CATEGORY_BEFORE`

This mode is intended for models that can consume native categorical features,
such as XGBoost with categorical handling enabled.

## Game-Level Odds Features

Odds are used in several ways:

1. Canonical line, spread, and moneyline columns are merged before rolling stats.
2. Percentages and prices are merged before rolling stats so team-level market
   behavior can be rolled.
3. Remaining raw game-level odds are merged after the home/away merge.
4. `engineer_odds_features()` creates robust market summaries.

The raw Yahoo and Sportsbook Review data is merged in
`merge_yahoo_sportsbook_odds()`:

- Yahoo and Sportsbook Review are outer-joined by `game_id`.
- Missing BetMGM total, spread, and moneyline values are filled from Yahoo when
  available.
- American odds prices are converted to decimal odds.
- Yahoo public betting percentages are retained when available.

`engineer_odds_features()` creates `odds_*` features, including:

- Per-book total line midpoints.
- Cross-book total line mean, median, standard deviation, range, IQR, and MAD.
- Total-price log ratios: over price divided by under price.
- No-vig total probability differences.
- Total-market vig by book and aggregate vig summaries.
- Counts and ratios of books with total lines and total prices present.
- Spread home-line summaries, favorite absolute spread, and home-favorite flags.
- No-vig moneyline home probabilities and moneyline vig.
- Favorite win probability and moneyline probability gap.
- Interactions such as total line times probability skew, total line times
  favorite spread, and line disagreement times vig.
- Implied home and away points from total and spread.
- `close_total_consensus`, based on cross-book close total median or mean.

## Betting Difference Features

`add_betting_stats_differences()` creates home-minus-away differences for
team-specific betting and market-form columns.

It looks for home/away pairs that are both:

- Betting-related, such as total lines, spreads, moneylines, percentages, prices,
  consensus values, `DIFF_FROM_*`, and `TOTAL_POINTS`.
- Rolling or derived, such as last-game windows, weighted averages,
  season-before averages/stds, and trend slopes.

The resulting columns generally end in `_DIFF_BEFORE`. These features directly
encode relative home-vs-away market context.

## Global Market Regime Features

`add_global_market_features()` adds league-wide, game-date-level features. These
features are computed from games strictly before the current calendar date, so
same-day games cannot influence each other.

The function resolves the active close total line via
`resolve_main_total_line_col()` and compares it with actual `TOTAL_POINTS`.

Feature families include:

- Rolling global market bias: actual total minus close total.
- Rolling global MAE.
- Rolling global market error standard deviation.
- Median error and median absolute error.
- Tail miss rates for errors above 10, 15, and 20 points.
- Over, under, and push rates.
- League-wide close-total averages.
- League-wide actual-total averages.
- Actual-minus-close average gaps.
- Short-window versus long-window regime ratios.
- Bias, MAE, close-total, and actual-total acceleration features.
- League activity counts over recent days and games.
- Open-to-close move features when a consensus opener exists.
- Whether the close line has recently beaten the open line.
- Cross-book total-line disagreement for the current game and rolling league
  disagreement.

Column names are suffixed with `_BEFORE`, for example:

- `GLOBAL_MARKET_BIAS_30G_BEFORE`
- `GLOBAL_MARKET_MAE_7D_BEFORE`
- `GLOBAL_MARKET_OVER_RATE_75G_BEFORE`
- `GLOBAL_CLOSE_TOTAL_AVG_DIFF_15G_75G_BEFORE`
- `THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE`

## Referee Features

Referee processing is handled by
`add_referee_features_to_training_data()`.

Historical referee rows are transformed into one row per game with deterministic
`REF_1`, `REF_2`, and `REF_3` slots. Names are canonicalized and sorted so crew
order does not affect the features.

For each current game's referees, `compute_referee_features()` compares games
with a given referee to games without that referee, using same-season history
when possible and previous-season history as fallback.

Metrics used:

- `TOTAL_POINTS`
- `DIFF_FROM_LINE`
- `TOTAL_PF`

Features include:

- `REF_AVG_<METRIC>_DIFF_BEFORE`
- `REF_STD_<METRIC>_DIFF_BEFORE`
- `REF_SUM_<METRIC>_DIFF_BEFORE`

Exact trio features also exist in the implementation, but
`create_df_to_predict()` currently calls referee processing with
`include_ref_trio_features=False`.

For same-day prediction, scheduled referee assignments are appended before
feature computation. If a scheduled referee has no historical match in the
database, the pipeline raises an error rather than silently producing unknown
referee features.

## Availability Effect Features

After the initial player and injury features are created,
`add_top3_availability_effect_features_for_columns()` estimates how important
specific player availability has historically been for a team.

It is called twice:

1. For top active/home-away player columns, producing
   `TOP3_AVAILABILITY_EFFECT_*`.
2. For top injured/home-away player columns, producing
   `TOP3_INJURED_AVAILABILITY_EFFECT_*`.

For each player, team, and game date, the function looks at the current and
previous season for that team before the game date. It compares:

- Team `TOTAL_POINTS` when the player was present versus injured.
- Team `DIFF_FROM_LINE` when the player was present versus injured.

Raw effects are shrunk toward zero using:

```text
effect_shrunk = effect_raw * n_eff / (n_eff + k)
```

where `n_eff` is the smaller of injured-game count and present-game count, and
`k` defaults to `10.0`.

Aggregate outputs include:

- Home and away mean effects on total points.
- Home and away mean effects on difference from line.
- Home and away max absolute effects.
- Counts of injured, present, and total games used.
- Counts and flags for players with enough history to compute an effect.

## Travel And Schedule Features

`compute_travel_features()` converts the game-level table back into a team-game
travel log and uses city coordinates to estimate travel distance.

For each team, it identifies the city where the current game is played and the
city of the previous game. It then computes great-circle distance with the
Haversine formula. Two consecutive home games are treated as zero travel.

Rolling travel sums are computed over prior calendar windows:

- 1 day
- 2 days
- 5 days
- 7 days
- 14 days

The final game-level columns are:

- `TOTAL_KM_IN_LAST_<N>_DAYS_HOME_TEAM`
- `TOTAL_KM_IN_LAST_<N>_DAYS_AWAY_TEAM`

`create_df_to_predict()` calls this with `log_scale=True`, so the final values
are `log1p(kilometers)`.

`add_high_value_features_for_team_points()` later derives fatigue and rest
compression features such as:

- `TRAVEL_RECENCY_RATIO_HOME_2D_OVER_14D_BEFORE`
- `TRAVEL_RECENCY_RATIO_AWAY_2D_OVER_14D_BEFORE`
- `REST_DAYS_DIFF_HOME_MINUS_AWAY_BEFORE`

## Date And Calendar Features

`add_game_date_features()` adds simple calendar indicators:

- `IS_WEEKEND_BEFORE`
- `MONTH_BEFORE`
- `IS_US_HOLIDAY_BEFORE`

These are derived from `GAME_DATE`.

## Matchup And Team Context Features

The merged feature set includes several team and matchup context families:

- Team identity columns for home and away teams.
- Conference and division features:
  - `SAME_CONFERENCE_BEFORE`
  - `SAME_DIVISION_BEFORE`
  - `IS_HOME_WEST_CONFERENCE_BEFORE`
  - `IS_AWAY_WEST_CONFERENCE_BEFORE`
- Playoff context:
  - `IS_PLAYOFF_GAME_BEFORE`
  - `PLAYOFF_GAMES_LAST_SEASON_TEAM_HOME`
  - `PLAYOFF_GAMES_LAST_SEASON_TEAM_AWAY`
- Team record and rest:
  - `WINS_BEFORE_THIS_GAME`
  - `TEAM_RECORD_BEFORE_GAME`
  - `REST_DAYS_BEFORE_MATCH`
- Historical matchup features from
  `get_last_5_matchup_excluding_current()`.
- Points-conceded features from historical games, including differences between
  expected scoring and home/away conceded averages.

## Derived High-Value Features

`add_derived_features_after_computed_stats()` and
`add_high_value_features_for_team_points()` create compact interaction features
from the larger feature set.

Examples include:

- `TOTAL_PTS_SEASON_AVG_BEFORE`
- `TOTAL_PTS_LAST_GAMES_AVG_BEFORE`
- `BACK_TO_BACK_BEFORE`
- `DIFERENCE_HOME_OFF_AWAY_DEF_BEFORE`
- `DIFERENCE_AWAY_OFF_HOME_DEF_BEFORE`
- `IMPLIED_PTS_HOME_BEFORE`
- `IMPLIED_PTS_AWAY_BEFORE`
- `EXPECTED_POSS_FROM_PACE_BEFORE`
- `EXPECTED_PTS_HOME_FROM_OFFR_PACE_BEFORE`
- `EXPECTED_PTS_AWAY_FROM_OFFR_PACE_BEFORE`
- `OFFDEF_MISMATCH_HOME_OFF_MINUS_AWAY_DEF_BEFORE`
- `OFFDEF_MISMATCH_AWAY_OFF_MINUS_HOME_DEF_BEFORE`
- `PTS_FORM_Z_HOME_LAST5_VS_SEASON_BEFORE`
- `PTS_FORM_Z_AWAY_LAST5_VS_SEASON_BEFORE`
- `PTS_TREND_SLOPE_DIFF_HOME_MINUS_AWAY_BEFORE`
- `PTS_TREND_SLOPE_SUM_HOME_PLUS_AWAY_BEFORE`
- `INJURY_PTS_SHARE_HOME_BEFORE`
- `INJURY_PTS_SHARE_AWAY_BEFORE`
- `STAR_PTS_PCT_DIFF_HOME_MINUS_AWAY_BEFORE`
- `POSS_X_TSPCT_HOME_BEFORE`
- `POSS_X_TSPCT_AWAY_BEFORE`

These features are intentionally numeric and are designed to be useful for tree
models such as XGBoost.

## Final Column Selection

Final column selection is handled by `select_training_columns()`.

The selection rules intentionally favor features that are known before the game:

- Team metadata columns with `_TEAM_HOME` and `_TEAM_AWAY` suffixes are kept.
- Static game columns are kept.
- Any column containing `BEFORE` is kept.
- Team identity columns are kept because both the default one-hot columns and
  optional categorical columns include `_BEFORE`.
- All-star voting columns that survive the post-merge auxiliary-column drop are
  kept because they include `_BEFORE`.
- Known odds columns are kept.
- Columns beginning with `odds_` are kept.
- Columns beginning with `TOTAL_LINE_` are kept.
- `GAME_TIME` is kept only for same-day prediction mode.
- `TOTAL_POINTS` is kept as the target when present.

The function drops explicitly forbidden columns:

- `DIFFERENCE_FROM_LINE`
- `DIFF_FROM_LINE`
- `TOTAL_PF`
- `IS_OVER_LINE`

It also drops columns containing `DIFF_FROM` when they do not contain
`_BEFORE`, to avoid direct target leakage from same-game outcomes.

As a final safety check, if an original raw team column appears in the final
training DataFrame without `_BEFORE` and is not explicitly allowed, the function
raises a `ValueError`.

## Target

The primary target for total-points regression is:

- `TOTAL_POINTS`

It is the actual combined score for the game. The codebase also creates many
features based on differences from betting lines, but current final selection is
careful to retain only prior-game or otherwise pre-game versions of those values.

For same-day prediction rows, `TOTAL_POINTS` may be missing or unavailable for
future games. Downstream prediction code should treat it as absent target data
and use the feature columns only.

## Naming Conventions

Important conventions for future agents:

- `_BEFORE` means the feature is intended to use only information available
  before the game.
- `_TEAM_HOME` and `_TEAM_AWAY` identify side-specific columns after home/away
  merging.
- `_DIFF_BEFORE` usually means home-side feature minus away-side feature.
- `TEAM_HOME_<SLUG>_BEFORE` and `TEAM_AWAY_<SLUG>_BEFORE` are one-hot team
  identity features.
- `TEAM_HOME_CATEGORY_BEFORE` and `TEAM_AWAY_CATEGORY_BEFORE` are optional
  categorical team identity features.
- `ALL_STAR_*_BEFORE_TEAM_HOME` and `ALL_STAR_*_BEFORE_TEAM_AWAY` are
  side-specific All-Star fan-vote and score features.
- `TOTAL_LINE_<book>` is the canonical total line for a book.
- `SPREAD_<book>` and `MONEYLINE_<book>` are canonical selected-book team-side
  columns before the home/away merge.
- Raw game-level odds columns generally use lowercase prefixes like `total_`,
  `spread_`, and `ml_`.
- Odds-engineered features use the `odds_` prefix.

When adding new features, prefer using `_BEFORE` for any historical, rolling,
estimated, or pre-game-derived column that should be eligible for training
selection.

## Leakage Controls

The pipeline uses several controls to avoid target leakage:

- Rolling team features use `shift(1)` to exclude the current game.
- Player EWMA features use shifted prior appearances.
- Referee features use only games before the current game date.
- Global market features aggregate games from strictly earlier calendar dates.
- Injury availability effects use only games before the current game date.
- Same-day prediction uses a cutoff date before the scheduled games.
- Final selection drops raw `DIFF_FROM_*` columns unless they carry `_BEFORE`.
- Raw original boxscore columns are blocked from final selection unless they are
  static, explicitly allowed, or transformed into `_BEFORE` features.

## Adding Or Modifying Features

Use these guidelines when extending the training data:

1. Decide whether the feature is team-level or game-level.
   - Team-level features should usually be added before `merge_home_away_data()`.
   - Game-level features should usually be added after `merge_home_away_data()`.

2. Preserve pre-game semantics.
   - Rolling, cumulative, and historical aggregations should exclude the current
     game.
   - For same-day prediction, make sure scheduled rows do not use future
     boxscore results.

3. Follow naming conventions.
   - Add `_BEFORE` to any feature that should be selected automatically.
   - Use `_TEAM_HOME` and `_TEAM_AWAY` only after the home/away merge.
   - Use `odds_` for features created by odds engineering.

4. Check final selection.
   - If the feature is not selected, verify whether it contains `BEFORE`, starts
     with `odds_`, starts with `TOTAL_LINE_`, or is in the explicit odds/static
     allow lists.

5. Be careful with new sportsbook columns.
   - If the raw column follows existing naming conventions, many rolling and odds
     engineering functions can discover it dynamically.
   - If a book has a new naming pattern, update the relevant inference logic in
     odds merging, rolling stats, and odds engineering.

6. Validate historical and scheduled modes.
   - Historical training can pass even when scheduled mode breaks because
     scheduled mode needs compatible odds columns, scheduled referee matches, and
     injury-report mappings.

7. Validate all-star voting coverage.
   - The pipeline treats All-Star voting as a required source. If a new date range
     maps to a voting season that is not in the All-Star voting database, the
     training-data build will fail before the home/away merge.

8. Choose team encoding intentionally.
   - Default one-hot encoding is broad model compatible but adds 60 columns.
   - Categorical encoding is more compact, but downstream model training must
     preserve pandas categorical dtypes and enable native categorical support.

## Output Shape

The final DataFrame returned by `create_df_to_predict()` is game-level:

- One row per game.
- Side-specific home and away feature columns.
- Static game identifiers and dates.
- Pre-game features, mostly selected by `_BEFORE`.
- Team identity features, as one-hot columns by default or as categorical
  columns when requested.
- Side-specific All-Star fan-vote and score features.
- Raw and engineered odds features.
- `TOTAL_POINTS` target for completed historical games.

The exact number of columns changes as books, source schemas, and dynamic
feature discovery change.

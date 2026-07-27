from __future__ import annotations

from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.postgre_db.all_star_voting.create_db.create_all_star_voting_db import (
    create_all_star_voting_table,
)
from nba_ou.postgre_db.all_star_voting.process_all_star_voting_data import (
    OUTPUT_COLUMNS,
    prepare_all_star_voting_dataset,
)
from nba_ou.postgre_db.config.db_config import (
    connect_all_star_voting_db,
    get_schema_name_all_star_voting,
)


def _coerce_all_star_voting_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["season_year"] = pd.to_numeric(out["season_year"], errors="coerce").astype(
        "Int64"
    )
    out["player_id"] = out["player_id"].astype(str)
    out["fan_votes"] = pd.to_numeric(out["fan_votes"], errors="coerce").astype("Int64")

    rank_cols = ["fan_rank", "player_rank", "media_rank"]
    for col in rank_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    float_cols = ["fan_votes_pct", "score"]
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    required = [
        "conference",
        "position",
        "season",
        "season_year",
        "player_name",
        "player_id",
        "team_name",
    ]
    out = out.dropna(subset=required)
    return out.astype(object).where(pd.notna(out), None)


def upsert_all_star_voting_df(
    voting_df: pd.DataFrame,
    conn: psycopg.Connection | None = None,
) -> int:
    if voting_df.empty:
        return 0

    schema = get_schema_name_all_star_voting()
    table = schema
    close_conn = False
    if conn is None:
        conn = connect_all_star_voting_db()
        close_conn = True

    voting_df = _coerce_all_star_voting_df(voting_df)
    cols = OUTPUT_COLUMNS
    rows = [tuple(row) for row in voting_df[cols].itertuples(index=False, name=None)]

    insert_query = sql.SQL(
        """
        INSERT INTO {}.{} (
            {cols}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT (season_year, conference, position, player_id)
        DO UPDATE SET
            season = EXCLUDED.season,
            player_name = EXCLUDED.player_name,
            team_name = EXCLUDED.team_name,
            fan_votes = EXCLUDED.fan_votes,
            fan_votes_pct = EXCLUDED.fan_votes_pct,
            fan_rank = EXCLUDED.fan_rank,
            player_votes = EXCLUDED.player_votes,
            player_rank = EXCLUDED.player_rank,
            media_votes = EXCLUDED.media_votes,
            media_rank = EXCLUDED.media_rank,
            score = EXCLUDED.score
        """
    ).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in cols),
    )

    try:
        with conn.cursor() as cur:
            cur.executemany(insert_query, rows)
        conn.commit()
        return len(rows)
    finally:
        if close_conn:
            conn.close()


def build_and_upload_all_star_voting(
    input_csv: str | Path | None = None,
    *,
    drop_existing: bool = False,
    skip_unresolved: bool = False,
) -> dict[str, object]:
    if not create_all_star_voting_table(drop_existing=drop_existing):
        raise RuntimeError("Failed to create all-star voting table.")

    if input_csv is None:
        voting_df = prepare_all_star_voting_dataset(skip_unresolved=skip_unresolved)
    else:
        voting_df = prepare_all_star_voting_dataset(
            input_csv=Path(input_csv), skip_unresolved=skip_unresolved
        )

    inserted = upsert_all_star_voting_df(voting_df)
    return {
        "rows_prepared": len(voting_df),
        "rows_upserted": inserted,
        "season_years": sorted(voting_df["season_year"].unique().tolist()),
    }


if __name__ == "__main__":
    result = build_and_upload_all_star_voting(drop_existing=False)
    print(result)

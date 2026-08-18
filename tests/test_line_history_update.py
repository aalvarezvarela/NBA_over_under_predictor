"""Daily update planning: season resolution, refresh window, completeness.

The completeness rules are the fiddly part -- a book that launched mid-season
must not make every earlier game look partial, and Caesars (which SBR no longer
serves) must never make a game look partial at all.
"""

from datetime import date

import pandas as pd
import pytest
from nba_ou.postgre_db.line_history_aiven import update as up


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["A", "B", "C", "D"],
            "game_date": [
                date(2026, 1, 10),
                date(2026, 1, 11),
                date(2026, 1, 12),
                date(2025, 1, 12),
            ],
            "season_year": [2025, 2025, 2025, 2024],
        }
    )


class TestSeasonResolution:
    @pytest.mark.parametrize(
        "today,expected",
        [
            (date(2026, 1, 15), 2025),  # January belongs to the season that
            (date(2026, 6, 13), 2025),  # started the previous October
            (date(2026, 8, 18), 2025),  # offseason -> the season just finished
            (date(2026, 10, 21), 2026),  # opening night
        ],
    )
    def test_current_season_year(self, today, expected):
        assert up.current_season_year(today) == expected

    def test_defaults_to_the_current_season(self):
        # The scheduled job relies on this: no hardcoded year to update yearly.
        assert up.resolve_season_years(today=date(2026, 8, 18)) == [2025]

    def test_explicit_range(self):
        assert up.resolve_season_years(2021, 2024) == [2021, 2022, 2023, 2024]

    def test_single_season_from_either_end(self):
        assert up.resolve_season_years(2023, None) == [2023]
        assert up.resolve_season_years(None, 2023) == [2023]

    def test_reversed_range_is_rejected(self):
        with pytest.raises(ValueError):
            up.resolve_season_years(2024, 2021)


class TestRefreshWindow:
    def test_covers_recent_dates_that_had_games(self):
        dates = up.recent_dates(_games(), 2025, refresh_days=3, today=date(2026, 1, 12))
        assert dates == [date(2026, 1, 10), date(2026, 1, 11), date(2026, 1, 12)]

    def test_excludes_dates_outside_the_window(self):
        dates = up.recent_dates(_games(), 2025, refresh_days=1, today=date(2026, 1, 12))
        assert dates == [date(2026, 1, 11), date(2026, 1, 12)]

    def test_other_seasons_are_not_pulled_in(self):
        dates = up.recent_dates(_games(), 2024, refresh_days=3, today=date(2026, 1, 12))
        assert dates == []

    def test_offseason_window_is_empty(self):
        # Out of season there are no recent games, so a run gap-fills only.
        assert (
            up.recent_dates(_games(), 2025, refresh_days=3, today=date(2026, 8, 18))
            == []
        )

    def test_zero_disables_the_refresh(self):
        assert (
            up.recent_dates(_games(), 2025, refresh_days=0, today=date(2026, 1, 12))
            == []
        )


class TestDiscontinuedBooks:
    def test_caesars_is_not_expected_any_more(self):
        # SBR stopped serving it; expecting it would mean re-fetching those
        # games forever waiting for data that will never come.
        assert "caesars" in up.DISCONTINUED_BOOKS

    def test_current_books_are_still_expected(self):
        for slug in (
            "bet365",
            "betmgm",
            "draftkings",
            "fanduel",
            "fanatics_sportsbook",
        ):
            assert slug not in up.DISCONTINUED_BOOKS


class TestFindIncompleteGames:
    """The query compares each game to the best-covered game on its own date."""

    class _Conn:
        def __init__(self, books, rows):
            self._books, self._rows = books, rows
            self.excluded = None
            self.share = None

        def cursor(self):
            outer = self

            class _Cur:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def execute(self, query, params=None):
                    self._books_query = "lh_book" in str(query)
                    if not self._books_query:
                        outer.excluded = params[1]
                        outer.share = params[2]

                def fetchall(self):
                    return outer._books if self._books_query else outer._rows

            return _Cur()

    def test_discontinued_books_are_excluded_from_the_comparison(self):
        conn = self._Conn([(3, "caesars"), (6, "fanduel")], [])
        up.find_incomplete_games(conn, 2025)
        assert conn.excluded == [3]

    def test_no_discontinued_book_registered_still_filters_safely(self):
        # A sentinel keeps "<> ALL(...)" valid when the list would be empty.
        conn = self._Conn([(6, "fanduel")], [])
        up.find_incomplete_games(conn, 2025)
        assert conn.excluded == [-1]

    def test_returns_games_missing_an_expected_book(self):
        conn = self._Conn([(3, "caesars")], [("A", date(2026, 1, 10), 2)])
        out = up.find_incomplete_games(conn, 2025)
        assert out["game_id"].tolist() == ["A"]
        assert out.loc[0, "missing_books"] == 2

    def test_share_threshold_is_passed_to_the_query(self):
        # A book only counts as expected once it priced this share of the
        # date's games, which is what stops a launch day from flagging the
        # rest of the slate forever.
        conn = self._Conn([(3, "caesars")], [])
        up.find_incomplete_games(conn, 2025, min_share=0.75)
        assert conn.share == 0.75

    def test_default_share_is_a_majority(self):
        assert up.DEFAULT_EXPECTED_BOOK_SHARE == 0.5

    def test_fully_covered_season_returns_nothing(self):
        conn = self._Conn([(3, "caesars")], [])
        assert up.find_incomplete_games(conn, 2025).empty


class TestPlanUpdate:
    class _Conn:
        """Enough of a connection for plan_update's two queries."""

        def __init__(self, stored_ids, first_date, incomplete_rows):
            self._stored = stored_ids
            self._first = first_date
            self._incomplete = incomplete_rows

        def cursor(self):
            outer = self

            class _Cur:
                mode = ""

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def execute(self, query, params=None):
                    text = str(query)
                    if "lh_book" in text:
                        self.mode = "books"
                    elif "MIN(game_date)" in text:
                        self.mode = "range"
                    elif "game_books" in text:
                        self.mode = "incomplete"
                    else:
                        self.mode = "ids"

                def fetchall(self):
                    if self.mode == "books":
                        return [(3, "caesars")]
                    if self.mode == "incomplete":
                        return outer._incomplete
                    return [(g,) for g in outer._stored]

                def fetchone(self):
                    return (outer._first, None)

            return _Cur()

    def test_combines_recent_gap_and_partial_dates(self):
        conn = self._Conn(
            stored_ids={"A", "B"},
            first_date=date(2021, 10, 19),
            incomplete_rows=[("B", date(2026, 1, 11), 1)],
        )
        result = up.plan_update(
            conn, _games(), 2025, refresh_days=1, today=date(2026, 1, 12)
        )
        # C is absent from the store, so its date is a gap.
        assert date(2026, 1, 12) in result.gap_dates
        assert date(2026, 1, 11) in result.incomplete_dates
        assert result.target_dates == [
            date(2026, 1, 11),
            date(2026, 1, 12),
        ]

    def test_gap_dates_stay_within_the_target_season(self):
        conn = self._Conn(
            stored_ids=set(), first_date=date(2021, 10, 19), incomplete_rows=[]
        )
        result = up.plan_update(
            conn, _games(), 2025, refresh_days=0, today=date(2026, 1, 12)
        )
        # D belongs to 2024-25 and must not be dragged into a 2025-26 run.
        assert date(2025, 1, 12) not in result.gap_dates

    def test_incomplete_check_can_be_skipped(self):
        conn = self._Conn(
            stored_ids={"A", "B", "C"},
            first_date=date(2021, 10, 19),
            incomplete_rows=[("B", date(2026, 1, 11), 1)],
        )
        result = up.plan_update(
            conn,
            _games(),
            2025,
            refresh_days=0,
            include_incomplete=False,
            today=date(2026, 1, 12),
        )
        assert result.incomplete_dates == []
        assert result.target_dates == []

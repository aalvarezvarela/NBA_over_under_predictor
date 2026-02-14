# src/utils/patch_nba_api_http.py

from __future__ import annotations

from utils.nba_http_patched import PatchedNBAStatsHTTP


def patch_nba_api_http() -> None:
    """
    Monkey-patch nba_api endpoint modules so they use PatchedNBAStatsHTTP.
    Call this once at startup before creating any endpoints.
    """
    # Patch LeagueGameFinder module symbol
    import nba_api.stats.endpoints.leaguegamefinder as lgf_mod

    lgf_mod.NBAStatsHTTP = PatchedNBAStatsHTTP

    # If you use other endpoints, patch them too (examples):
    # import nba_api.stats.endpoints.leaguedashteamstats as t_mod
    # t_mod.NBAStatsHTTP = PatchedNBAStatsHTTP
    #
    # import nba_api.stats.endpoints.boxscoretraditionalv2 as b_mod
    # b_mod.NBAStatsHTTP = PatchedNBAStatsHTTP

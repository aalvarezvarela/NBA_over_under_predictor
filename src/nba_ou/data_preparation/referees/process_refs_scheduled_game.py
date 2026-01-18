from nba_ou.fetch_data.referees.fetch_refs_data import (
    fetch_nba_referee_assignments_today,
)
from nba_ou.config.settings import SETTINGS


df_teams = fetch_nba_referee_assignments_today(SETTINGS.nba_official_assignments_url)
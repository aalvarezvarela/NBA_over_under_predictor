"""
Compatibility wrapper for the refs+injuries update flow.

Delegates to the new module under nba_ou.postgre_db.injuries_refs.
"""

from nba_ou.postgre_db.injuries_refs.update_ref_injuries_database.update_refs_injuries_database import (
    update_refs_injuries_database,
)

__all__ = ["update_refs_injuries_database"]


if __name__ == "__main__":
    INJURY_DATA_FOLDER = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/injury_data/"
    )
    REF_DATA_FOLDER = "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/ref_data/"

    update_refs_injuries_database(
        injury_folder=INJURY_DATA_FOLDER, ref_folder=REF_DATA_FOLDER
    )

"""Schema versions for the generated training datasets.

Bumped whenever a regenerated CSV gains or loses columns, so a new build lands
beside the old one instead of overwriting it. Both training pipelines pin their
``expected_checksum`` against a specific file, and an in-place overwrite turns
that guard into a failure at the *next* run rather than a clean, dated artifact.

History
-------
``2_0``
    Totals only. ``TOTAL_POINTS`` is the sole outcome column; per-team final
    scores are dropped by the selection gates.

``2_1``
    Adds the spread market and moneyline data readiness:

    * ``PTS_TEAM_HOME`` / ``PTS_TEAM_AWAY`` / ``HOME_MARGIN`` survive selection
      (outcome columns, blocked from every feature matrix).
    * ``ODDS_SPREAD_LINE_HOME_<book>`` -- every book's spread normalised to the
      implied-home-margin convention.
    * Cross-book spread consensus features (median-based).
    * ``ODDS_ML_PRICE_HOME`` / ``_AWAY`` and de-vigged probabilities.
    * Intermediate dataset only: ``SPREAD_ERROR``, derived per snapshot against
      the Bet365 spread as of THAT snapshot.

    Totals columns and semantics are unchanged, so a 2_1 file trains the existing
    totals strategies identically to a 2_0 file built from the same games.

``2_2``
    Keeps the 2_1 columns but changes closing spread semantics to match totals:
    asymmetrically priced spread quotes are centered to their estimated
    -110/-110 equivalent, and implausibly extreme spread prices are nulled
    before centering. Intermediate spread snapshots already used centered
    ``norm_line`` values; 2_2 applies the same extreme-price guard there too.
"""

from __future__ import annotations

#: Current schema version for both generated training datasets.
TRAINING_DATA_SCHEMA_VERSION = "2_2"

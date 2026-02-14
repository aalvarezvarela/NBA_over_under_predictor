import numpy as np
import pandas as pd

BET_BOOKS = ["betmgm", "caesars", "fanduel", "draftkings", "fanatics_sportsbook"]


def as_float(s: pd.Series) -> pd.Series:
    """Centralized numeric coercion to float."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def decimal_to_prob(dec_odds: pd.Series) -> pd.Series:
    """
    Decimal odds (includes stake) -> implied probability.
    p = 1 / dec
    """
    d = as_float(dec_odds)
    out = np.where(d > 1.0, 1.0 / d, np.nan)  # decimal odds must be > 1
    return pd.Series(out, index=dec_odds.index)


def no_vig_two_way_prob(p_a: pd.Series, p_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Remove vig for a two-way market by normalization.
    """
    a = as_float(p_a)
    b = as_float(p_b)
    denom = a + b
    a_nv = np.where(denom > 0, a / denom, np.nan)
    b_nv = np.where(denom > 0, b / denom, np.nan)
    return pd.Series(a_nv, index=p_a.index), pd.Series(b_nv, index=p_b.index)


def safe_mean(x: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return x.mean(axis=1, skipna=True)


def safe_std(x: pd.DataFrame | pd.Series, ddof: int = 0) -> pd.Series:
    if isinstance(x, pd.Series):
        return pd.Series(np.nan, index=x.index)
    return x.std(axis=1, skipna=True, ddof=ddof)


def safe_range(x: pd.DataFrame) -> pd.Series:
    return x.max(axis=1, skipna=True) - x.min(axis=1, skipna=True)


def safe_iqr(x: pd.DataFrame) -> pd.Series:
    """Interquartile range (Q3 - Q1) for robust dispersion measure."""
    if isinstance(x, pd.Series):
        return pd.Series(np.nan, index=x.index)
    q25 = x.quantile(0.25, axis=1)
    q75 = x.quantile(0.75, axis=1)
    return q75 - q25


def safe_max_abs(x: pd.DataFrame) -> pd.Series:
    """Maximum absolute value across columns."""
    if isinstance(x, pd.Series):
        return x.abs()
    return x.abs().max(axis=1, skipna=True)


def engineer_odds_features(
    df: pd.DataFrame,
    prefix: str = "odds_",
    books: list[str] | None = None,
) -> pd.DataFrame:
    """
    Odds feature engineering focused on TOTAL lines and TOTAL prices, assuming prices are ALREADY decimal odds.

    Primary features:
      - per-book total line mid and move from opener (prefer per-book opener; fallback to consensus opener if present)
      - aggregates across books: mean/std/range of current total lines (excluding openers), mean/std of moves
      - per-book total price asymmetry: over/under decimal diff, implied prob diff (no-vig), vig
      - aggregates across books: mean/std/range of no-vig prob(over), bias from 0.50

    Keeps a small set of spread and moneyline features that are plausibly predictive of total points.

    Expected (if available):
      - Total current lines/prices per book:
          total_{book}_line_over, total_{book}_line_under
          total_{book}_price_over, total_{book}_price_under   (DECIMAL)
      - Total opener per book (optional, preferred):
          total_{book}_opener_line_over, total_{book}_opener_line_under
          total_{book}_opener_price_over, total_{book}_opener_price_under   (DECIMAL)
      - Consensus total opener (optional fallback):
          total_consensus_opener_line_over, total_consensus_opener_line_under
          total_consensus_opener_price_over, total_consensus_opener_price_under   (DECIMAL)
      - Spread per book (optional):
          spread_{book}_line_home, spread_{book}_line_away
          spread_consensus_opener_line_home, spread_consensus_opener_line_away   (optional)
      - Moneyline per book (optional, DECIMAL):
          ml_{book}_price_home, ml_{book}_price_away
    """
    out = df.copy()
    books = books or BET_BOOKS  # uses your existing constant

    # -----------------------------
    # 0) Consensus total opener (fallback)
    # -----------------------------
    cons_open_over = "total_consensus_opener_line_over"
    cons_open_under = "total_consensus_opener_line_under"
    cons_open_price_over = "total_consensus_opener_price_over"
    cons_open_price_under = "total_consensus_opener_price_under"

    if cons_open_over in out.columns and cons_open_under in out.columns:
        tmp = (
            out[[cons_open_over, cons_open_under]]
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        out[f"{prefix}consensus_total_opener_line_mid"] = safe_mean(tmp)

    if cons_open_price_over in out.columns and cons_open_price_under in out.columns:
        p_over = decimal_to_prob(out[cons_open_price_over])
        p_under = decimal_to_prob(out[cons_open_price_under])
        p_over_nv, p_under_nv = no_vig_two_way_prob(p_over, p_under)
        out[f"{prefix}consensus_total_opener_prob_over_novig"] = p_over_nv
        out[f"{prefix}consensus_total_opener_prob_under_novig"] = p_under_nv
        out[f"{prefix}consensus_total_opener_vig"] = (p_over + p_under) - 1.0
        out[f"{prefix}consensus_total_opener_prob_diff_novig"] = (
            p_over_nv - p_under_nv
        )  # in [-1, 1]

    # -----------------------------
    # 1) TOTAL lines and TOTAL prices (per book) + movements from opener
    # -----------------------------
    total_line_mids = []
    total_line_moves = []
    total_prob_over_novig = []
    total_prob_diff_novig = []
    total_vigs = []
    total_price_log_ratios = []

    for book in books:
        # Current
        line_over = f"total_{book}_line_over"
        line_under = f"total_{book}_line_under"
        price_over = f"total_{book}_price_over"
        price_under = f"total_{book}_price_under"

        has_lines = line_over in out.columns and line_under in out.columns
        has_prices = price_over in out.columns and price_under in out.columns

        # Opener (preferred per book)
        open_line_over = f"total_{book}_opener_line_over"
        open_line_under = f"total_{book}_opener_line_under"
        open_price_over = f"total_{book}_opener_price_over"
        open_price_under = f"total_{book}_opener_price_under"

        has_open_lines = (
            open_line_over in out.columns and open_line_under in out.columns
        )
        has_open_prices = (
            open_price_over in out.columns and open_price_under in out.columns
        )

        # Current total line mid
        if has_lines:
            col_mid = f"{prefix}book_total_line_mid_{book}"
            tmp = (
                out[[line_over, line_under]]
                .apply(pd.to_numeric, errors="coerce")
                .astype(float)
            )
            out[col_mid] = safe_mean(tmp)
            total_line_mids.append(col_mid)

        # Current total price asymmetry (decimal and prob space)
        if has_prices:
            dec_over = as_float(out[price_over])
            dec_under = as_float(out[price_under])

            out[f"{prefix}book_total_price_diff_over_minus_under_{book}"] = (
                dec_over - dec_under
            )
            # Use log ratio for stability
            col_log_ratio = f"{prefix}book_total_price_log_ratio_over_div_under_{book}"
            with np.errstate(divide="ignore", invalid="ignore"):
                out[col_log_ratio] = np.log(dec_over / dec_under)
            out[col_log_ratio] = out[col_log_ratio].replace([np.inf, -np.inf], np.nan)
            total_price_log_ratios.append(col_log_ratio)

            p_over = decimal_to_prob(out[price_over])
            p_under = decimal_to_prob(out[price_under])
            p_over_nv, p_under_nv = no_vig_two_way_prob(p_over, p_under)

            col_p_over = f"{prefix}book_total_prob_over_novig_{book}"
            col_p_under = f"{prefix}book_total_prob_under_novig_{book}"
            col_p_diff = f"{prefix}book_total_prob_diff_novig_{book}"  # 2*p_over_nv - 1
            col_vig = f"{prefix}book_total_vig_{book}"

            out[col_p_over] = p_over_nv
            out[col_p_under] = p_under_nv
            out[col_p_diff] = p_over_nv - p_under_nv
            out[col_vig] = (p_over + p_under) - 1.0

            total_prob_over_novig.append(col_p_over)
            total_prob_diff_novig.append(col_p_diff)
            total_vigs.append(col_vig)

        # Movement from opener (one per house if available, else fallback to consensus opener)
        if has_lines:
            opener_mid_col = None

            if has_open_lines:
                opener_mid_col = f"{prefix}book_total_opener_line_mid_{book}"
                tmp = (
                    out[[open_line_over, open_line_under]]
                    .apply(pd.to_numeric, errors="coerce")
                    .astype(float)
                )
                out[opener_mid_col] = safe_mean(tmp)
            elif f"{prefix}consensus_total_opener_line_mid" in out.columns:
                opener_mid_col = f"{prefix}consensus_total_opener_line_mid"

            if opener_mid_col is not None:
                move_col = f"{prefix}book_total_line_move_from_opener_{book}"
                out[move_col] = (
                    out[f"{prefix}book_total_line_mid_{book}"] - out[opener_mid_col]
                )
                total_line_moves.append(move_col)

        if has_prices:
            opener_prob_over_col = None

            if has_open_prices:
                p_over_o = decimal_to_prob(out[open_price_over])
                p_under_o = decimal_to_prob(out[open_price_under])
                p_over_o_nv, p_under_o_nv = no_vig_two_way_prob(p_over_o, p_under_o)
                opener_prob_over_col = (
                    f"{prefix}book_total_opener_prob_over_novig_{book}"
                )
                out[opener_prob_over_col] = p_over_o_nv
            elif f"{prefix}consensus_total_opener_prob_over_novig" in out.columns:
                opener_prob_over_col = f"{prefix}consensus_total_opener_prob_over_novig"

            if opener_prob_over_col is not None:
                out[f"{prefix}book_total_prob_move_over_novig_{book}"] = (
                    out[f"{prefix}book_total_prob_over_novig_{book}"]
                    - out[opener_prob_over_col]
                )

    # Aggregates across books: current total line levels (excluding opener by construction)
    if total_line_mids:
        tmp_lines = (
            out[total_line_mids].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        out[f"{prefix}total_line_books_mean"] = safe_mean(tmp_lines)
        out[f"{prefix}total_line_books_std"] = safe_std(tmp_lines)
        out[f"{prefix}total_line_books_range"] = safe_range(tmp_lines)
        out[f"{prefix}total_line_books_iqr"] = safe_iqr(tmp_lines)

    # Aggregates across books: movement from opener (house opener if available, else consensus fallback)
    if total_line_moves:
        tmp_moves = (
            out[total_line_moves].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        out[f"{prefix}total_line_move_books_mean"] = safe_mean(tmp_moves)
        out[f"{prefix}total_line_move_books_std"] = safe_std(tmp_moves)
        out[f"{prefix}total_line_move_books_abs_mean"] = safe_mean(tmp_moves.abs())
        out[f"{prefix}total_line_move_books_max_abs"] = safe_max_abs(tmp_moves)

    # Aggregates across books: probability and price skew
    if total_prob_over_novig:
        tmp_p = (
            out[total_prob_over_novig]
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        out[f"{prefix}total_prob_over_books_mean"] = safe_mean(tmp_p)
        out[f"{prefix}total_prob_over_books_std"] = safe_std(tmp_p)
        out[f"{prefix}total_prob_over_books_range"] = safe_range(tmp_p)
        out[f"{prefix}total_prob_over_books_bias_from_50"] = (
            out[f"{prefix}total_prob_over_books_mean"] - 0.5
        )

    if total_prob_diff_novig:
        tmp_d = (
            out[total_prob_diff_novig]
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        out[f"{prefix}total_prob_diff_books_mean"] = safe_mean(tmp_d)
        out[f"{prefix}total_prob_diff_books_std"] = safe_std(tmp_d)

    if total_vigs:
        tmp_v = out[total_vigs].apply(pd.to_numeric, errors="coerce").astype(float)
        out[f"{prefix}total_vig_books_mean"] = safe_mean(tmp_v)
        out[f"{prefix}total_vig_books_std"] = safe_std(tmp_v)

    # Aggregates across books: log-ratio (market skew)
    if total_price_log_ratios:
        tmp_lr = (
            out[total_price_log_ratios]
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        out[f"{prefix}total_price_log_ratio_books_mean"] = safe_mean(tmp_lr)
        out[f"{prefix}total_price_log_ratio_books_std"] = safe_std(tmp_lr)
        out[f"{prefix}total_price_log_ratio_books_range"] = safe_range(tmp_lr)

    # -----------------------------
    # 2) SPREAD features using FAVORITE logic (fixed calculation)
    # -----------------------------
    spread_home_lines = []
    spread_fav_abs = []
    spread_home_is_fav = []
    spread_fav_abs_moves = []

    # Consensus spread opener (optional) - use favorite logic
    if (
        "spread_consensus_opener_line_home" in out.columns
        and "spread_consensus_opener_line_away" in out.columns
    ):
        h_open = as_float(out["spread_consensus_opener_line_home"])
        a_open = as_float(out["spread_consensus_opener_line_away"])
        # Favorite is the more negative line (keep as Series for safety)
        fav_open = pd.Series(np.minimum(h_open, a_open), index=out.index)
        out[f"{prefix}spread_consensus_opener_fav_line"] = fav_open
        out[f"{prefix}spread_consensus_opener_fav_abs"] = np.abs(fav_open)
        out[f"{prefix}spread_consensus_opener_home_is_fav"] = (h_open < a_open).astype(
            float
        )

    for book in books:
        l_home = f"spread_{book}_line_home"
        l_away = f"spread_{book}_line_away"
        if l_home in out.columns and l_away in out.columns:
            h_line = as_float(out[l_home])
            a_line = as_float(out[l_away])

            # Keep home line (encodes direction and magnitude)
            col_home = f"{prefix}spread_home_line_{book}"
            out[col_home] = h_line
            spread_home_lines.append(col_home)

            # Favorite line (more negative, keep as Series)
            fav_line = pd.Series(np.minimum(h_line, a_line), index=out.index)
            col_fav_abs = f"{prefix}spread_fav_abs_{book}"
            out[col_fav_abs] = np.abs(fav_line)
            spread_fav_abs.append(col_fav_abs)

            # Home is favorite indicator
            col_home_fav = f"{prefix}spread_home_is_fav_{book}"
            out[col_home_fav] = (h_line < a_line).astype(float)
            spread_home_is_fav.append(col_home_fav)

            # Movement from opener (using fav_abs)
            if f"{prefix}spread_consensus_opener_fav_abs" in out.columns:
                mv = f"{prefix}spread_fav_abs_move_{book}"
                out[mv] = (
                    out[col_fav_abs] - out[f"{prefix}spread_consensus_opener_fav_abs"]
                )
                spread_fav_abs_moves.append(mv)

    # Aggregates across books
    if spread_home_lines:
        tmp = out[spread_home_lines].apply(pd.to_numeric, errors="coerce").astype(float)
        out[f"{prefix}spread_home_books_mean"] = safe_mean(tmp)
        out[f"{prefix}spread_home_books_std"] = safe_std(tmp)

    if spread_fav_abs:
        tmp = out[spread_fav_abs].apply(pd.to_numeric, errors="coerce").astype(float)
        out[f"{prefix}spread_fav_abs_books_mean"] = safe_mean(tmp)
        out[f"{prefix}spread_fav_abs_books_std"] = safe_std(tmp)
        out[f"{prefix}spread_fav_abs_books_range"] = safe_range(tmp)

    if spread_home_is_fav:
        tmp = (
            out[spread_home_is_fav].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        out[f"{prefix}spread_home_is_fav_books_mean"] = safe_mean(tmp)

    if spread_fav_abs_moves:
        tmp = (
            out[spread_fav_abs_moves]
            .apply(pd.to_numeric, errors="coerce")
            .astype(float)
        )
        out[f"{prefix}spread_fav_abs_move_books_mean"] = safe_mean(tmp)
        out[f"{prefix}spread_fav_abs_move_books_abs_mean"] = safe_mean(tmp.abs())
        out[f"{prefix}spread_fav_abs_move_books_max_abs"] = safe_max_abs(tmp)

    # -----------------------------
    # 3) Keep a minimal set of MONEYLINE features (implied win prob, no vig)
    # -----------------------------
    ml_home_probs = []
    ml_away_probs = []
    ml_vigs = []

    for book in books:
        p_home = f"ml_{book}_price_home"
        p_away = f"ml_{book}_price_away"

        if p_home in out.columns and p_away in out.columns:
            ph = decimal_to_prob(out[p_home])
            pa = decimal_to_prob(out[p_away])
            ph_nv, pa_nv = no_vig_two_way_prob(ph, pa)

            col_home = f"{prefix}ml_home_prob_novig_{book}"
            col_away = f"{prefix}ml_away_prob_novig_{book}"
            col_vig = f"{prefix}ml_vig_{book}"

            out[col_home] = ph_nv
            out[col_away] = pa_nv
            out[col_vig] = (ph + pa) - 1.0

            ml_home_probs.append(col_home)
            ml_away_probs.append(col_away)
            ml_vigs.append(col_vig)

    if ml_home_probs and ml_away_probs:
        out[f"{prefix}ml_home_prob_books_mean"] = safe_mean(
            out[ml_home_probs].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        out[f"{prefix}ml_away_prob_books_mean"] = safe_mean(
            out[ml_away_probs].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        out[f"{prefix}ml_favorite_prob"] = out[
            [f"{prefix}ml_home_prob_books_mean", f"{prefix}ml_away_prob_books_mean"]
        ].max(axis=1, skipna=True)
        out[f"{prefix}ml_win_prob_gap"] = (
            out[f"{prefix}ml_home_prob_books_mean"]
            - out[f"{prefix}ml_away_prob_books_mean"]
        ).abs()

    if ml_vigs:
        out[f"{prefix}ml_vig_books_mean"] = safe_mean(
            out[ml_vigs].apply(pd.to_numeric, errors="coerce").astype(float)
        )

    # -----------------------------
    # 4) Interactions and derived features for predicting line error
    # -----------------------------

    # Basic interactions
    if (
        f"{prefix}total_line_books_mean" in out.columns
        and f"{prefix}total_prob_over_books_bias_from_50" in out.columns
    ):
        out[f"{prefix}total_line_x_prob_bias"] = (
            out[f"{prefix}total_line_books_mean"]
            * out[f"{prefix}total_prob_over_books_bias_from_50"]
        )

    # Use corrected spread_fav_abs instead of spread_mid
    if (
        f"{prefix}total_line_books_mean" in out.columns
        and f"{prefix}spread_fav_abs_books_mean" in out.columns
    ):
        out[f"{prefix}total_line_x_spread_fav_abs"] = (
            out[f"{prefix}total_line_books_mean"]
            * out[f"{prefix}spread_fav_abs_books_mean"]
        )

    if (
        f"{prefix}total_line_move_books_abs_mean" in out.columns
        and f"{prefix}ml_win_prob_gap" in out.columns
    ):
        out[f"{prefix}move_abs_x_ml_gap"] = (
            out[f"{prefix}total_line_move_books_abs_mean"]
            * out[f"{prefix}ml_win_prob_gap"]
        )

    # Disagreement + vig interaction (uncertainty proxy)
    if (
        f"{prefix}total_line_books_std" in out.columns
        and f"{prefix}total_vig_books_mean" in out.columns
    ):
        out[f"{prefix}line_std_x_vig"] = (
            out[f"{prefix}total_line_books_std"] * out[f"{prefix}total_vig_books_mean"]
        )

    # Residual skew after line movement
    if (
        f"{prefix}total_line_move_books_abs_mean" in out.columns
        and f"{prefix}total_prob_diff_books_mean" in out.columns
    ):
        out[f"{prefix}move_abs_x_prob_diff_abs"] = (
            out[f"{prefix}total_line_move_books_abs_mean"]
            * out[f"{prefix}total_prob_diff_books_mean"].abs()
        )

    # -----------------------------
    # 5) Implied team totals from total and spread
    # -----------------------------
    # From consensus total T and representative spread S (home line, signed):
    # implied_home_points = T/2 - S/2
    # implied_away_points = T - implied_home
    if (
        f"{prefix}total_line_books_mean" in out.columns
        and f"{prefix}spread_home_books_mean" in out.columns
    ):
        total_line = out[f"{prefix}total_line_books_mean"]
        spread_home = out[f"{prefix}spread_home_books_mean"]

        # Home spread is negative when home is favored
        # implied_home = T/2 - S/2 (when S is negative, home gets more points)
        out[f"{prefix}implied_home_points"] = total_line / 2.0 - spread_home / 2.0
        out[f"{prefix}implied_away_points"] = (
            total_line - out[f"{prefix}implied_home_points"]
        )

        # Ratio of implied totals (useful for lopsided matchups)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{prefix}implied_points_ratio"] = (
                out[f"{prefix}implied_home_points"]
                / out[f"{prefix}implied_away_points"]
            )
        out[f"{prefix}implied_points_ratio"] = out[
            f"{prefix}implied_points_ratio"
        ].replace([np.inf, -np.inf], np.nan)
        # Max/min implied points
        out[f"{prefix}implied_points_max"] = out[
            [f"{prefix}implied_home_points", f"{prefix}implied_away_points"]
        ].max(axis=1)
        out[f"{prefix}implied_points_min"] = out[
            [f"{prefix}implied_home_points", f"{prefix}implied_away_points"]
        ].min(axis=1)
        out[f"{prefix}implied_points_gap"] = (
            out[f"{prefix}implied_points_max"] - out[f"{prefix}implied_points_min"]
        )

    return out

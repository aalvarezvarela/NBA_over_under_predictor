from __future__ import annotations

import numpy as np
import pandas as pd


def american_to_prob(price: pd.Series) -> pd.Series:
    """Convert American odds to implied probability (with vig)."""
    p = price.astype(float)
    out = pd.Series(np.nan, index=p.index, dtype="float64")

    pos = p > 0
    neg = p < 0

    out.loc[pos] = 100.0 / (p.loc[pos] + 100.0)
    out.loc[neg] = (-p.loc[neg]) / ((-p.loc[neg]) + 100.0)

    return out


def no_vig_two_way_prob(p_a: pd.Series, p_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Remove vig for a 2-way market via proportional normalization."""
    denom = p_a + p_b
    return p_a / denom, p_b / denom


def safe_mean(df: pd.DataFrame) -> pd.Series:
    return df.mean(axis=1, skipna=True)


def safe_std(df: pd.DataFrame) -> pd.Series:
    return df.std(axis=1, skipna=True)


def safe_range(df: pd.DataFrame) -> pd.Series:
    return df.max(axis=1, skipna=True) - df.min(axis=1, skipna=True)


def engineer_odds_features(df: pd.DataFrame, prefix: str = "odds_") -> pd.DataFrame:
    """
    Create odds features for predicting NBA TOTAL_POINTS (regression).
    All newly created columns are prefixed with `prefix` (default: "odds_").
    Does NOT require TOTAL_POINTS.

    The function is defensive: it only creates features when required columns exist.
    """
    out = df.copy()

    books_total = ["betmgm", "caesars", "fanduel", "draftkings"]
    books_spread = ["betmgm", "caesars", "fanduel", "draftkings"]
    books_ml = ["betmgm", "caesars", "fanduel", "draftkings"]

    # -----------------------------
    # 1) Consensus opener (total)
    # -----------------------------
    opener_line_over = "total_consensus_opener_line_over"
    opener_line_under = "total_consensus_opener_line_under"
    opener_price_over = "total_consensus_opener_price_over"
    opener_price_under = "total_consensus_opener_price_under"

    if opener_line_over in out.columns and opener_line_under in out.columns:
        out[f"{prefix}consensus_total_opener_line_mid"] = safe_mean(
            out[[opener_line_over, opener_line_under]]
        )

    if opener_price_over in out.columns and opener_price_under in out.columns:
        p_over = american_to_prob(out[opener_price_over])
        p_under = american_to_prob(out[opener_price_under])
        p_over_nv, p_under_nv = no_vig_two_way_prob(p_over, p_under)

        out[f"{prefix}consensus_total_opener_prob_over_novig"] = p_over_nv
        out[f"{prefix}consensus_total_opener_prob_under_novig"] = p_under_nv
        out[f"{prefix}consensus_total_opener_vig"] = (p_over + p_under) - 1.0

    # -----------------------------
    # 2) Yahoo public skew features
    # -----------------------------
    def add_skew(col_a: str, col_b: str, new_name: str) -> None:
        if col_a in out.columns and col_b in out.columns:
            out[f"{prefix}{new_name}"] = out[col_a].astype(float) - out[col_b].astype(
                float
            )

    # Total: bets and money skew, plus sharpness proxy
    add_skew("total_pct_bets_over", "total_pct_bets_under", "total_bets_skew")
    add_skew("total_pct_money_over", "total_pct_money_under", "total_money_skew")
    if (
        f"{prefix}total_money_skew" in out.columns
        and f"{prefix}total_bets_skew" in out.columns
    ):
        out[f"{prefix}total_sharp_proxy"] = (
            out[f"{prefix}total_money_skew"] - out[f"{prefix}total_bets_skew"]
        )

    # Total: consensus vs public (over side only)
    if (
        "total_consensus_pct_over" in out.columns
        and "total_pct_bets_over" in out.columns
    ):
        out[f"{prefix}total_consensus_vs_bets_over"] = out[
            "total_consensus_pct_over"
        ].astype(float) - out["total_pct_bets_over"].astype(float)
    if (
        "total_consensus_pct_over" in out.columns
        and "total_pct_money_over" in out.columns
    ):
        out[f"{prefix}total_consensus_vs_money_over"] = out[
            "total_consensus_pct_over"
        ].astype(float) - out["total_pct_money_over"].astype(float)

    # Spread: bets and money skew (home minus away)
    add_skew(
        "spread_pct_bets_home",
        "spread_pct_bets_away",
        "spread_bets_skew_home_minus_away",
    )
    add_skew(
        "spread_pct_money_home",
        "spread_pct_money_away",
        "spread_money_skew_home_minus_away",
    )
    if (
        f"{prefix}spread_money_skew_home_minus_away" in out.columns
        and f"{prefix}spread_bets_skew_home_minus_away" in out.columns
    ):
        out[f"{prefix}spread_sharp_proxy"] = (
            out[f"{prefix}spread_money_skew_home_minus_away"]
            - out[f"{prefix}spread_bets_skew_home_minus_away"]
        )

    # Moneyline: bets and money skew (home minus away)
    add_skew(
        "moneyline_pct_bets_home",
        "moneyline_pct_bets_away",
        "ml_bets_skew_home_minus_away",
    )
    add_skew(
        "moneyline_pct_money_home",
        "moneyline_pct_money_away",
        "ml_money_skew_home_minus_away",
    )
    if (
        f"{prefix}ml_money_skew_home_minus_away" in out.columns
        and f"{prefix}ml_bets_skew_home_minus_away" in out.columns
    ):
        out[f"{prefix}ml_sharp_proxy"] = (
            out[f"{prefix}ml_money_skew_home_minus_away"]
            - out[f"{prefix}ml_bets_skew_home_minus_away"]
        )

    # -----------------------------
    # 3) Total book features
    # -----------------------------
    total_line_mids = []
    total_prob_overs = []

    for book in books_total:
        line_over = f"total_{book}_line_over"
        line_under = f"total_{book}_line_under"
        price_over = f"total_{book}_price_over"
        price_under = f"total_{book}_price_under"

        has_lines = line_over in out.columns and line_under in out.columns
        has_prices = price_over in out.columns and price_under in out.columns

        if has_lines:
            col_mid = f"{prefix}book_total_line_mid_{book}"
            out[col_mid] = safe_mean(out[[line_over, line_under]]).astype(float)
            total_line_mids.append(col_mid)

        if has_prices:
            p_over = american_to_prob(out[price_over])
            p_under = american_to_prob(out[price_under])
            p_over_nv, p_under_nv = no_vig_two_way_prob(p_over, p_under)

            col_p_over = f"{prefix}book_total_prob_over_novig_{book}"
            col_p_under = f"{prefix}book_total_prob_under_novig_{book}"
            out[col_p_over] = p_over_nv
            out[col_p_under] = p_under_nv
            out[f"{prefix}book_total_vig_{book}"] = (p_over + p_under) - 1.0

            total_prob_overs.append(col_p_over)

        # Movement relative to consensus opener
        if f"{prefix}consensus_total_opener_line_mid" in out.columns and has_lines:
            out[f"{prefix}total_line_move_{book}"] = (
                out[f"{prefix}book_total_line_mid_{book}"]
                - out[f"{prefix}consensus_total_opener_line_mid"]
            )

        if (
            f"{prefix}consensus_total_opener_prob_over_novig" in out.columns
            and has_prices
        ):
            out[f"{prefix}total_prob_move_{book}"] = (
                out[f"{prefix}book_total_prob_over_novig_{book}"]
                - out[f"{prefix}consensus_total_opener_prob_over_novig"]
            )

    if total_line_mids:
        tmp = out[total_line_mids].astype(float)
        out[f"{prefix}total_line_books_mean"] = safe_mean(tmp)
        out[f"{prefix}total_line_books_std"] = safe_std(tmp)
        out[f"{prefix}total_line_books_range"] = safe_range(tmp)

        if f"{prefix}consensus_total_opener_line_mid" in out.columns:
            out[f"{prefix}total_line_books_mean_minus_opener"] = (
                out[f"{prefix}total_line_books_mean"]
                - out[f"{prefix}consensus_total_opener_line_mid"]
            )

    if total_prob_overs:
        tmp = out[total_prob_overs].astype(float)
        out[f"{prefix}total_prob_over_books_mean"] = safe_mean(tmp)
        out[f"{prefix}total_prob_over_books_std"] = safe_std(tmp)
        out[f"{prefix}total_prob_over_books_range"] = safe_range(tmp)
        out[f"{prefix}total_prob_over_books_bias_from_50"] = (
            out[f"{prefix}total_prob_over_books_mean"] - 0.5
        )

    # -----------------------------
    # 4) Spread features (across books)
    # -----------------------------
    if (
        "spread_consensus_opener_line_home" in out.columns
        and "spread_consensus_opener_line_away" in out.columns
    ):
        out[f"{prefix}spread_consensus_opener_mid"] = safe_mean(
            out[
                [
                    "spread_consensus_opener_line_home",
                    "spread_consensus_opener_line_away",
                ]
            ]
        ).astype(float)
        out[f"{prefix}spread_consensus_opener_abs"] = out[
            f"{prefix}spread_consensus_opener_mid"
        ].abs()

    spread_mids = []
    spread_abs = []

    for book in books_spread:
        l_home = f"spread_{book}_line_home"
        l_away = f"spread_{book}_line_away"

        if l_home in out.columns and l_away in out.columns:
            col_mid = f"{prefix}spread_mid_{book}"
            out[col_mid] = safe_mean(out[[l_home, l_away]]).astype(float)
            spread_mids.append(col_mid)

            col_abs = f"{prefix}spread_abs_{book}"
            out[col_abs] = out[col_mid].abs()
            spread_abs.append(col_abs)

            if f"{prefix}spread_consensus_opener_mid" in out.columns:
                out[f"{prefix}spread_move_{book}"] = (
                    out[col_mid] - out[f"{prefix}spread_consensus_opener_mid"]
                )

    if spread_mids:
        tmp = out[spread_mids].astype(float)
        out[f"{prefix}spread_books_mean"] = safe_mean(tmp)
        out[f"{prefix}spread_books_std"] = safe_std(tmp)
        out[f"{prefix}spread_books_range"] = safe_range(tmp)

    if spread_abs:
        tmp = out[spread_abs].astype(float)
        out[f"{prefix}spread_books_abs_mean"] = safe_mean(tmp)
        out[f"{prefix}spread_books_abs_std"] = safe_std(tmp)

    # -----------------------------
    # 5) Moneyline features (implied win probability, no vig)
    # -----------------------------
    ml_home_probs = []
    ml_away_probs = []

    for book in books_ml:
        p_home = f"ml_{book}_price_home"
        p_away = f"ml_{book}_price_away"

        if p_home in out.columns and p_away in out.columns:
            ph = american_to_prob(out[p_home])
            pa = american_to_prob(out[p_away])
            ph_nv, pa_nv = no_vig_two_way_prob(ph, pa)

            col_home = f"{prefix}ml_home_prob_novig_{book}"
            col_away = f"{prefix}ml_away_prob_novig_{book}"

            out[col_home] = ph_nv
            out[col_away] = pa_nv
            out[f"{prefix}ml_vig_{book}"] = (ph + pa) - 1.0

            ml_home_probs.append(col_home)
            ml_away_probs.append(col_away)

    if ml_home_probs and ml_away_probs:
        out[f"{prefix}ml_home_prob_books_mean"] = safe_mean(
            out[ml_home_probs].astype(float)
        )
        out[f"{prefix}ml_away_prob_books_mean"] = safe_mean(
            out[ml_away_probs].astype(float)
        )
        out[f"{prefix}ml_favorite_prob"] = out[
            [f"{prefix}ml_home_prob_books_mean", f"{prefix}ml_away_prob_books_mean"]
        ].max(axis=1, skipna=True)
        out[f"{prefix}ml_win_prob_gap"] = (
            out[f"{prefix}ml_home_prob_books_mean"]
            - out[f"{prefix}ml_away_prob_books_mean"]
        ).abs()

    # -----------------------------
    # 6) Simple interaction features
    # -----------------------------
    if (
        f"{prefix}total_line_books_mean" in out.columns
        and f"{prefix}total_prob_over_books_bias_from_50" in out.columns
    ):
        out[f"{prefix}total_line_x_prob_bias"] = (
            out[f"{prefix}total_line_books_mean"]
            * out[f"{prefix}total_prob_over_books_bias_from_50"]
        )

    if (
        f"{prefix}spread_books_abs_mean" in out.columns
        and f"{prefix}total_line_books_mean" in out.columns
    ):
        out[f"{prefix}total_line_x_spread_abs"] = (
            out[f"{prefix}total_line_books_mean"]
            * out[f"{prefix}spread_books_abs_mean"]
        )

    return out

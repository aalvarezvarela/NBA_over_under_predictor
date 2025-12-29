import numpy as np
import pandas as pd

from .db_config import connect_predictions_db
from .update_total_points_predictions import get_predictions_table_name


def get_games_with_total_scored_points(
    date: str = None, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    Connects to the nba_predictions database and returns all predictions
    where total_scored_points IS NOT NULL. Optionally filter by date or date range.

    Args:
        date (str, optional): Filter for a specific date (YYYY-MM-DD).
        start_date (str, optional): Start date for range (inclusive, YYYY-MM-DD).
        end_date (str, optional): End date for range (inclusive, YYYY-MM-DD).

    Returns:
        pd.DataFrame: All columns from nba_predictions for rows with a known final total.
    """
    table_name = get_predictions_table_name()
    conn = connect_predictions_db()
    where_clauses = ["total_scored_points IS NOT NULL"]
    params = []
    if date:
        where_clauses.append("game_date = %s")
        params.append(date)
    if start_date:
        where_clauses.append("game_date >= %s")
        params.append(start_date)
    if end_date:
        where_clauses.append("game_date <= %s")
        params.append(end_date)
    where_sql = " AND ".join(where_clauses)
    query = f"""
        SELECT *
        FROM {table_name}
        WHERE {where_sql}
    """
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def add_ou_betting_metrics(
    df: pd.DataFrame,
    *,
    line_col: str = "total_over_under_line",
    actual_total_col: str = "total_scored_points",
    regressor_pred_total_col: str = "predicted_total_score",
    classifier_pred_side_col: str = "classifier_prediction_model2",
    over_odds_col: str = "average_total_over_money",
    under_odds_col: str = "average_total_under_money",
) -> pd.DataFrame:
    """
    Adds over/under correctness and $1-bet profit metrics for:
      - regressor (using predicted total vs line)
      - classifier (using predicted side vs line)
      - both (only if both models pick the same side; else profit=0)

    Output columns added:
      - regressor_side, classifier_side, actual_side
      - regressor_correct, classifier_correct, both_correct
      - profit_regressor, profit_classifier, profit_both_agree

    Odds handling:
      - Accepts either American odds (e.g., -110, +120) OR decimal odds (e.g., 1.91, 2.20).
      - If abs(odds) >= 100, assumes American and converts to decimal first.
      - Profit on a $1 stake when winning = (decimal_odds - 1); loss = -1; push/unknown = 0.
      - Push (actual_total == line) => profit = 0, correct = False.
      - Missing data => profit = 0, correct = False (conservative).
    """
    required = [
        line_col,
        actual_total_col,
        regressor_pred_total_col,
        classifier_pred_side_col,
        over_odds_col,
        under_odds_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def american_to_decimal(a: pd.Series) -> pd.Series:
        a_num = pd.to_numeric(a, errors="coerce")
        dec = pd.Series(np.nan, index=a.index, dtype=float)
        pos = a_num > 0
        neg = a_num < 0
        dec[pos] = 1.0 + (a_num[pos] / 100.0)
        dec[neg] = 1.0 + (100.0 / a_num[neg].abs())
        return dec

    def to_decimal_odds(odds: pd.Series) -> pd.Series:
        """
        If abs(odds) >= 100 -> treat as American, convert.
        Else treat as decimal already (expects values like 1.01+).
        """
        o = pd.to_numeric(odds, errors="coerce")
        dec = o.astype(float).copy()

        american_mask = o.notna() & (o.abs() >= 100)
        if american_mask.any():
            dec.loc[american_mask] = american_to_decimal(o.loc[american_mask]).values

        # Guardrails: typical decimal odds should be > 1.0; otherwise set NaN
        dec = dec.where(dec > 1.0, np.nan)
        return dec

    def normalize_side(series: pd.Series) -> pd.Series:
        s = series.copy()
        if s.dtype == object:
            s_str = s.astype(str).str.strip().str.lower()
            over_mask = s_str.isin(["over", "o", "1", "true", "yes"])
            under_mask = s_str.isin(["under", "u", "0", "false", "no"])

            s_num = pd.to_numeric(s, errors="coerce")
            num_mask = s_num.notna() & ~(over_mask | under_mask)

            side = pd.Series(np.nan, index=s.index, dtype=object)
            side[over_mask] = "over"
            side[under_mask] = "under"
            side[num_mask & (s_num > 0)] = "over"
            side[num_mask & (s_num <= 0)] = "under"
            return side

        s_num = pd.to_numeric(s, errors="coerce")
        return pd.Series(
            np.where(s_num > 0, "over", np.where(s_num <= 0, "under", np.nan)),
            index=s.index,
            dtype=object,
        )

    # --- sides ---
    line = pd.to_numeric(df[line_col], errors="coerce")
    actual_total = pd.to_numeric(df[actual_total_col], errors="coerce")
    reg_pred_total = pd.to_numeric(df[regressor_pred_total_col], errors="coerce")

    df["actual_side"] = np.where(
        actual_total > line, "over", np.where(actual_total < line, "under", np.nan)
    )
    df["regressor_side"] = np.where(
        reg_pred_total > line, "over", np.where(reg_pred_total < line, "under", np.nan)
    )
    df["classifier_side"] = normalize_side(df[classifier_pred_side_col])
    df["both_agree"] = df["regressor_side"] == df["classifier_side"]
    df["both_agree_side"] = np.where(df["both_agree"], df["regressor_side"], np.nan)

    # --- correctness (push or missing => False) ---
    df["regressor_correct"] = (df["regressor_side"] == df["actual_side"]) & df[
        "actual_side"
    ].notna()
    df["classifier_correct"] = (df["classifier_side"] == df["actual_side"]) & df[
        "actual_side"
    ].notna()
    df["both_agree_correct"] = (df["both_agree_side"] == df["actual_side"]) & df[
        "actual_side"
    ].notna()

    # --- profit per $1 stake (using decimal odds) ---
    df[over_odds_col] = to_decimal_odds(df[over_odds_col])
    df[under_odds_col] = to_decimal_odds(df[under_odds_col])

    def profit_from_pick(prediction, is_correct, odds_over, odds_under) -> pd.Series:
        """
        Profit for a $1 bet following `pick_side` ('over'/'under').
        If win => +(decimal_odds - 1), if loss => -1, if push/unknown => 0.
        """
        if not prediction:
            return 0.0
        profit = -1
        if is_correct and prediction == "over":
            profit = odds_over - 1
        elif is_correct and prediction == "under":
            profit = odds_under - 1

        return profit

    df["profit_regressor"] = df.apply(
        lambda row: profit_from_pick(
            row["regressor_side"],
            row["regressor_correct"],
            row[over_odds_col],
            row[under_odds_col],
        ),
        axis=1,
    )
    df["profit_classifier"] = df.apply(
        lambda row: profit_from_pick(
            row["classifier_side"],
            row["classifier_correct"],
            row[over_odds_col],
            row[under_odds_col],
        ),
        axis=1,
    )

    # both agree profit
    df["profit_both_agree"] = df.apply(
        lambda row: profit_from_pick(
            row["both_agree_side"],
            row["both_agree_correct"],
            row[over_odds_col],
            row[under_odds_col],
        )
        if pd.notna(row["both_agree_side"])
        else 0.0,
        axis=1,
    )

    # Ensure bool dtype
    for c in ["regressor_correct", "classifier_correct", "both_agree_correct"]:
        df[c] = df[c].fillna(False).astype(bool)

    return df


def compute_ou_betting_statistics(
    df: pd.DataFrame,
    *,
    actual_side_col: str = "actual_side",
    reg_correct_col: str = "regressor_correct",
    clf_correct_col: str = "classifier_correct",
    both_correct_col: str = "both_agree_correct",
    reg_profit_col: str = "profit_regressor",
    clf_profit_col: str = "profit_classifier",
    both_profit_col: str = "profit_both_agree",
    both_pick_col: str = "both_agree_side",
    print_report: bool = True,
    title: str = "Over/Under Betting Performance Summary",
    float_decimals: int = 4,
) -> pd.DataFrame:
    """
    Builds summary statistics for:
      - regressor
      - classifier
      - both-agree strategy (only bets when both agree)

    Returns a single-row DataFrame with:
      - n_games_total
      - n_resolved (non-push rows, where actual_side is known)
      - per-method: n_bets, accuracy, total_profit, avg_profit_per_bet

    If print_report=True, prints a formatted report.
    """

    required = [
        actual_side_col,
        reg_correct_col,
        clf_correct_col,
        both_correct_col,
        reg_profit_col,
        clf_profit_col,
        both_profit_col,
        both_pick_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns (run add_ou_betting_metrics first): {missing}"
        )

    n_games_total = int(len(df))

    # "Resolved" games are those that are not pushes and have known actual side
    resolved_mask = df[actual_side_col].notna()
    n_resolved = int(resolved_mask.sum())

    def _accuracy(mask: pd.Series, correct_col: str) -> float:
        denom = int(mask.sum())
        if denom == 0:
            return float("nan")
        return float(df.loc[mask, correct_col].mean())

    def _profit_stats(mask: pd.Series, profit_col: str) -> tuple[int, float, float]:
        """
        Returns: (n_bets, total_profit, avg_profit_per_bet)
        avg_profit_per_bet excludes zeros (0 means no-bet / push / unknown by your convention).
        """
        profits = pd.to_numeric(df.loc[mask, profit_col], errors="coerce").fillna(0.0)

        total_profit = float(profits.sum())

        bet_mask = profits != 0
        n_bets = int(bet_mask.sum())
        avg_profit_per_bet = (
            float(profits.loc[bet_mask].mean()) if n_bets > 0 else float("nan")
        )

        return n_bets, total_profit, avg_profit_per_bet

    # Regressor and classifier: bet whenever game is resolved (non-push)
    reg_mask = resolved_mask
    clf_mask = resolved_mask

    # Both-agree: bet only when both agree AND game resolved
    both_mask = resolved_mask & df[both_pick_col].notna()

    reg_n_bets, reg_total_profit, reg_avg_profit = _profit_stats(
        reg_mask, reg_profit_col
    )
    clf_n_bets, clf_total_profit, clf_avg_profit = _profit_stats(
        clf_mask, clf_profit_col
    )
    both_n_bets, both_total_profit, both_avg_profit = _profit_stats(
        both_mask, both_profit_col
    )

    out = {
        "n_games_total": n_games_total,
        "n_resolved": n_resolved,
        "regressor_n_bets": reg_n_bets,
        "regressor_accuracy": _accuracy(reg_mask, reg_correct_col),
        "regressor_total_profit": reg_total_profit,
        "regressor_avg_profit_per_bet": reg_avg_profit,
        "classifier_n_bets": clf_n_bets,
        "classifier_accuracy": _accuracy(clf_mask, clf_correct_col),
        "classifier_total_profit": clf_total_profit,
        "classifier_avg_profit_per_bet": clf_avg_profit,
        "both_agree_n_bets": both_n_bets,
        "both_agree_accuracy": _accuracy(both_mask, both_correct_col),
        "both_agree_total_profit": both_total_profit,
        "both_agree_avg_profit_per_bet": both_avg_profit,
    }

    stats_df = pd.DataFrame([out])

    if print_report:
        _print_ou_stats_report(
            stats_df=stats_df,
            title=title,
            float_decimals=float_decimals,
        )

    return stats_df


def _print_ou_stats_report(
    *,
    stats_df: pd.DataFrame,
    title: str,
    float_decimals: int = 4,
) -> None:
    """Pretty-prints the single-row stats_df produced by compute_ou_betting_statistics."""
    if stats_df.empty:
        print(f"{title}\n(no data)")
        return

    row = stats_df.iloc[0].to_dict()

    n_games_total = int(row["n_games_total"])
    n_resolved = int(row["n_resolved"])

    def fmt_pct(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{100.0 * float(x):.{float_decimals}f}%"

    def fmt_num(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{float(x):.{float_decimals}f}"

    def fmt_int(x: int) -> str:
        return f"{int(x):d}"

    methods = [
        ("Regressor", "regressor"),
        ("Classifier", "classifier"),
        ("Both Agree (bet only when both pick same side)", "both_agree"),
    ]

    # Header
    print()
    print(title)
    print("=" * len(title))
    print(f"Total rows:     {fmt_int(n_games_total)}")
    print(
        f"Resolved rows:  {fmt_int(n_resolved)} (non-push games where actual side is known)"
    )
    print()

    # Table-like section
    col1_w = 46
    col2_w = 10
    col3_w = 14
    col4_w = 18

    header = (
        f"{'Method'.ljust(col1_w)}"
        f"{'Bets'.rjust(col2_w)}"
        f"{'Accuracy'.rjust(col3_w)}"
        f"{'Total Profit'.rjust(col4_w)}"
        f"{'Avg Profit/Bet'.rjust(col4_w)}"
    )
    print(header)
    print("-" * len(header))

    for label, prefix in methods:
        n_bets = row[f"{prefix}_n_bets"]
        acc = row[f"{prefix}_accuracy"]
        total_profit = row[f"{prefix}_total_profit"]
        avg_profit = row[f"{prefix}_avg_profit_per_bet"]

        line = (
            f"{label.ljust(col1_w)}"
            f"{fmt_int(n_bets).rjust(col2_w)}"
            f"{fmt_pct(acc).rjust(col3_w)}"
            f"{fmt_num(total_profit).rjust(col4_w)}"
            f"{fmt_num(avg_profit).rjust(col4_w)}"
        )
        print(line)

    print()
    print("Notes")
    print("-----")
    print("1) Accuracy denominators:")
    print("   - Regressor / Classifier: resolved rows (non-push).")
    print(
        "   - Both Agree: only rows where both models agree AND the game is resolved."
    )
    print(
        "2) Avg Profit/Bet excludes 0 profits (0 indicates no bet placed, or push/unknown in your conventions)."
    )
    print()


if __name__ == "__main__":
    df = get_games_with_total_scored_points()
    print(df)
    df_with_metrics = add_ou_betting_metrics(df)
    print(df_with_metrics)
    # Save to Excel
    # df_with_metrics.to_excel("/home/adrian_alvarez/Projects/NBA_over_under_predictor/Predictions/temp.xlsx", index=False)
    stats = compute_ou_betting_statistics(df_with_metrics, print_report=True)
    # print(stats)

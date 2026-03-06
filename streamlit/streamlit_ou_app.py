import os
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

import sys

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from scripts.predict_nba_games import predict_nba_games as run_nba_predictor
except Exception:
    run_nba_predictor = None

from nba_ou.postgre_db.predictions.update.update_evaluation_predictions import (
    get_games_with_total_scored_points,
)
from nba_ou.utils.streamlit_utils import get_team_logo_url

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

MODEL_ORDER = [
    "full_dataset",
    "full_dataset_total_points",
    "recent_games",
    "recent_games_total_points",
    "TabPFNRegressor",
]
MODEL_LABELS = {
    "full_dataset": "Full (Diff)",
    "full_dataset_total_points": "Full (Total)",
    "recent_games": "Recent (Diff)",
    "recent_games_total_points": "Recent (Total)",
    "TabPFNRegressor": "TabPFN",
}
MODEL_PREFIXES = {
    "full_dataset": "full_diff",
    "full_dataset_total_points": "full_total",
    "recent_games": "recent_diff",
    "recent_games_total_points": "recent_total",
    "TabPFNRegressor": "tabpfn",
}

TOTAL_POINTS_MODELS = [
    "full_dataset_total_points",
    "recent_games_total_points",
    "TabPFNRegressor",
]
DIFF_FROM_LINE_MODELS = ["full_dataset", "recent_games"]


def set_runtime_env_from_secrets() -> None:
    try:
        os.environ["SUPABASE_DB_URL"] = st.secrets["DatabaseSupabase"][
            "SUPABASE_DB_URL"
        ]
        os.environ["SUPABASE_DB_PASSWORD"] = st.secrets["DatabaseSupabase"][
            "SUPABASE_DB_PASSWORD"
        ]
    except Exception:
        pass

    try:
        os.environ["ODDS_API_KEY"] = st.secrets["Odds"]["ODDS_API_KEY"]
    except Exception:
        pass


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
          /* Page container */
          .main .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.0rem;
            max-width: 1400px;
          }

          /* Sidebar polish */
          section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.15);
          }
          section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
          }

          /* Sidebar text sizing */
          section[data-testid="stSidebar"] h3 {
            font-size: 1.75rem !important;
            font-weight: 700 !important;
          }
          section[data-testid="stSidebar"] div[role="radiogroup"] label {
            font-size: 1.5rem !important;
            font-weight: 600 !important;
          }
          section[data-testid="stSidebar"] div[role="radiogroup"] label div {
            font-size: 1.5rem !important;
          }
          section[data-testid="stSidebar"] .stCaption {
            font-size: 1.15rem !important;
          }

          /* Typography */
          h1, h2, h3 {
            letter-spacing: -0.02em;
          }
          h1 {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.2rem !important;
          }
          h2 {
            font-size: 1.8rem !important;
            font-weight: 750 !important;
          }
          h3 {
            font-size: 1.35rem !important;
            font-weight: 700 !important;
          }

          /* Metrics */
          .stMetric label {
            font-size: 1.05rem !important;
            font-weight: 650 !important;
          }
          .stMetric [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
          }

          /* DataFrame readability */
          div[data-testid="stDataFrame"] div[role="gridcell"] {
            padding: 0.65rem !important;
          }
          div[data-testid="stDataFrame"] div[role="columnheader"] {
            font-weight: 750 !important;
            padding: 0.85rem !important;
          }

          /* "Hero" header container */
          .app-hero {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 16px;
            padding: 18px 18px;
            background: linear-gradient(135deg,
              rgba(102, 126, 234, 0.15) 0%,
              rgba(118, 75, 162, 0.12) 100%);
            margin-bottom: 16px;
          }
          .app-subtitle {
            font-size: 1.05rem;
            opacity: 0.85;
            margin-top: 2px;
            margin-bottom: 10px;
          }
          .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 6px;
          }
          .chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(49, 51, 63, 0.10);
            font-size: 0.95rem;
            font-weight: 600;
          }

          /* Reduce visual noise on separators */
          hr {
            margin: 1.0rem 0;
            opacity: 0.25;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
          <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:12px;">
            <div style="flex: 1;">
              <div style="font-size: 0.95rem; font-weight: 700; opacity: 0.85;">NBA analytics</div>
              <div style="margin-top: 2px;">
                <span style="font-size: 2.2rem; font-weight: 900; letter-spacing: -0.02em;">
                  Over/Under Predictor
                </span>
              </div>
              <div class="app-subtitle">
                Predictions, results, and historical performance in one place.
              </div>
              <div class="chip-row">
                <span class="chip">📊 Full (Total)</span>
                <span class="chip">📊 Recent (Total)</span>
                <span class="chip">📊 TabPFN</span>
                <span class="chip">📏 Full (Diff)</span>
                <span class="chip">📏 Recent (Diff)</span>
                <span class="chip">🕐 Madrid (CEST)</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_model_type(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text in MODEL_ORDER:
        return text

    lowered = text.lower()
    # Check longer prefixes first to avoid premature match
    if lowered.startswith("full_dataset_total_points"):
        return "full_dataset_total_points"
    if lowered.startswith("full_dataset"):
        return "full_dataset"
    if lowered.startswith("recent_games_total_points"):
        return "recent_games_total_points"
    if lowered.startswith("recent_games"):
        return "recent_games"
    if lowered in {"tabpfnregressor", "tabpfn"} or lowered.startswith("tabpfn"):
        return "TabPFNRegressor"

    return None


def normalize_pick(value: object) -> str | float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()
    if text in {"OVER", "O", "1", "TRUE", "YES"}:
        return "OVER"
    if text in {"UNDER", "U", "0", "FALSE", "NO"}:
        return "UNDER"
    if text == "PUSH":
        return "PUSH"

    return np.nan


def format_madrid_datetime(series: pd.Series, fmt: str) -> pd.Series:
    dt_utc = pd.to_datetime(series, errors="coerce", utc=True)
    return dt_utc.dt.tz_convert("Europe/Madrid").dt.strftime(fmt)


def build_game_level_predictions(
    df: pd.DataFrame,
    prediction_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "model_type" not in work.columns:
        work["model_type"] = np.nan
    if "model_name" not in work.columns:
        work["model_name"] = np.nan
    if "prediction_source" not in work.columns:
        work["prediction_source"] = np.nan

    model_source = (
        work["model_type"]
        .where(work["model_type"].notna(), work["model_name"])
        .where(
            work["model_type"].notna() | work["model_name"].notna(),
            work["prediction_source"],
        )
    )
    work["model_type"] = model_source.apply(normalize_model_type)
    work = work[work["model_type"].isin(MODEL_ORDER)].copy()

    if work.empty:
        return pd.DataFrame()

    if "prediction_datetime" in work.columns:
        pred_dt = pd.to_datetime(work["prediction_datetime"], errors="coerce", utc=True)
    else:
        pred_dt = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")

    if "prediction_date" in work.columns:
        pred_date_dt = pd.to_datetime(
            work["prediction_date"], errors="coerce", utc=True
        )
    else:
        pred_date_dt = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")
    work["prediction_datetime_utc"] = pred_dt.fillna(pred_date_dt)

    if prediction_cutoff is not None:
        cutoff = pd.Timestamp(prediction_cutoff)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        work = work[work["prediction_datetime_utc"] <= cutoff].copy()

    if work.empty:
        return pd.DataFrame()

    if "game_time" in work.columns:
        work["game_time_utc"] = pd.to_datetime(
            work["game_time"], errors="coerce", utc=True
        )
    else:
        work["game_time_utc"] = pd.NaT

    line = pd.to_numeric(work.get("total_over_under_line"), errors="coerce")
    line_at_prediction = pd.to_numeric(
        work.get("total_bet365_line_at_prediction"), errors="coerce"
    )
    work["line_for_calc"] = line.fillna(line_at_prediction)
    work["pred_total_points"] = pd.to_numeric(
        work.get("pred_total_points"), errors="coerce"
    )
    work["pred_line_error"] = pd.to_numeric(
        work.get("pred_line_error"), errors="coerce"
    )

    # New schema allows either target value. Backfill missing side for downstream views.
    work["pred_total_points"] = work["pred_total_points"].where(
        work["pred_total_points"].notna(),
        work["line_for_calc"] + work["pred_line_error"],
    )
    work["pred_line_error"] = work["pred_line_error"].where(
        work["pred_line_error"].notna(),
        work["pred_total_points"] - work["line_for_calc"],
    )

    base_cols = [
        "game_id",
        "season_type",
        "game_date",
        "game_time",
        "game_time_utc",
        "team_name_team_home",
        "team_name_team_away",
        "total_over_under_line",
        "total_scored_points",
        "home_pts",
        "away_pts",
        "prediction_date",
        "prediction_datetime_utc",
        "time_to_match_minutes",
    ]
    available_base_cols = [col for col in base_cols if col in work.columns]

    base = (
        work.sort_values("prediction_datetime_utc")
        .groupby("game_id", as_index=False)
        .tail(1)[available_base_cols]
        .copy()
    )

    base = base.rename(
        columns={"prediction_datetime_utc": "latest_prediction_datetime"}
    )

    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        per_model = work[work["model_type"] == model_type].copy()
        if per_model.empty:
            continue

        # Prefer rows that carry the direct total-points target when duplicates exist.
        if "prediction_value_type" in per_model.columns:
            per_model["_prediction_priority"] = (
                per_model["prediction_value_type"].astype(str).str.upper()
                == "TOTAL_POINTS"
            ).astype(int)
        else:
            per_model["_prediction_priority"] = 0

        model_cols = [
            "game_id",
            "pred_pick",
            "pred_total_points",
            "pred_line_error",
            "prediction_datetime_utc",
            "_prediction_priority",
        ]
        available_model_cols = [col for col in model_cols if col in per_model.columns]

        per_model = (
            per_model.sort_values(["prediction_datetime_utc", "_prediction_priority"])
            .groupby("game_id", as_index=False)
            .tail(1)[available_model_cols]
        )

        rename_map = {
            "pred_pick": f"pick_{prefix}",
            "pred_total_points": f"pred_total_{prefix}",
            "pred_line_error": f"line_error_{prefix}",
            "prediction_datetime_utc": f"pred_dt_{prefix}",
        }
        per_model = per_model.rename(columns=rename_map)

        base = base.merge(per_model, on="game_id", how="left")

    line = pd.to_numeric(base.get("total_over_under_line"), errors="coerce")
    actual_total = pd.to_numeric(base.get("total_scored_points"), errors="coerce")

    # NumPy 2 raises DTypePromotionError when np.where mixes strings with np.nan.
    actual_side = pd.Series(index=base.index, dtype="object")
    actual_side.loc[actual_total > line] = "OVER"
    actual_side.loc[actual_total < line] = "UNDER"
    actual_side.loc[actual_total == line] = "PUSH"
    base["actual_side"] = actual_side

    def pick_from_diff(diff: pd.Series) -> pd.Series:
        out = pd.Series(index=diff.index, dtype="object")
        out.loc[diff > 0] = "OVER"
        out.loc[diff < 0] = "UNDER"
        out.loc[diff == 0] = "PUSH"
        return out

    model_pick_cols: list[str] = []
    model_total_cols: list[str] = []
    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        pick_col = f"pick_{prefix}"
        pred_total_col = f"pred_total_{prefix}"
        line_diff_col = f"line_diff_{prefix}"
        error_col = f"error_{prefix}"
        correct_col = f"correct_{prefix}"

        if pick_col not in base.columns:
            base[pick_col] = np.nan
        if pred_total_col not in base.columns:
            base[pred_total_col] = np.nan

        base[pick_col] = base[pick_col].apply(normalize_pick)
        base[pred_total_col] = pd.to_numeric(base[pred_total_col], errors="coerce")
        base[line_diff_col] = base[pred_total_col] - line

        derived_pick = pick_from_diff(base[line_diff_col])
        base[pick_col] = base[pick_col].where(base[pick_col].notna(), derived_pick)

        base[error_col] = base[pred_total_col] - actual_total
        base[correct_col] = (base[pick_col] == base["actual_side"]) & base[
            "actual_side"
        ].isin(["OVER", "UNDER"])

        model_pick_cols.append(pick_col)
        model_total_cols.append(pred_total_col)

    # Simple consensus: average of all available line diffs across models
    model_diff_cols = [f"line_diff_{MODEL_PREFIXES[m]}" for m in MODEL_ORDER]
    base["consensus_line_diff"] = base[model_diff_cols].mean(axis=1, skipna=True)
    base["consensus_pred_total"] = line + base["consensus_line_diff"]
    base["consensus_pick"] = pick_from_diff(base["consensus_line_diff"])
    base["consensus_error"] = base["consensus_pred_total"] - actual_total
    base["consensus_correct"] = (base["consensus_pick"] == base["actual_side"]) & base[
        "actual_side"
    ].isin(["OVER", "UNDER"])

    # Consensus without TabPFN (average of non-TabPFN model diffs)
    no_tabpfn_diff_cols = [
        f"line_diff_{MODEL_PREFIXES[m]}" for m in MODEL_ORDER if m != "TabPFNRegressor"
    ]
    base["consensus_no_tabpfn_line_diff"] = base[no_tabpfn_diff_cols].mean(
        axis=1, skipna=True
    )
    base["consensus_no_tabpfn_pred_total"] = (
        line + base["consensus_no_tabpfn_line_diff"]
    )
    base["consensus_no_tabpfn_pick"] = pick_from_diff(
        base["consensus_no_tabpfn_line_diff"]
    )
    base["consensus_no_tabpfn_error"] = (
        base["consensus_no_tabpfn_pred_total"] - actual_total
    )
    base["consensus_no_tabpfn_correct"] = (
        base["consensus_no_tabpfn_pick"] == base["actual_side"]
    ) & base["actual_side"].isin(["OVER", "UNDER"])

    base["all_models_available"] = base[model_total_cols].notna().all(axis=1)
    base["all_models_agree"] = (
        base[model_pick_cols].nunique(axis=1, dropna=True).eq(1)
        & base["all_models_available"]
    )

    if "game_time_utc" in base.columns:
        base = base.sort_values("game_time_utc")

    return base.reset_index(drop=True)


def build_upcoming_display(
    df: pd.DataFrame, show_pred_times: bool = False
) -> pd.DataFrame:
    display = pd.DataFrame()
    display["Matchup"] = df["team_name_team_home"] + " vs " + df["team_name_team_away"]
    display["Game Time (Madrid)"] = format_madrid_datetime(
        df["game_time_utc"], "%Y-%m-%d %H:%M"
    )
    display["O/U Line"] = pd.to_numeric(
        df["total_over_under_line"], errors="coerce"
    ).round(1)

    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        label = MODEL_LABELS[model_type]
        display[f"{label} Total"] = pd.to_numeric(
            df[f"pred_total_{prefix}"], errors="coerce"
        ).round(1)
        display[f"{label} Diff"] = pd.to_numeric(
            df[f"line_diff_{prefix}"], errors="coerce"
        ).round(2)
        display[f"{label} Pick"] = df[f"pick_{prefix}"]

        # Optionally show prediction time for each model
        if show_pred_times and f"pred_dt_{prefix}" in df.columns:
            display[f"{label} Time"] = format_madrid_datetime(
                df[f"pred_dt_{prefix}"], "%m-%d %H:%M"
            )

    display["Consensus Total"] = pd.to_numeric(
        df["consensus_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus Diff"] = pd.to_numeric(
        df["consensus_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus Pick"] = df["consensus_pick"]

    display["Consensus (No TabPFN) Total"] = pd.to_numeric(
        df["consensus_no_tabpfn_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus (No TabPFN) Diff"] = pd.to_numeric(
        df["consensus_no_tabpfn_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus (No TabPFN) Pick"] = df["consensus_no_tabpfn_pick"]

    if "time_to_match_minutes" in df.columns:
        display["Time to Game (min)"] = (
            pd.to_numeric(df["time_to_match_minutes"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return display


def build_past_display(df: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame()
    display["Matchup"] = df["team_name_team_home"] + " vs " + df["team_name_team_away"]
    display["Game Time (Madrid)"] = format_madrid_datetime(df["game_time_utc"], "%H:%M")
    display["O/U Line"] = pd.to_numeric(
        df["total_over_under_line"], errors="coerce"
    ).round(1)
    display["Actual Total"] = pd.to_numeric(
        df["total_scored_points"], errors="coerce"
    ).round(1)
    display["Actual Side"] = df["actual_side"]

    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        label = MODEL_LABELS[model_type]
        display[f"{label} Total"] = pd.to_numeric(
            df[f"pred_total_{prefix}"], errors="coerce"
        ).round(1)
        display[f"{label} Diff"] = pd.to_numeric(
            df[f"line_diff_{prefix}"], errors="coerce"
        ).round(2)
        display[f"{label} Pick"] = df[f"pick_{prefix}"]
        display[f"{label} Correct"] = df[f"correct_{prefix}"].map(
            {True: "✅", False: "❌"}
        )

    display["Consensus Total"] = pd.to_numeric(
        df["consensus_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus Diff"] = pd.to_numeric(
        df["consensus_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus Pick"] = df["consensus_pick"]
    display["Consensus Correct"] = df["consensus_correct"].map(
        {True: "✅", False: "❌"}
    )

    display["Consensus (No TabPFN) Total"] = pd.to_numeric(
        df["consensus_no_tabpfn_pred_total"], errors="coerce"
    ).round(1)
    display["Consensus (No TabPFN) Diff"] = pd.to_numeric(
        df["consensus_no_tabpfn_line_diff"], errors="coerce"
    ).round(2)
    display["Consensus (No TabPFN) Pick"] = df["consensus_no_tabpfn_pick"]
    display["Consensus (No TabPFN) Correct"] = df["consensus_no_tabpfn_correct"].map(
        {True: "✅", False: "❌"}
    )

    return display


def format_pick_label(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    text = str(value).upper()
    if text == "OVER":
        return "Over"
    if text == "UNDER":
        return "Under"
    if text == "PUSH":
        return "Push"
    if text == "MIXED":
        return "Mixed"
    return str(value)


def get_pick_icon(pick_label: str) -> str:
    if pick_label == "Under":
        return "🔵"
    if pick_label == "Over":
        return "🔴"
    return "⚪"


def _build_model_cell_html(
    row: pd.Series,
    model_type: str,
    *,
    show_pred_total: bool = True,
) -> str:
    """Build HTML for a single model prediction cell."""
    prefix = MODEL_PREFIXES[model_type]
    label = MODEL_LABELS[model_type]
    diff = pd.to_numeric(row.get(f"line_diff_{prefix}"), errors="coerce")
    pick = row.get(f"pick_{prefix}")
    diff_text = f"{diff:+.1f}" if pd.notna(diff) else "—"

    if pd.notna(pick) and pick in ("OVER", "UNDER"):
        arrow = "▲" if pick == "OVER" else "▼"
        clr = "#e74c3c" if pick == "OVER" else "#2980b9"
        pk_text = str(pick)
    else:
        arrow, clr, pk_text = "—", "#95a5a6", "N/A"

    pred_html = ""
    if show_pred_total:
        pred = pd.to_numeric(row.get(f"pred_total_{prefix}"), errors="coerce")
        pred_text = f"{pred:.1f}" if pd.notna(pred) else "—"
        pred_html = (
            f'<div style="font-size:1.3rem;font-weight:800;margin:2px 0;">'
            f"{pred_text}</div>"
        )

    return (
        f'<div style="flex:1;text-align:center;padding:10px 4px;'
        f"background:rgba(128,128,128,0.06);border-radius:8px;"
        f'border:1px solid rgba(128,128,128,0.1);">'
        f'<div style="font-size:0.8rem;font-weight:700;color:#888;'
        f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:3px;">'
        f"{label}</div>"
        f"{pred_html}"
        f'<div style="font-size:1.05rem;font-weight:700;color:{clr};">{diff_text}</div>'
        f'<div style="font-size:0.85rem;font-weight:700;color:{clr};margin-top:3px;">'
        f"{arrow} {pk_text}</div>"
        f"</div>"
    )


def _render_game_card(row: pd.Series, include_actual: bool) -> None:
    """Render a single game prediction card as self-contained HTML."""
    home_team = row["team_name_team_home"]
    away_team = row["team_name_team_away"]
    home_logo = get_team_logo_url(home_team)
    away_logo = get_team_logo_url(away_team)

    game_dt = pd.to_datetime(row["game_time_utc"], errors="coerce", utc=True)
    if pd.isna(game_dt):
        game_time, game_date = "TBD", "TBD"
    else:
        dt_madrid = game_dt.tz_convert("Europe/Madrid")
        game_time = dt_madrid.strftime("%I:%M %p")
        game_date = dt_madrid.strftime("%b %d, %Y")

    line_val = pd.to_numeric(row.get("total_over_under_line"), errors="coerce")
    line_text = f"{line_val:.1f}" if pd.notna(line_val) else "N/A"

    consensus_diff = pd.to_numeric(row.get("consensus_line_diff"), errors="coerce")
    consensus_pick = row.get("consensus_pick")
    consensus_total = pd.to_numeric(row.get("consensus_pred_total"), errors="coerce")

    # Bet recommendation styling
    if pd.notna(consensus_pick) and consensus_pick in ("OVER", "UNDER"):
        is_over = consensus_pick == "OVER"
        bet_label = "BET OVER ▲" if is_over else "BET UNDER ▼"
        accent = "#e74c3c" if is_over else "#2980b9"
        banner_bg = "rgba(231,76,60,0.12)" if is_over else "rgba(41,128,185,0.12)"
    else:
        bet_label = "PUSH —"
        accent = "#7f8c8d"
        banner_bg = "rgba(127,140,141,0.08)"

    margin_text = f"{consensus_diff:+.1f}" if pd.notna(consensus_diff) else "—"
    cons_total_text = f"{consensus_total:.1f}" if pd.notna(consensus_total) else "—"

    # Model agreement
    picks_list: list[str] = []
    for m in MODEL_ORDER:
        p = MODEL_PREFIXES[m]
        pk = row.get(f"pick_{p}")
        if pd.notna(pk) and pk in ("OVER", "UNDER"):
            picks_list.append(str(pk))
    n_avail = len(picks_list)
    if picks_list:
        dominant = max(set(picks_list), key=picks_list.count)
        n_agree = picks_list.count(dominant)
    else:
        n_agree = 0
    agree_text = f"{n_agree}/{n_avail} agree" if n_avail else "No models"

    # Build model cells
    tp_cells = "".join(
        _build_model_cell_html(row, m, show_pred_total=True)
        for m in TOTAL_POINTS_MODELS
    )
    dl_cells = "".join(
        _build_model_cell_html(row, m, show_pred_total=False)
        for m in DIFF_FROM_LINE_MODELS
    )

    # Actual result banner (past games only)
    actual_banner = ""
    if include_actual:
        actual_total = pd.to_numeric(row.get("total_scored_points"), errors="coerce")
        actual_side = row.get("actual_side")
        cons_correct = row.get("consensus_correct")

        if pd.notna(actual_side) and actual_side in ("OVER", "UNDER", "PUSH"):
            a_clr = (
                "#e74c3c"
                if actual_side == "OVER"
                else "#2980b9"
                if actual_side == "UNDER"
                else "#7f8c8d"
            )
            a_text = (
                f"{actual_side} ({actual_total:.1f} pts)"
                if pd.notna(actual_total)
                else actual_side
            )
        else:
            a_clr, a_text = "#95a5a6", "Pending"

        icon = (
            "✅"
            if pd.notna(cons_correct) and cons_correct
            else "❌"
            if pd.notna(cons_correct)
            else "⏳"
        )
        actual_banner = (
            f'<div style="background:{a_clr};color:white;text-align:center;'
            f'padding:10px;font-weight:700;font-size:1.05rem;">'
            f"{icon} RESULT: {a_text}</div>"
        )

    # Per-model correctness summary for past games
    model_results_html = ""
    if include_actual:
        result_cells = ""
        for m in MODEL_ORDER:
            p = MODEL_PREFIXES[m]
            lbl = MODEL_LABELS[m]
            flag = row.get(f"correct_{p}")
            r_icon = (
                "✅" if pd.notna(flag) and flag else "❌" if pd.notna(flag) else "—"
            )
            result_cells += (
                f'<div style="flex:1;text-align:center;font-size:0.9rem;'
                f'font-weight:600;">'
                f'<div style="font-size:0.75rem;color:#888;'
                f'text-transform:uppercase;">{lbl}</div>'
                f"{r_icon}</div>"
            )
        model_results_html = (
            f'<div style="display:flex;gap:4px;padding:8px 14px 4px;'
            f'border-top:1px solid rgba(128,128,128,0.1);">'
            f"{result_cells}</div>"
        )

    card_html = f"""
    <div style="border:2px solid {accent};border-radius:16px;overflow:hidden;
                margin-bottom:16px;box-shadow:0 4px 16px rgba(0,0,0,0.08);
                font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
                background:#fff;">
      <!-- Header -->
      <div style="background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
                  padding:18px 16px;color:white;">
        <div style="display:flex;align-items:center;justify-content:space-between;">
          <div style="flex:1;text-align:center;">
            <img src="{home_logo}" width="64" style="margin-bottom:4px;"
                 onerror="this.style.display='none'">
            <div style="font-size:1.05rem;font-weight:700;line-height:1.2;">
              {home_team}</div>
          </div>
          <div style="flex:0.5;text-align:center;">
            <div style="font-size:0.85rem;opacity:0.8;">{game_date}</div>
            <div style="font-size:1.5rem;font-weight:900;margin:4px 0;">VS</div>
            <div style="font-size:1.0rem;font-weight:600;">🕐 {game_time}</div>
          </div>
          <div style="flex:1;text-align:center;">
            <img src="{away_logo}" width="64" style="margin-bottom:4px;"
                 onerror="this.style.display='none'">
            <div style="font-size:1.05rem;font-weight:700;line-height:1.2;">
              {away_team}</div>
          </div>
        </div>
      </div>
      {actual_banner}
      <!-- Consensus Banner -->
      <div style="background:{banner_bg};padding:12px 16px;
                  border-bottom:1px solid rgba(128,128,128,0.12);">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <div style="font-size:1.6rem;font-weight:900;color:{accent};
                        letter-spacing:-0.01em;">{bet_label}</div>
            <div style="font-size:0.8rem;color:#888;margin-top:1px;">{agree_text}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:0.7rem;font-weight:600;color:#aaa;
                        text-transform:uppercase;">O/U Line</div>
            <div style="font-size:1.5rem;font-weight:800;color:#333;">{line_text}</div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:0.7rem;font-weight:600;color:#aaa;
                        text-transform:uppercase;">Predicted</div>
            <div style="font-size:1.3rem;font-weight:800;color:#333;">
              {cons_total_text}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.7rem;font-weight:600;color:#aaa;
                        text-transform:uppercase;">Margin</div>
            <div style="font-size:1.3rem;font-weight:700;color:{accent};">
              {margin_text}</div>
          </div>
        </div>
      </div>
      <!-- Models -->
      <div style="padding:12px 14px;">
        <div style="font-size:0.75rem;font-weight:700;color:#999;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:6px;">
          📊 Total Points Predictions
        </div>
        <div style="display:flex;gap:6px;margin-bottom:12px;">
          {tp_cells}
        </div>
        <div style="font-size:0.75rem;font-weight:700;color:#999;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:6px;">
          📏 Diff from Line Predictions
        </div>
        <div style="display:flex;gap:6px;">
          {dl_cells}
        </div>
      </div>
      {model_results_html}
    </div>
    """
    card_height = 570 if not include_actual else 640
    st.components.v1.html(card_html, height=card_height)


def render_prediction_cards(df: pd.DataFrame, include_actual: bool = False) -> None:
    df = df.sort_values("game_time_utc").reset_index(drop=True)
    cols_per_row = 2

    for idx in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            row_idx = idx + col_idx
            if row_idx >= len(df):
                break
            row = df.iloc[row_idx]
            with col:
                _render_game_card(row, include_actual)


def summarize_model_performance(df: pd.DataFrame) -> pd.DataFrame:
    resolved_mask = df["actual_side"].isin(["OVER", "UNDER"])
    rows = []

    for model_type in MODEL_ORDER:
        prefix = MODEL_PREFIXES[model_type]
        picks = df[f"pick_{prefix}"]
        valid_mask = resolved_mask & picks.isin(["OVER", "UNDER"])

        n_games = int(valid_mask.sum())
        accuracy = (
            float(df.loc[valid_mask, f"correct_{prefix}"].mean()) if n_games else np.nan
        )
        mean_error = (
            float(df.loc[valid_mask, f"error_{prefix}"].mean()) if n_games else np.nan
        )
        mae = (
            float(df.loc[valid_mask, f"error_{prefix}"].abs().mean())
            if n_games
            else np.nan
        )
        mean_abs_line_diff = (
            float(df.loc[valid_mask, f"line_diff_{prefix}"].abs().mean())
            if n_games
            else np.nan
        )

        rows.append(
            {
                "Model": MODEL_LABELS[model_type],
                "Games": n_games,
                "Accuracy (%)": None if pd.isna(accuracy) else round(accuracy * 100, 2),
                "Mean Error": None if pd.isna(mean_error) else round(mean_error, 2),
                "MAE": None if pd.isna(mae) else round(mae, 2),
                "Avg |Diff vs Line|": None
                if pd.isna(mean_abs_line_diff)
                else round(mean_abs_line_diff, 2),
            }
        )

    # Add consensus row
    consensus_picks = df["consensus_pick"]
    consensus_valid_mask = resolved_mask & consensus_picks.isin(["OVER", "UNDER"])

    n_consensus_games = int(consensus_valid_mask.sum())
    consensus_accuracy = (
        float(df.loc[consensus_valid_mask, "consensus_correct"].mean())
        if n_consensus_games
        else np.nan
    )
    consensus_mean_error = (
        float(df.loc[consensus_valid_mask, "consensus_error"].mean())
        if n_consensus_games
        else np.nan
    )
    consensus_mae = (
        float(df.loc[consensus_valid_mask, "consensus_error"].abs().mean())
        if n_consensus_games
        else np.nan
    )
    consensus_mean_abs_line_diff = (
        float(df.loc[consensus_valid_mask, "consensus_line_diff"].abs().mean())
        if n_consensus_games
        else np.nan
    )

    rows.append(
        {
            "Model": "Consensus",
            "Games": n_consensus_games,
            "Accuracy (%)": None
            if pd.isna(consensus_accuracy)
            else round(consensus_accuracy * 100, 2),
            "Mean Error": None
            if pd.isna(consensus_mean_error)
            else round(consensus_mean_error, 2),
            "MAE": None if pd.isna(consensus_mae) else round(consensus_mae, 2),
            "Avg |Diff vs Line|": None
            if pd.isna(consensus_mean_abs_line_diff)
            else round(consensus_mean_abs_line_diff, 2),
        }
    )

    # Add consensus (no TabPFN) row
    consensus_no_tabpfn_picks = df["consensus_no_tabpfn_pick"]
    consensus_no_tabpfn_valid_mask = resolved_mask & consensus_no_tabpfn_picks.isin(
        ["OVER", "UNDER"]
    )

    n_consensus_no_tabpfn_games = int(consensus_no_tabpfn_valid_mask.sum())
    consensus_no_tabpfn_accuracy = (
        float(
            df.loc[consensus_no_tabpfn_valid_mask, "consensus_no_tabpfn_correct"].mean()
        )
        if n_consensus_no_tabpfn_games
        else np.nan
    )
    consensus_no_tabpfn_mean_error = (
        float(
            df.loc[consensus_no_tabpfn_valid_mask, "consensus_no_tabpfn_error"].mean()
        )
        if n_consensus_no_tabpfn_games
        else np.nan
    )
    consensus_no_tabpfn_mae = (
        float(
            df.loc[consensus_no_tabpfn_valid_mask, "consensus_no_tabpfn_error"]
            .abs()
            .mean()
        )
        if n_consensus_no_tabpfn_games
        else np.nan
    )
    consensus_no_tabpfn_mean_abs_line_diff = (
        float(
            df.loc[consensus_no_tabpfn_valid_mask, "consensus_no_tabpfn_line_diff"]
            .abs()
            .mean()
        )
        if n_consensus_no_tabpfn_games
        else np.nan
    )

    rows.append(
        {
            "Model": "Consensus (No TabPFN)",
            "Games": n_consensus_no_tabpfn_games,
            "Accuracy (%)": None
            if pd.isna(consensus_no_tabpfn_accuracy)
            else round(consensus_no_tabpfn_accuracy * 100, 2),
            "Mean Error": None
            if pd.isna(consensus_no_tabpfn_mean_error)
            else round(consensus_no_tabpfn_mean_error, 2),
            "MAE": None
            if pd.isna(consensus_no_tabpfn_mae)
            else round(consensus_no_tabpfn_mae, 2),
            "Avg |Diff vs Line|": None
            if pd.isna(consensus_no_tabpfn_mean_abs_line_diff)
            else round(consensus_no_tabpfn_mean_abs_line_diff, 2),
        }
    )

    return pd.DataFrame(rows)


def compute_threshold_accuracy_table(
    df: pd.DataFrame,
    *,
    model_thresholds: dict[str, tuple[float, ...]] | None = None,
) -> pd.DataFrame:
    if model_thresholds is None:
        model_thresholds = {m: (0.0, 0.5, 1.0, 1.5, 2.0) for m in MODEL_ORDER}
        model_thresholds["consensus"] = (0.0, 0.5, 1.0, 1.5, 2.0)
        model_thresholds["consensus_no_tabpfn"] = (0.0, 0.5, 1.0, 1.5, 2.0)

    resolved_mask = df["actual_side"].isin(["OVER", "UNDER"])
    rows: list[dict] = []

    for model_type, thresholds in model_thresholds.items():
        if model_type == "consensus":
            pick_col = "consensus_pick"
            correct_col = "consensus_correct"
            line_diff_col = "consensus_line_diff"
            model_label = "Consensus"
        elif model_type == "consensus_no_tabpfn":
            pick_col = "consensus_no_tabpfn_pick"
            correct_col = "consensus_no_tabpfn_correct"
            line_diff_col = "consensus_no_tabpfn_line_diff"
            model_label = "Consensus (No TabPFN)"
        else:
            prefix = MODEL_PREFIXES[model_type]
            pick_col = f"pick_{prefix}"
            correct_col = f"correct_{prefix}"
            line_diff_col = f"line_diff_{prefix}"
            model_label = MODEL_LABELS[model_type]

        for threshold in thresholds:
            mask = (
                resolved_mask
                & df[pick_col].isin(["OVER", "UNDER"])
                & (pd.to_numeric(df[line_diff_col], errors="coerce").abs() >= threshold)
            )
            n_games = int(mask.sum())
            accuracy = float(df.loc[mask, correct_col].mean()) if n_games else np.nan

            rows.append(
                {
                    "Model": model_label,
                    "Filter": f"|Diff vs Line| >= {threshold:g}",
                    "Games": n_games,
                    "Accuracy (%)": None
                    if pd.isna(accuracy)
                    else round(accuracy * 100, 2),
                }
            )

    return pd.DataFrame(rows)


def compute_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out_rows = []
    temp = df.copy()
    temp["game_date_dt"] = pd.to_datetime(temp["game_date"], errors="coerce")
    temp = temp.dropna(subset=["game_date_dt"])  # keep only valid dates

    for game_date, group in temp.groupby(temp["game_date_dt"].dt.date):
        resolved = group[group["actual_side"].isin(["OVER", "UNDER"])]

        for model_type in MODEL_ORDER:
            prefix = MODEL_PREFIXES[model_type]
            valid = resolved[resolved[f"pick_{prefix}"].isin(["OVER", "UNDER"])]
            n_games = len(valid)

            out_rows.append(
                {
                    "game_date": pd.to_datetime(game_date),
                    "model_type": model_type,
                    "model_label": MODEL_LABELS[model_type],
                    "n_games": n_games,
                    "accuracy": valid[f"correct_{prefix}"].mean()
                    if n_games
                    else np.nan,
                    "mae": valid[f"error_{prefix}"].abs().mean() if n_games else np.nan,
                }
            )

        # Add consensus metrics for this date
        consensus_valid = resolved[resolved["consensus_pick"].isin(["OVER", "UNDER"])]
        n_consensus_games = len(consensus_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": "consensus",
                "model_label": "Consensus",
                "n_games": n_consensus_games,
                "accuracy": consensus_valid["consensus_correct"].mean()
                if n_consensus_games
                else np.nan,
                "mae": consensus_valid["consensus_error"].abs().mean()
                if n_consensus_games
                else np.nan,
            }
        )

        # Add consensus (no TabPFN) metrics for this date
        consensus_no_tabpfn_valid = resolved[
            resolved["consensus_no_tabpfn_pick"].isin(["OVER", "UNDER"])
        ]
        n_consensus_no_tabpfn_games = len(consensus_no_tabpfn_valid)

        out_rows.append(
            {
                "game_date": pd.to_datetime(game_date),
                "model_type": "consensus_no_tabpfn",
                "model_label": "Consensus (No TabPFN)",
                "n_games": n_consensus_no_tabpfn_games,
                "accuracy": consensus_no_tabpfn_valid[
                    "consensus_no_tabpfn_correct"
                ].mean()
                if n_consensus_no_tabpfn_games
                else np.nan,
                "mae": consensus_no_tabpfn_valid["consensus_no_tabpfn_error"]
                .abs()
                .mean()
                if n_consensus_no_tabpfn_games
                else np.nan,
            }
        )

    return pd.DataFrame(out_rows).sort_values(["game_date", "model_type"])


def show_upcoming_predictions() -> None:
    st.markdown("### 🔄 Update Predictions")
    st.caption(
        "Run the prediction model to generate fresh predictions for today's games."
    )
    st.markdown("")

    if run_nba_predictor is None:
        st.warning("Predictor module could not be imported in this environment.")
    elif st.button("Run Predictor Now", type="primary", use_container_width=True):
        try:
            with st.spinner("Running predictor. This may take a few minutes..."):
                run_nba_predictor(run_tabpfn_client=True)

            st.success("Predictions updated. Reloading...")
            time.sleep(1.5)
            st.rerun()
        except Exception as exc:
            st.error(f"Error running predictor: {exc}")
            st.exception(exc)

    st.markdown("---")

    # Add time selector for filtering predictions
    st.markdown("### ⏰ Select Prediction Time")

    use_custom_time = st.checkbox(
        "Filter predictions by time (use most recent prediction per model until selected time)",
        value=False,
    )

    prediction_cutoff = None
    if use_custom_time:
        col_date, col_time = st.columns(2)

        with col_date:
            selected_datetime = st.date_input(
                "Date", value=datetime.now().date(), key="pred_cutoff_date"
            )

        with col_time:
            selected_time = st.time_input(
                "Time (Madrid)", value=datetime.now().time(), key="pred_cutoff_time"
            )

        # Combine date and time, localize to Madrid timezone
        cutoff_madrid = datetime.combine(selected_datetime, selected_time)
        cutoff_madrid = pd.Timestamp(cutoff_madrid, tz="Europe/Madrid")
        prediction_cutoff = cutoff_madrid.tz_convert("UTC")

        st.caption(
            f"📌 Using predictions up to: {cutoff_madrid.strftime('%Y-%m-%d %H:%M')} Madrid"
        )

    with st.spinner("Loading upcoming predictions..."):
        raw = get_games_with_total_scored_points(only_null=True)

        # Build predictions with optional cutoff
        games = build_game_level_predictions(raw, prediction_cutoff=prediction_cutoff)

    if games.empty:
        st.info("No upcoming predictions found for the selected time.")
        return

    latest_prediction_time = pd.to_datetime(
        games["latest_prediction_datetime"], errors="coerce", utc=True
    ).max()

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Upcoming Games", len(games))
    with metric_cols[1]:
        available_all = int(games["all_models_available"].sum())
        st.metric("All Models Available", f"{available_all}/{len(games)}")
    with metric_cols[2]:
        if pd.notna(latest_prediction_time):
            st.metric(
                "Latest Prediction",
                latest_prediction_time.tz_convert("Europe/Madrid").strftime(
                    "%Y-%m-%d %H:%M"
                ),
            )
        else:
            st.metric("Latest Prediction", "N/A")

    if pd.notna(latest_prediction_time):
        cutoff_note = ""
        if prediction_cutoff is not None:
            cutoff_note = f" (filtered up to {prediction_cutoff.tz_convert('Europe/Madrid').strftime('%Y-%m-%d %H:%M:%S')} Madrid)"
        st.caption(
            "Latest prediction: "
            + latest_prediction_time.tz_convert("Europe/Madrid").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + " (Madrid)"
            + cutoff_note
        )

    st.markdown("---")
    st.markdown("## Today's Predictions")
    st.caption("View upcoming games with AI-powered over/under predictions.")
    st.markdown("")
    use_cards = st.toggle("Use Card View", value=True)

    if use_cards:
        render_prediction_cards(games, include_actual=False)
    else:
        st.dataframe(
            build_upcoming_display(games, show_pred_times=use_custom_time),
            width="stretch",
            hide_index=True,
            height=600,
        )

    st.markdown("---")
    st.markdown("")
    with st.expander("ℹ️ **How to Read the Predictions**", expanded=False):
        st.markdown(
            """
        ### 📊 Understanding the Predictions

        - **🏀 Matchup**: Home team vs Away team with logos
        - **⏰ Game Time**: When the game starts (Madrid timezone)
        - **📏 O/U Line**: The bookmaker's over/under betting line
        - **📊 Total Points Models**: Full (Total), Recent (Total), TabPFN — predict total points directly
        - **📏 Diff from Line Models**: Full (Diff), Recent (Diff) — predict the difference from the line
        - **🎯 Consensus**: Average of all model margins to decide OVER/UNDER
        - **Margin**: How far the consensus prediction is from the line (positive = OVER)

        ### ⏰ Time Filtering

        - **Latest Predictions** (default): Shows the most recent prediction for each model
        - **Filter by Time**: Select a specific date/time to see predictions as they were at that moment
          - Each model will show its most recent prediction up to the selected time
          - Useful for analyzing how predictions evolved over time

        **Note**: Predictions are updated periodically. Most recent prediction time shown above.
        """
        )


def show_past_games_results() -> None:
    st.markdown("## Past Games Results")
    st.caption("Compare predictions vs actual totals for a selected date.")
    st.markdown("")

    default_date = datetime.now() - timedelta(days=1)
    selected_date = st.date_input("Select Date", value=default_date)
    date_str = selected_date.strftime("%Y-%m-%d")

    with st.spinner("Loading completed games..."):
        raw = get_games_with_total_scored_points(only_null=False, date=date_str)

    if raw.empty:
        st.warning(f"No completed games found for {date_str}.")
        return

    if "prediction_datetime" in raw.columns:
        prediction_dt = pd.to_datetime(
            raw["prediction_datetime"], errors="coerce", utc=True
        )
    else:
        prediction_dt = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")

    if "prediction_date" in raw.columns:
        prediction_date_dt = pd.to_datetime(
            raw["prediction_date"], errors="coerce", utc=True
        )
    else:
        prediction_date_dt = pd.Series(
            pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]"
        )
    raw["prediction_datetime_utc"] = prediction_dt.fillna(prediction_date_dt)

    unique_times = sorted(
        raw["prediction_datetime_utc"].dropna().unique(), reverse=True
    )
    if not unique_times:
        st.warning(f"No prediction timestamps found for {date_str}.")
        return

    st.markdown("### ⏰ Select Prediction Time")
    st.caption("Choose which prediction time to analyze (most recent is default).")
    mapping: dict[str, pd.Timestamp] = {}
    options: list[str] = []
    for ts in unique_times:
        ts_madrid = pd.Timestamp(ts).tz_convert("Europe/Madrid")
        label = ts_madrid.strftime("%Y-%m-%d %H:%M:%S")
        options.append(label)
        mapping[label] = pd.Timestamp(ts)

    selected_label = st.selectbox(
        "Prediction Time:",
        options=options,
        index=0,
        help="Select which prediction snapshot to analyze.",
    )

    selected_cutoff = mapping[selected_label]
    games = build_game_level_predictions(raw, prediction_cutoff=selected_cutoff)

    if games.empty:
        st.warning("No games available at the selected prediction time.")
        return

    resolved_mask = games["actual_side"].isin(["OVER", "UNDER"])
    n_resolved = int(resolved_mask.sum())

    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric("🎮 Games Played", n_resolved)
    with metrics_cols[1]:
        consensus_mask = resolved_mask & games["consensus_pick"].isin(["OVER", "UNDER"])
        consensus_correct = int(games.loc[consensus_mask, "consensus_correct"].sum())
        consensus_total = int(consensus_mask.sum())
        st.metric("🎯 Consensus", f"{consensus_correct}/{consensus_total}")
    with metrics_cols[2]:
        consensus_no_tabpfn_mask = resolved_mask & games[
            "consensus_no_tabpfn_pick"
        ].isin(["OVER", "UNDER"])
        consensus_no_tabpfn_correct = int(
            games.loc[consensus_no_tabpfn_mask, "consensus_no_tabpfn_correct"].sum()
        )
        consensus_no_tabpfn_total = int(consensus_no_tabpfn_mask.sum())
        st.metric(
            "📊 No TabPFN",
            f"{consensus_no_tabpfn_correct}/{consensus_no_tabpfn_total}",
        )

    st.markdown("")
    model_metric_cols = st.columns(len(MODEL_ORDER))
    for idx, model_type in enumerate(MODEL_ORDER):
        prefix = MODEL_PREFIXES[model_type]
        label = MODEL_LABELS[model_type]
        model_mask = resolved_mask & games[f"pick_{prefix}"].isin(["OVER", "UNDER"])
        correct = int(games.loc[model_mask, f"correct_{prefix}"].sum())
        total = int(model_mask.sum())
        with model_metric_cols[idx]:
            st.metric(label, f"{correct}/{total}")

    st.markdown("---")
    st.markdown(f"### 🏀 Games on {date_str}")
    st.markdown("")
    use_cards = st.toggle("Use Card View", value=True, key="past_cards")

    if use_cards:
        render_prediction_cards(games, include_actual=True)
    else:
        st.dataframe(
            build_past_display(games), width="stretch", hide_index=True, height=600
        )


def show_historical_performance() -> None:
    st.markdown("## Historical Betting Performance")
    st.caption("Analyze model accuracy and prediction error over time.")
    st.markdown("")

    use_date_filter = st.checkbox("Filter by Date Range", value=False)
    start_date = None
    end_date = None

    if use_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2026-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    with st.spinner("Loading historical predictions..."):
        raw = get_games_with_total_scored_points(
            only_null=False,
            start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
            end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
        )
        games = build_game_level_predictions(raw)

    if games.empty:
        st.warning("No historical rows found for the selected range.")
        return

    resolved_mask = games["actual_side"].isin(["OVER", "UNDER"])
    total_games = len(games)
    resolved_games = int(resolved_mask.sum())
    analyzed_days = pd.to_datetime(games["game_date"], errors="coerce").nunique()

    consensus_mask = resolved_mask & games["consensus_pick"].isin(["OVER", "UNDER"])
    consensus_n = int(consensus_mask.sum())
    consensus_acc = (
        (
            games.loc[consensus_mask, "consensus_pick"]
            == games.loc[consensus_mask, "actual_side"]
        ).mean()
        if consensus_n
        else np.nan
    )

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric("Games", total_games)
    with top_cols[1]:
        st.metric("Resolved", resolved_games)
    with top_cols[2]:
        st.metric("Days", int(analyzed_days))
    with top_cols[3]:
        st.metric(
            "Weighted Consensus",
            f"{(consensus_acc * 100):.2f}%" if pd.notna(consensus_acc) else "N/A",
        )

    st.markdown("### 💰 Overall Model Statistics")
    st.markdown("")
    summary_df = summarize_model_performance(games)
    st.dataframe(summary_df, width="stretch", hide_index=True)

    st.markdown("")
    st.markdown("### 🎯 Accuracy by |Diff vs O/U Line|")
    st.caption("All models: thresholds at >=0, >=0.5, >=1, >=1.5, >=2.")
    threshold_df = compute_threshold_accuracy_table(games)
    st.dataframe(
        threshold_df,
        width="stretch",
        hide_index=True,
        height=240,
    )

    st.markdown("---")
    st.markdown("### 📅 Daily Accuracy")
    daily = compute_daily_metrics(games)

    if daily.empty:
        st.warning("No daily metrics available.")
        return

    accuracy_pivot = (
        daily.pivot(index="game_date", columns="model_label", values="accuracy")
        .sort_index()
        .mul(100)
    )

    st.dataframe(
        accuracy_pivot.reset_index().rename(columns={"game_date": "Date"}).round(2),
        width="stretch",
        hide_index=True,
        height=350,
    )

    smooth_window = st.slider("Smoothing window (days)", 1, 14, 1)

    fig_acc, ax_acc = plt.subplots(figsize=(14, 6), dpi=140)
    ax_acc.set_facecolor("white")
    ax_acc.grid(True, alpha=0.2)

    for model_label in accuracy_pivot.columns:
        y = accuracy_pivot[model_label]
        if smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1).mean()
        ax_acc.plot(accuracy_pivot.index, y, linewidth=2.5, label=model_label)
        ax_acc.scatter(
            accuracy_pivot.index, accuracy_pivot[model_label], s=16, alpha=0.25
        )

    ax_acc.axhline(50, linestyle="--", alpha=0.55, linewidth=1.3)
    ax_acc.set_title("Daily Accuracy by Model", fontsize=16, fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xlabel("Date")
    ax_acc.set_ylim(0, 100)
    ax_acc.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax_acc.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax_acc.xaxis.get_major_locator())
    )
    ax_acc.legend(frameon=False)
    fig_acc.tight_layout()
    st.pyplot(fig_acc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📉 Daily Mean Absolute Error")

    mae_pivot = daily.pivot(
        index="game_date", columns="model_label", values="mae"
    ).sort_index()

    fig_mae, ax_mae = plt.subplots(figsize=(14, 6), dpi=140)
    ax_mae.set_facecolor("white")
    ax_mae.grid(True, alpha=0.2)

    for model_label in mae_pivot.columns:
        y = mae_pivot[model_label]
        if smooth_window > 1:
            y = y.rolling(window=smooth_window, min_periods=1).mean()
        ax_mae.plot(mae_pivot.index, y, linewidth=2.5, label=model_label)
        ax_mae.scatter(mae_pivot.index, mae_pivot[model_label], s=16, alpha=0.25)

    ax_mae.set_title("Daily MAE by Model", fontsize=16, fontweight="bold")
    ax_mae.set_ylabel("MAE (points)")
    ax_mae.set_xlabel("Date")
    ax_mae.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    ax_mae.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax_mae.xaxis.get_major_locator())
    )
    ax_mae.legend(frameon=False)
    fig_mae.tight_layout()
    st.pyplot(fig_mae, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="NBA Over/Under Predictor",
        page_icon="🏀",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": None,
            "Report a bug": None,
            "About": "NBA Over/Under Predictor: upcoming predictions, past results, and historical performance.",
        },
    )

    set_runtime_env_from_secrets()
    inject_global_css()

    with st.sidebar:
        st.markdown("### NBA Predictor Menu")
        view_option = st.radio(
            label="Go to",
            options=[
                "Upcoming Predictions",
                "Past Games Results",
                "Historical Performance",
            ],
            index=0,
        )
        st.markdown("---")

    render_header()

    if view_option == "Upcoming Predictions":
        show_upcoming_predictions()
    elif view_option == "Past Games Results":
        show_past_games_results()
    else:
        show_historical_performance()


if __name__ == "__main__":
    main()

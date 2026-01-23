import os
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Suppress pandas SQLAlchemy warnings
# Import nba_predictor main function
from scripts.predict_nba_games import main as run_nba_predictor
from postgre_DB.update_evaluation_predictions import (
    add_ou_betting_metrics,
    compute_daily_accuracy,
    compute_daily_prediction_errors,
    compute_ou_betting_statistics,
    get_games_with_total_scored_points,
)
from utils.streamlit_utils import format_upcoming_games_display, render_game_cards

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")
os.environ["SUPABASE_DB_URL"] = st.secrets["DatabaseSupabase"]["SUPABASE_DB_URL"]
os.environ["SUPABASE_DB_PASSWORD"] = st.secrets["DatabaseSupabase"][
    "SUPABASE_DB_PASSWORD"
]
try:
    os.environ["ODDS_API_KEY"] = st.secrets["Odds"]["ODDS_API_KEY"]
except KeyError:
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

          /* DataFrame readability - keep your current style intent */
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
              <div style="font-size: 0.95rem; font-weight: 700; opacity: 0.85;">
                NBA analytics
              </div>
              <div style="margin-top: 2px;">
                <span style="font-size: 2.2rem; font-weight: 900; letter-spacing: -0.02em;">
                  Over/Under Predictor
                </span>
              </div>
              <div class="app-subtitle">
                Predictions, results, and historical performance in one place.
              </div>
              <div class="chip-row">
                <span class="chip">Model: regressor + classifier</span>
                <span class="chip">Lines: bookmaker O/U</span>
                <span class="chip">Timezone: Madrid (CEST)</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_password():
    """Returns True if the user has entered the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["App"]["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 1rem;">🏀 NBA Over/Under Predictor</h1>
            <p style="font-size: 1.2rem; opacity: 0.85;">Please enter the password to access the application</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Use a small form so pressing Enter or clicking the button both submit
        with st.form(key="login_form"):
            pwd = st.text_input(
                "Password",
                type="password",
                key="password_input",
                label_visibility="collapsed",
                placeholder="Enter password...",
            )
            submitted = st.form_submit_button("Enter")

            if submitted:
                st.session_state["password"] = pwd
                password_entered()

                if st.session_state.get("password_correct"):
                    st.rerun()

        if (
            "password_correct" in st.session_state
            and not st.session_state["password_correct"]
        ):
            st.error("😕 Password incorrect. Please try again.")

    return False


def main():
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

    inject_global_css()

    # Check password before showing content
    if not check_password():
        st.stop()

    # Sidebar navigation (cleaner than huge radios across the top)
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

    # Main header
    render_header()

    # Route to views
    if view_option == "Upcoming Predictions":
        show_upcoming_predictions()
    elif view_option == "Past Games Results":
        show_past_games_results()
    else:
        show_historical_performance()


def show_upcoming_predictions():
    """Display upcoming game predictions."""
    # Fetch upcoming games
    with st.spinner("Loading upcoming games..."):
        df_upcoming = get_games_with_total_scored_points(only_null=True)

    if df_upcoming.empty:
        st.info("No upcoming games with predictions available.")
        return

    # Get most recent prediction for each game
    df_upcoming = (
        df_upcoming.sort_values("prediction_date")
        .groupby("game_id", as_index=False)
        .tail(1)
    )

    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Upcoming Games", len(df_upcoming))
    with col2:
        n_agree = (
            df_upcoming["regressor_prediction"]
            == df_upcoming["classifier_prediction_model2"]
        ).sum()
        st.metric("Models Agree", f"{n_agree}/{len(df_upcoming)}")
    with col3:
        latest_pred = pd.to_datetime(df_upcoming["prediction_date"]).max()
        # Convert to Madrid timezone
        if latest_pred.tz is None:
            latest_pred = latest_pred.tz_localize("UTC")
        latest_pred_madrid = latest_pred.tz_convert("Europe/Madrid")
        st.metric("Latest Prediction", latest_pred_madrid.strftime("%Y-%m-%d %H:%M"))

    st.markdown("---")

    # Add button to run predictor
    st.markdown("### 🔄 Update Predictions")
    st.caption(
        "Run the prediction model to generate fresh predictions for today's games."
    )
    st.markdown("")

    if run_nba_predictor is None:
        st.warning(
            "⚠️ NBA Predictor module not available. Please check your installation."
        )
    else:
        if st.button("🚀 Run Predictor Now", type="primary", width="stretch"):
            try:
                # Save original sys.argv
                original_argv = sys.argv.copy()

                # Set sys.argv to simulate command-line call without saving Excel
                sys.argv = ["streamlit_app.py"]

                with st.spinner(
                    "🔄 Running NBA predictor... This may take a few minutes."
                ):
                    # Run the predictor
                    result = run_nba_predictor()

                # Restore original sys.argv
                sys.argv = original_argv

                if result == 0:
                    st.success("✅ Predictions updated successfully! Reloading page...")
                    import time

                    time.sleep(2)  # Brief pause to show success message
                    st.rerun()
                else:
                    st.error(
                        "❌ Predictor completed with errors. Please check the logs."
                    )

            except Exception as e:
                # Restore original sys.argv in case of error
                sys.argv = original_argv
                st.error(f"❌ Error running predictor: {str(e)}")
                st.exception(e)

    st.markdown("---")
    st.markdown("## Today's Predictions")
    st.caption("View upcoming games with AI-powered over/under predictions.")
    st.markdown("")

    # Add a toggle for display mode
    use_cards = st.toggle("Use Card View", value=True)

    if use_cards:
        # Render games as cards with logos
        render_game_cards(df_upcoming)
    else:
        # Fallback to table view
        display_df = format_upcoming_games_display(df_upcoming)
        st.dataframe(
            display_df,
            width="stretch",
            height=600,
            hide_index=True,
        )

    # Add footer with explanation
    st.markdown("---")
    st.markdown("")
    with st.expander("ℹ️ **How to Read the Predictions**", expanded=False):
        st.markdown("""
        ### 📊 Understanding the Predictions
        
        - **🏀 Matchup**: Home team vs Away team with logos
        - **⏰ Game Time**: When the game starts (Madrid timezone)
        - **📏 O/U Line**: The bookmaker's over/under betting line
        - **🎯 Predicted Total**: Our AI model's prediction for total points scored
        - **📈 Margin**: How far our prediction is from the bookmaker's line
          - *Negative value* = We predict UNDER
          - *Positive value* = We predict OVER
        - **🤖 Regressor**: Prediction from our regression model
        - **🧠 Classifier**: Prediction from our classification model
        - **✅ Agreement**: Shows if both models agree on the prediction
        - **Both Agree**: ✅ if both models agree, ❌ if they disagree
        - **Over Odds / Under Odds**: Decimal odds for betting
        - **Time to Game**: Minutes until game starts
        
        **Note**: Predictions are updated periodically. Most recent prediction time shown above.
        """)


def show_past_games_results():
    """Display past game predictions with actual results."""
    st.markdown("## Past Games Results")
    st.caption("Compare predictions vs actual totals for a selected date.")
    st.markdown("")

    # Date selector - default to yesterday
    from datetime import datetime, timedelta

    yesterday = datetime.now() - timedelta(days=1)

    selected_date = st.date_input(
        "Select Date:",
        value=yesterday,
        help="Choose a date to see predictions and actual results",
    )

    st.markdown("---")

    # Load data for selected date
    with st.spinner("Loading games for selected date..."):
        date_str = selected_date.strftime("%Y-%m-%d")
        df_games = get_games_with_total_scored_points(
            only_null=False,
            date=date_str,
        )

    if df_games.empty:
        st.warning(f"No completed games found for {date_str}.")
        return

    # Get unique prediction times available for this date
    df_games["prediction_datetime"] = pd.to_datetime(df_games["prediction_date"])

    # Normalize to UTC for consistent comparison
    if df_games["prediction_datetime"].dt.tz is None:
        df_games["prediction_datetime"] = df_games[
            "prediction_datetime"
        ].dt.tz_localize("UTC")
    else:
        df_games["prediction_datetime"] = df_games["prediction_datetime"].dt.tz_convert(
            "UTC"
        )

    unique_prediction_times = sorted(
        df_games["prediction_datetime"].unique(), reverse=True
    )

    if len(unique_prediction_times) == 0:
        st.warning(f"No predictions found for {date_str}.")
        return

    # Add prediction time selector
    st.markdown("### ⏰ Select Prediction Time")
    st.caption("Choose which prediction time to analyze (most recent is default).")

    # Format prediction times for display and create mapping
    prediction_time_mapping = {}
    prediction_time_options = []

    for pred_time in unique_prediction_times:
        pred_time_madrid = pred_time.tz_convert("Europe/Madrid")
        display_str = pred_time_madrid.strftime("%Y-%m-%d %H:%M:%S")
        prediction_time_options.append(display_str)
        prediction_time_mapping[display_str] = pred_time

    selected_prediction_time = st.selectbox(
        "Prediction Time:",
        options=prediction_time_options,
        index=0,  # Default to most recent (first in list since sorted reverse=True)
        help="Select which prediction to analyze. The most recent prediction is selected by default.",
    )

    # Get the actual UTC timestamp for filtering
    selected_pred_dt = prediction_time_mapping[selected_prediction_time]

    # Filter to keep predictions at or before the selected time
    # For each game, use the selected time if available, otherwise use the closest earlier prediction
    df_games = df_games[df_games["prediction_datetime"] <= selected_pred_dt].copy()

    # For each game, keep only the most recent prediction (at or before selected time)
    df_games = (
        df_games.sort_values("prediction_datetime", ascending=False)
        .groupby("game_id", as_index=False)
        .first()
    )

    # Add betting metrics to get correctness
    df_with_metrics = add_ou_betting_metrics(df_games)

    # Display summary metrics
    n_games = len(df_with_metrics)
    reg_correct = df_with_metrics["regressor_correct"].sum()
    clf_correct = df_with_metrics["classifier_correct"].sum()
    both_correct = df_with_metrics["both_agree_correct"].sum()
    both_agree_count = df_with_metrics["both_agree"].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎮 Games Played", n_games)
    with col2:
        st.metric("🤖 Regressor Correct", f"{reg_correct}/{n_games}")
    with col3:
        st.metric("🧠 Classifier Correct", f"{clf_correct}/{n_games}")
    with col4:
        st.metric("✅ Both Agree Correct", f"{both_correct}/{both_agree_count}")

    st.markdown("---")
    st.markdown(f"### 🏀 Games on {date_str}")
    st.markdown("")

    # Add toggle for card/table view
    use_cards = st.toggle("Use Card View", value=True, key="past_games_toggle")

    if use_cards:
        # Render games as cards with results
        render_past_game_cards(df_with_metrics)
    else:
        # Fallback to table view
        display_df = format_past_games_display(df_with_metrics)
        st.dataframe(
            display_df,
            width="stretch",
            height=600,
            hide_index=True,
        )


def render_past_game_cards(df: pd.DataFrame):
    """Render past games as cards showing predictions vs actual results."""
    # Sort by game time
    df = df.sort_values("game_time").reset_index(drop=True)

    # Create two-column layout
    cols_per_row = 2

    for idx in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)

        for col_idx, col in enumerate(cols):
            row_idx = idx + col_idx
            if row_idx >= len(df):
                break

            row = df.iloc[row_idx]

            with col:
                home_team = row["team_name_team_home"]
                away_team = row["team_name_team_away"]
                # Convert to Madrid timezone
                game_dt = pd.to_datetime(row["game_time"])
                if game_dt.tz is None:
                    game_dt = game_dt.tz_localize("UTC")
                game_dt_madrid = game_dt.tz_convert("Europe/Madrid")
                game_time = game_dt_madrid.strftime("%I:%M %p")
                game_date = game_dt_madrid.strftime("%b %d, %Y")

                # Get predictions and actual
                regressor_pred = row["regressor_prediction"]
                classifier_pred = row["classifier_prediction_model2"]
                both_agree = row["both_agree"]

                regressor_correct = row["regressor_correct"]
                classifier_correct = row["classifier_correct"]

                # Actual results
                actual_total = row["total_scored_points"]
                ou_line = row["total_over_under_line"]
                actual_side = row["actual_side"]

                predicted_total = row["predicted_total_score"]
                margin = row["margin_difference_prediction_vs_over_under"]
                over_odds = row["average_total_over_money"]
                under_odds = row["average_total_under_money"]

                # Calculate prediction error (actual - predicted)
                prediction_error = actual_total - predicted_total

                # Determine styling based on correctness
                if regressor_correct and classifier_correct:
                    border_color = "#4CAF50"  # Green - both correct
                elif regressor_correct or classifier_correct:
                    border_color = "#FFA500"  # Orange - one correct
                else:
                    border_color = "#F44336"  # Red - both wrong

                # Create card header
                header_html = f"""
                <div style="
                    border: 2px solid {border_color};
                    border-radius: 12px 12px 0 0;
                    overflow: hidden;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.15);
                ">
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        color: white;
                    ">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="flex: 1; text-align: center;">
                                <img src="{st.session_state.get('logo_' + home_team, f'https://a.espncdn.com/i/teamlogos/nba/500/1.png')}" width="100" style="margin-bottom: 8px;" onerror="this.style.display='none'">
                                <div style="font-size: 1.1rem; font-weight: 700;">{home_team}</div>
                            </div>
                            
                            <div style="flex: 0.4; text-align: center;">
                                <div style="font-size: 2rem; font-weight: 900; margin-bottom: 5px;">VS</div>
                                <div style="font-size: 0.95rem; font-weight: 600;">{game_date}</div>
                                <div style="font-size: 1.1rem; font-weight: 700; margin-top: 3px;">🕐 {game_time}</div>
                            </div>
                            
                            <div style="flex: 1; text-align: center;">
                                <img src="{st.session_state.get('logo_' + away_team, f'https://a.espncdn.com/i/teamlogos/nba/500/1.png')}" width="100" style="margin-bottom: 8px;" onerror="this.style.display='none'">
                                <div style="font-size: 1.1rem; font-weight: 700;">{away_team}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """

                from utils.streamlit_utils import get_team_logo_url

                header_html = f"""
                <div style="
                    border: 2px solid {border_color};
                    border-radius: 12px 12px 0 0;
                    overflow: hidden;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.15);
                ">
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        color: white;
                    ">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="flex: 1; text-align: center;">
                                <img src="{get_team_logo_url(home_team)}" width="100" style="margin-bottom: 8px;">
                                <div style="font-size: 1.1rem; font-weight: 700;">{home_team}</div>
                            </div>
                            
                            <div style="flex: 0.4; text-align: center;">
                                <div style="font-size: 2rem; font-weight: 900; margin-bottom: 5px;">VS</div>
                                <div style="font-size: 0.95rem; font-weight: 600;">{game_date}</div>
                                <div style="font-size: 1.1rem; font-weight: 700; margin-top: 3px;">🕐 {game_time}</div>
                            </div>
                            
                            <div style="flex: 1; text-align: center;">
                                <img src="{get_team_logo_url(away_team)}" width="100" style="margin-bottom: 8px;">
                                <div style="font-size: 1.1rem; font-weight: 700;">{away_team}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """

                st.components.v1.html(header_html, height=200)

                # Stats section
                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 2px solid {border_color}; border-top: none; 
                             border-radius: 0 0 12px 12px; padding: 15px; margin-top: -5px; 
                             background: white;">
                        """,
                        unsafe_allow_html=True,
                    )

                    # Actual result indicator
                    actual_color = "#2196F3" if actual_side == "under" else "#FF5722"
                    st.markdown(
                        f"""
                        <div style="text-align: center; margin-bottom: 10px; padding: 10px; 
                             background: {actual_color}; color: white; border-radius: 8px;">
                            <span style="font-size: 1.3rem; font-weight: 700;">
                                ⚡ ACTUAL: {actual_side.upper() if actual_side else 'N/A'} ({actual_total:.1f} points)
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Predictions row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("O/U Line", f"{ou_line:.1f}")
                    with col2:
                        st.metric("Predicted", f"{predicted_total:.1f}")
                    with col3:
                        st.metric("Actual", f"{actual_total:.1f}")
                    with col4:
                        error_color = (
                            "normal" if abs(prediction_error) <= 5 else "inverse"
                        )
                        st.metric(
                            "Pred. Error",
                            f"{prediction_error:+.1f}",
                            delta=None,
                            delta_color=error_color,
                        )

                    # Model predictions with correctness
                    col1, col2 = st.columns(2)
                    with col1:
                        reg_icon = "🔵" if regressor_pred == "Under" else "🔴"
                        reg_status = "✅" if regressor_correct else "❌"
                        st.metric(
                            "🤖 Regressor", f"{reg_icon} {regressor_pred} {reg_status}"
                        )
                    with col2:
                        clf_icon = "🔵" if classifier_pred == "Under" else "🔴"
                        clf_status = "✅" if classifier_correct else "❌"
                        st.metric(
                            "🧠 Classifier",
                            f"{clf_icon} {classifier_pred} {clf_status}",
                        )

                    # Odds
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Over Odds", f"{over_odds:.2f}")
                    with col2:
                        st.metric("Under Odds", f"{under_odds:.2f}")

                    # Close div
                    st.markdown("</div>", unsafe_allow_html=True)


def format_past_games_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format past games dataframe for table display."""
    display_df = pd.DataFrame()

    display_df["Matchup"] = (
        df["team_name_team_home"] + " vs " + df["team_name_team_away"]
    )
    # Convert to Madrid timezone
    game_times = pd.to_datetime(df["game_time"])
    if game_times.dt.tz is None:
        game_times = game_times.dt.tz_localize("UTC")
    display_df["Game Time"] = game_times.dt.tz_convert("Europe/Madrid").dt.strftime(
        "%H:%M"
    )
    display_df["O/U Line"] = df["total_over_under_line"].round(1)
    display_df["Predicted"] = df["predicted_total_score"].round(1)
    display_df["Actual"] = df["total_scored_points"].round(1)
    display_df["Pred. Error"] = (
        df["total_scored_points"] - df["predicted_total_score"]
    ).round(1)
    display_df["Actual Side"] = df["actual_side"].str.upper()
    display_df["Regressor"] = (
        df["regressor_prediction"]
        + " "
        + df["regressor_correct"].map({True: "✅", False: "❌"})
    )
    display_df["Classifier"] = (
        df["classifier_prediction_model2"]
        + " "
        + df["classifier_correct"].map({True: "✅", False: "❌"})
    )
    display_df["Over Odds"] = df["average_total_over_money"].round(2)
    display_df["Under Odds"] = df["average_total_under_money"].round(2)

    return display_df


def show_historical_performance():
    """Display historical betting performance analysis."""
    st.markdown("## Historical Betting Performance")
    st.caption("Analyze prediction accuracy and profitability over time.")
    st.markdown("")

    # Optional date filter in an expander
    use_date_filter = st.checkbox("🗓️ Filter by Date Range", value=False)

    start_date = None
    end_date = None

    if use_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime("2026-01-01"),
                help="Select the start date for analysis",
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime("today"),
                help="Select the end date for analysis",
            )

    st.markdown("---")

    # Load data automatically
    with st.spinner("Loading historical data and computing statistics..."):
        # Fetch past games
        df_past = get_games_with_total_scored_points(
            only_null=False,
            start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
            end_date=end_date.strftime("%Y-%m-%d") if end_date else None,
        )

        if df_past.empty:
            st.warning("No historical data found for the selected date range.")
            return

        # Keep only the most recent prediction for each game_id
        df_past = (
            df_past.sort_values("prediction_date")
            .groupby("game_id", as_index=False)
            .tail(1)
        )

        # Add betting metrics
        df_with_metrics = add_ou_betting_metrics(df_past)

        # Display overall statistics
        st.markdown("### 💰 Overall Betting Statistics")
        st.markdown("")

        # Compute stats but don't print (we'll display in Streamlit)
        stats = compute_ou_betting_statistics(df_with_metrics, print_report=False)

        # Extract stats
        row = stats.iloc[0]
        n_games = int(row["n_games_total"])
        n_resolved = int(row["n_resolved"])
        n_days = int(row["n_days"])
        mean_error = row["mean_prediction_error"]
        mean_abs_error = row["mean_abs_prediction_error"]

        # Display summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Total Games", n_games)
        with col2:
            st.metric("✅ Resolved Games", n_resolved)
        with col3:
            st.metric("📅 Days Analyzed", n_days)
        with col4:
            st.metric(
                "📏 Mean Error (pts)",
                f"{mean_error:+.2f}" if not pd.isna(mean_error) else "N/A",
            )
        with col5:
            st.metric(
                "📐 Mean Abs Error (pts)",
                f"{mean_abs_error:.2f}" if not pd.isna(mean_abs_error) else "N/A",
            )

        st.markdown("")

        # Create comparison table with larger styling
        comparison_data = {
            "🎯 Strategy": ["🤖 Regressor", "🧠 Classifier", "✅ Both Agree"],
            "Bets": [
                int(row["regressor_n_bets"]),
                int(row["classifier_n_bets"]),
                int(row["both_agree_n_bets"]),
            ],
            "Accuracy (%)": [
                f"{row['regressor_accuracy']*100:.2f}",
                f"{row['classifier_accuracy']*100:.2f}",
                f"{row['both_agree_accuracy']*100:.2f}",
            ],
            "Total Profit (€)": [
                f"{row['regressor_total_profit']:.2f}",
                f"{row['classifier_total_profit']:.2f}",
                f"{row['both_agree_total_profit']:.2f}",
            ],
            "Avg Profit/Bet (€)": [
                f"{row['regressor_avg_profit_per_bet']:.2f}",
                f"{row['classifier_avg_profit_per_bet']:.2f}",
                f"{row['both_agree_avg_profit_per_bet']:.2f}",
            ],
            "Avg Profit/Day (€)": [
                f"{row['regressor_avg_profit_per_day']:.2f}",
                f"{row['classifier_avg_profit_per_day']:.2f}",
                f"{row['both_agree_avg_profit_per_day']:.2f}",
            ],
            "Avg Stake/Day (€)": [
                f"{row['regressor_avg_stake_per_day']:.2f}",
                f"{row['classifier_avg_stake_per_day']:.2f}",
                f"{row['both_agree_avg_stake_per_day']:.2f}",
            ],
        }

        comparison_df = pd.DataFrame(comparison_data)

        # Display with larger column config
        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True,
            height=200,
        )

        st.markdown("")
        st.info(
            "💡 **Note**: Accuracy = correct predictions / resolved games (excluding pushes). "
            "Profit assumes 1€ stake per bet. Win profit = (odds - 1), Loss = -1€."
        )

        # Daily accuracy analysis
        st.markdown("---")
        st.markdown("### 📅 Daily Accuracy Breakdown")
        st.markdown("")

        daily_accuracy = compute_daily_accuracy(df_with_metrics)

        if not daily_accuracy.empty:
            # Format daily accuracy for display
            daily_display = daily_accuracy.copy()
            daily_display["regressor_accuracy"] = (
                daily_display["regressor_accuracy"] * 100
            ).round(2)
            daily_display["classifier_accuracy"] = (
                daily_display["classifier_accuracy"] * 100
            ).round(2)
            daily_display["both_agree_accuracy"] = (
                daily_display["both_agree_accuracy"] * 100
            ).round(2)

            # Rename columns for display with emojis
            daily_display.columns = [
                "📅 Date",
                "🎮 Games",
                "🤖 Regressor (%)",
                "🧠 Classifier (%)",
                "✅ Both Agree (%)",
            ]

            st.dataframe(
                daily_display,
                width="stretch",
                hide_index=True,
                height=500,
            )

            # Plot daily accuracy
            st.markdown("---")
            st.markdown("### 📈 Daily Accuracy Chart")

            st.markdown("")

            # --- Prep ---
            df = daily_accuracy.copy()
            df["game_date"] = pd.to_datetime(df["game_date"])
            df = df.sort_values("game_date")

            # Optional: choose smoothing window (in days/points)
            smooth_window = st.slider(
                "Smoothing window (rolling average of days)", 1, 14, 1
            )
            use_smoothing = smooth_window > 1

            series_cols = {
                "Regressor": "regressor_accuracy",
                "Classifier": "classifier_accuracy",
                "Both Agree": "both_agree_accuracy",
            }

            # --- Figure ---
            fig, ax = plt.subplots(figsize=(14, 7), dpi=140)

            # Subtle styling
            ax.set_facecolor("white")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            ax.grid(True, which="major", axis="both", alpha=0.18, linewidth=1)
            ax.grid(True, which="minor", axis="y", alpha=0.08, linewidth=0.8)

            dates = df["game_date"]

            for label, col in series_cols.items():
                y = df[col].astype(float) * 100.0

                if use_smoothing:
                    y_smooth = y.rolling(window=smooth_window, min_periods=1).mean()
                    ax.plot(
                        dates,
                        y_smooth,
                        label=f"{label} (smoothed)",
                        linewidth=2.6,
                    )
                    # Optional: show raw points faintly for context
                    ax.scatter(dates, y, s=18, alpha=0.25)

                else:
                    ax.plot(dates, y, label=label, linewidth=2.6)
                    ax.scatter(dates, y, s=26, alpha=0.9)

            # 50% reference line
            ax.axhline(50, linestyle="--", alpha=0.6, linewidth=1.4)
            ax.text(
                dates.iloc[0],
                50.8,
                "50% break-even",
                fontsize=11,
                alpha=0.75,
                va="bottom",
            )

            # Labels / title
            ax.set_title(
                "Daily Prediction Accuracy by Strategy",
                fontsize=18,
                fontweight="bold",
                pad=14,
            )
            ax.set_xlabel("Date", fontsize=13, fontweight="bold")
            ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
            ax.set_ylim(0, 100)

            # Date axis: fewer, smarter ticks
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
            )
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

            # Minor ticks on Y for nicer grid
            ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

            ax.legend(frameon=False, fontsize=11, ncol=2)

            fig.tight_layout()

            # Streamlit rendering: responsive + crisp
            st.pyplot(fig, width="stretch")

        else:
            st.warning("No daily accuracy data available for the selected range.")

        # Prediction Error Analysis
        st.markdown("---")
        st.markdown("### 📉 Daily Prediction Error Analysis")
        st.markdown("")
        st.caption(
            "Analyze how far off our predictions were from actual game totals over time."
        )
        st.markdown("")

        daily_errors = compute_daily_prediction_errors(df_with_metrics)

        if not daily_errors.empty:
            # Display error statistics
            col1, col2 = st.columns(2)
            with col1:
                overall_mean_error = daily_errors["mean_error"].mean()
                st.metric("Overall Mean Error", f"{overall_mean_error:+.2f} pts")
            with col2:
                overall_mean_abs_error = daily_errors["mean_abs_error"].mean()
                st.metric("Overall Mean Abs Error", f"{overall_mean_abs_error:.2f} pts")

            st.markdown("")
            st.info(
                "💡 **Mean Error**: Average difference (predicted - actual). Positive = predicting too high, Negative = predicting too low.\\n\\n"
                "**Mean Absolute Error**: Average magnitude of errors regardless of direction."
            )

            # Smoothing option
            st.markdown("")
            smooth_window_errors = st.slider(
                "Smoothing window for error chart (rolling average of days)",
                1,
                14,
                2,
                key="error_smooth_slider",
            )
            use_smoothing_errors = smooth_window_errors > 1

            # --- Figure ---
            fig_errors, ax_errors = plt.subplots(figsize=(14, 7), dpi=140)

            # Styling
            ax_errors.set_facecolor("white")
            for spine in ("top", "right"):
                ax_errors.spines[spine].set_visible(False)

            ax_errors.grid(True, which="major", axis="both", alpha=0.18, linewidth=1)
            ax_errors.grid(True, which="minor", axis="y", alpha=0.08, linewidth=0.8)

            dates = pd.to_datetime(daily_errors["game_date"])
            daily_errors = daily_errors.sort_values("game_date")

            # Plot mean error (can be positive or negative)
            if use_smoothing_errors:
                mean_error_smooth = (
                    daily_errors["mean_error"]
                    .rolling(window=smooth_window_errors, min_periods=1)
                    .mean()
                )
                ax_errors.plot(
                    dates,
                    mean_error_smooth,
                    color="#667eea",
                    linewidth=2.5,
                    label=f"Mean Error ({smooth_window_errors}-day avg)",
                    alpha=0.9,
                )
            else:
                ax_errors.plot(
                    dates,
                    daily_errors["mean_error"],
                    color="#667eea",
                    linewidth=2,
                    marker="o",
                    markersize=5,
                    label="Mean Error",
                    alpha=0.8,
                )

            # Plot mean absolute error (always positive)
            if use_smoothing_errors:
                mean_abs_error_smooth = (
                    daily_errors["mean_abs_error"]
                    .rolling(window=smooth_window_errors, min_periods=1)
                    .mean()
                )
                ax_errors.plot(
                    dates,
                    mean_abs_error_smooth,
                    color="#FF6B6B",
                    linewidth=2.5,
                    label=f"Mean Abs Error ({smooth_window_errors}-day avg)",
                    alpha=0.9,
                )
            else:
                ax_errors.plot(
                    dates,
                    daily_errors["mean_abs_error"],
                    color="#FF6B6B",
                    linewidth=2,
                    marker="s",
                    markersize=5,
                    label="Mean Abs Error",
                    alpha=0.8,
                )

            # Zero reference line
            ax_errors.axhline(0, linestyle="--", alpha=0.6, linewidth=1.4, color="gray")
            ax_errors.text(
                dates.iloc[0],
                0.5,
                "Perfect prediction",
                fontsize=11,
                alpha=0.75,
                va="bottom",
            )

            # Labels / title
            ax_errors.set_title(
                "Daily Prediction Error (Predicted - Actual Points)",
                fontsize=18,
                fontweight="bold",
                pad=14,
            )
            ax_errors.set_xlabel("Date", fontsize=13, fontweight="bold")
            ax_errors.set_ylabel("Error (points)", fontsize=13, fontweight="bold")

            # Date axis formatting
            ax_errors.xaxis.set_major_locator(
                mdates.AutoDateLocator(minticks=5, maxticks=9)
            )
            ax_errors.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax_errors.xaxis.get_major_locator())
            )
            plt.setp(ax_errors.get_xticklabels(), rotation=0, ha="center")

            # Minor ticks on Y for nicer grid
            ax_errors.yaxis.set_minor_locator(plt.MultipleLocator(1))

            ax_errors.legend(frameon=False, fontsize=11, ncol=2)

            fig_errors.tight_layout()

            # Streamlit rendering
            st.pyplot(fig_errors, width="stretch")

        else:
            st.warning("No prediction error data available for the selected range.")


if __name__ == "__main__":
    main()

import os
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Suppress pandas SQLAlchemy warnings
# Import nba_predictor main function
# try:
from nba_predictor import main as run_nba_predictor
from postgre_DB.update_evaluation_predictions import (
    add_ou_betting_metrics,
    compute_daily_accuracy,
    compute_ou_betting_statistics,
    get_games_with_total_scored_points,
)
from utils.streamlit_utils import format_upcoming_games_display, render_game_cards

# except (ImportError, KeyError, Exception):
#     # Catch ImportError, KeyError (from import cache issues), and any other import-time errors
#     run_nba_predictor = None
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
                # Mirror into the expected session key and validate
                st.session_state["password"] = pwd
                password_entered()

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

    # Keep only the most recent prediction for each game
    df_games = (
        df_games.sort_values("prediction_date")
        .groupby("game_id", as_index=False)
        .tail(1)
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
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("O/U Line", f"{ou_line:.1f}")
                    with col2:
                        st.metric("Predicted", f"{predicted_total:.1f}")
                    with col3:
                        st.metric("Margin", f"{margin:+.1f}")

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

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Games", n_games)
        with col2:
            st.metric("✅ Resolved Games", n_resolved)
        with col3:
            st.metric("📅 Days Analyzed", n_days)

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

            fig, ax = plt.subplots(figsize=(14, 8))

            # Convert game_date to datetime
            dates = pd.to_datetime(daily_accuracy["game_date"])

            # Plot each strategy
            ax.plot(
                dates,
                daily_accuracy["regressor_accuracy"] * 100,
                marker="o",
                label="Regressor",
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                dates,
                daily_accuracy["classifier_accuracy"] * 100,
                marker="s",
                label="Classifier",
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                dates,
                daily_accuracy["both_agree_accuracy"] * 100,
                marker="^",
                label="Both Agree",
                linewidth=2,
                markersize=6,
            )

            # Add 50% reference line
            ax.axhline(
                y=50,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="50% (Break-even)",
            )

            # Formatting with larger fonts
            ax.set_xlabel("Date", fontsize=16, fontweight="bold")
            ax.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold")
            ax.set_title(
                "Daily Prediction Accuracy by Strategy",
                fontsize=20,
                fontweight="bold",
            )
            ax.legend(loc="best", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=12)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45, ha="right")

            # Set y-axis range
            ax.set_ylim(0, 100)

            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.warning("No daily accuracy data available for the selected range.")


if __name__ == "__main__":
    main()

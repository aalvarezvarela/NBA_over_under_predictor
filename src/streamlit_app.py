import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from postgre_DB.update_evaluation_predictions import (
    add_ou_betting_metrics,
    compute_daily_accuracy,
    compute_ou_betting_statistics,
    get_games_with_total_scored_points,
)
from utils.streamlit_utils import format_upcoming_games_display, render_game_cards


def main():
    st.set_page_config(
        page_title="NBA Over/Under Predictor", page_icon="🏀", layout="wide"
    )

    # Custom CSS for larger fonts and better styling
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.5rem !important;
        }
        h2 {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
        }
        h3 {
            font-size: 2rem !important;
            font-weight: 600 !important;
        }
        .stMetric label {
            font-size: 1.3rem !important;
            font-weight: 600 !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stDataFrame"] {
            font-size: 1.5rem !important;
        }
        div[data-testid="stDataFrame"] div[role="gridcell"] {
            font-size: 1.5rem !important;
            padding: 0.8rem !important;
        }
        div[data-testid="stDataFrame"] div[role="columnheader"] {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            padding: 1rem !important;
        }
        /* Radio buttons styling - much larger */
        div[role="radiogroup"] label {
            font-size: 2.8rem !important;
            font-weight: 700 !important;
            padding: 2rem 3rem !important;
            margin: 1rem !important;
        }
        div[role="radiogroup"] {
            gap: 3rem !important;
        }
        div[role="radiogroup"] label div {
            font-size: 2.8rem !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🏀 NBA Over/Under Predictions")
    st.markdown("---")

    # Add navigation tabs
    view_option = st.radio(
        "Select View:",
        ["📅 Upcoming Predictions", "📊 Historical Performance"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    if view_option == "📅 Upcoming Predictions":
        show_upcoming_predictions()
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
        st.metric("Latest Prediction", latest_pred.strftime("%Y-%m-%d %H:%M"))

    st.markdown("---")
    st.markdown("## 📅 Today's Predictions")
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
        - **⏰ Game Time**: When the game starts (UTC timezone)
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


def show_historical_performance():
    """Display historical betting performance analysis."""
    st.markdown("## 📊 Historical Betting Performance")
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

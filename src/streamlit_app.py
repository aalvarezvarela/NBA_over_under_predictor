import pandas as pd
import streamlit as st
from postgre_DB.update_evaluation_predictions import get_games_with_total_scored_points
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
            font-size: 1.1rem !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🏀 NBA Over/Under Predictions")
    st.markdown("---")

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
            use_container_width=True,
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


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import streamlit as st

# NBA team IDs used by ESPN (30 teams) - mapping team names to ESPN IDs
ESPN_NBA_TEAM_IDS = {
    "Atlanta Hawks": 1,
    "Boston Celtics": 2,
    "Brooklyn Nets": 17,
    "Charlotte Hornets": 30,
    "Chicago Bulls": 4,
    "Cleveland Cavaliers": 5,
    "Dallas Mavericks": 6,
    "Denver Nuggets": 7,
    "Detroit Pistons": 8,
    "Golden State Warriors": 9,
    "Houston Rockets": 10,
    "Indiana Pacers": 11,
    "LA Clippers": 12,
    "Los Angeles Clippers": 12,
    "Los Angeles Lakers": 13,
    "Memphis Grizzlies": 29,
    "Miami Heat": 14,
    "Milwaukee Bucks": 15,
    "Minnesota Timberwolves": 16,
    "New Orleans Pelicans": 3,
    "New York Knicks": 18,
    "Oklahoma City Thunder": 25,
    "Orlando Magic": 19,
    "Philadelphia 76ers": 20,
    "Phoenix Suns": 21,
    "Portland Trail Blazers": 22,
    "Sacramento Kings": 23,
    "San Antonio Spurs": 24,
    "Toronto Raptors": 28,
    "Utah Jazz": 26,
    "Washington Wizards": 27,
}


def get_team_logo_url(team_name: str) -> str:
    """Get NBA team logo URL from ESPN CDN."""
    team_id = ESPN_NBA_TEAM_IDS.get(team_name, 1)  # Default to Hawks if not found
    return f"https://a.espncdn.com/i/teamlogos/nba/500/{team_id}.png"


def render_game_cards(df: pd.DataFrame) -> None:
    """
    Render games as nice cards with team logos.

    Args:
        df: DataFrame with upcoming game predictions
    """
    # Sort by game time
    df = df.sort_values("game_time").reset_index(drop=True)

    for idx, row in df.iterrows():
        home_team = row["team_name_team_home"]
        away_team = row["team_name_team_away"]
        game_time = pd.to_datetime(row["game_time"]).strftime(
            "%b %d, %Y at %I:%M %p UTC"
        )

        # Get predictions
        regressor_pred = row["regressor_prediction"]
        classifier_pred = row["classifier_prediction_model2"]
        both_agree = regressor_pred == classifier_pred

        # Calculate values
        ou_line = row["total_over_under_line"]
        predicted_total = row["predicted_total_score"]
        margin = row["margin_difference_prediction_vs_over_under"]
        over_odds = row["average_total_over_money"]
        under_odds = row["average_total_under_money"]
        time_to_game = int(row["time_to_match_minutes"])

        # Determine styling based on agreement
        if both_agree:
            border_color = "#4CAF50"
            bg_color = "#E8F5E9"
            agree_msg = "✅ BOTH MODELS AGREE!"
            agree_bg = "#4CAF50"
        else:
            border_color = "#FF9800"
            bg_color = "#FFF3E0"
            agree_msg = "⚠️ MODELS DISAGREE"
            agree_bg = "#FF9800"

        # Create card container
        with st.container():
            # Card styling
            st.markdown(
                f"""
                <div style="
                    border: 3px solid {border_color};
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 25px;
                    background-color: {bg_color};
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                """,
                unsafe_allow_html=True,
            )

            # Team matchup with logos
            col1, col2, col3 = st.columns([1, 0.3, 1])

            with col1:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{get_team_logo_url(home_team)}" width="80">
                        <div style="font-size: 1.4rem; font-weight: 700; margin-top: 10px;">{home_team}</div>
                        <div style="font-size: 1.1rem; color: #666;">HOME</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="font-size: 2.5rem; font-weight: 800; color: #1976D2;">VS</div>
                        <div style="font-size: 1rem; color: #666; margin-top: 10px;">{game_time}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{get_team_logo_url(away_team)}" width="80">
                        <div style="font-size: 1.4rem; font-weight: 700; margin-top: 10px;">{away_team}</div>
                        <div style="font-size: 1.1rem; color: #666;">AWAY</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

            # Stats row 1: Line, Predicted, Margin
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📏 O/U LINE", f"{ou_line:.1f}")
            with col2:
                st.metric("🎯 PREDICTED", f"{predicted_total:.1f}")
            with col3:
                st.metric("📈 MARGIN", f"{margin:+.1f}")

            # Stats row 2: Model predictions
            col1, col2 = st.columns(2)
            with col1:
                reg_color = "🔵" if regressor_pred == "Under" else "🔴"
                st.metric("🤖 REGRESSOR", f"{reg_color} {regressor_pred.upper()}")
            with col2:
                clf_color = "🔵" if classifier_pred == "Under" else "🔴"
                st.metric("🧠 CLASSIFIER", f"{clf_color} {classifier_pred.upper()}")

            # Stats row 3: Odds and time
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 OVER ODDS", f"{over_odds:.2f}")
            with col2:
                st.metric("📊 UNDER ODDS", f"{under_odds:.2f}")
            with col3:
                st.metric("⏱️ TIME TO GAME", f"{time_to_game} min")

            # Agreement banner
            st.markdown(
                f"""
                <div style="
                    text-align: center; 
                    margin-top: 20px; 
                    padding: 15px; 
                    background: {agree_bg}; 
                    color: white; 
                    border-radius: 10px; 
                    font-size: 1.3rem; 
                    font-weight: 700;
                ">
                    {agree_msg}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Close card
            st.markdown("</div>", unsafe_allow_html=True)


def format_upcoming_games_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the upcoming games dataframe for nice display in Streamlit.

    Args:
        df: DataFrame with upcoming game predictions

    Returns:
        Formatted DataFrame ready for display
    """
    # Create a clean copy
    display_df = pd.DataFrame()

    # Matchup - combine home and away teams
    display_df["Matchup"] = (
        df["team_name_team_home"] + " vs " + df["team_name_team_away"]
    )

    # Game time
    display_df["Game Time (UTC)"] = pd.to_datetime(df["game_time"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    # O/U Line
    display_df["O/U Line"] = df["total_over_under_line"].round(1)

    # Predicted Total Score
    display_df["Predicted Total"] = df["predicted_total_score"].round(1)

    # Margin (difference between prediction and line)
    display_df["Margin"] = df["margin_difference_prediction_vs_over_under"].round(2)

    # Regressor prediction
    display_df["Regressor"] = df["regressor_prediction"]

    # Classifier prediction
    display_df["Classifier"] = df["classifier_prediction_model2"]

    # Both models agree indicator
    both_agree = df["regressor_prediction"] == df["classifier_prediction_model2"]
    display_df["Both Agree"] = both_agree.map({True: "✅", False: "❌"})

    # Odds - format to 2 decimals
    display_df["Over Odds"] = df["average_total_over_money"].round(2)
    display_df["Under Odds"] = df["average_total_under_money"].round(2)

    # Time to match
    display_df["Time to Game (min)"] = df["time_to_match_minutes"].astype(int)

    # Sort by game time
    display_df = display_df.sort_values("Game Time (UTC)").reset_index(drop=True)

    return display_df


def style_predictions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply conditional styling to the predictions dataframe.

    Args:
        df: Formatted DataFrame

    Returns:
        Styled DataFrame
    """

    def highlight_agreement(row):
        if row["Both Agree"] == "✅":
            return ["background-color: #90EE90"] * len(row)
        else:
            return ["background-color: #FFB6C6"] * len(row)

    def highlight_margin(val):
        try:
            if val < -2:
                return "color: blue; font-weight: bold"
            elif val > 2:
                return "color: red; font-weight: bold"
            else:
                return ""
        except:
            return ""

    styled = df.style.apply(highlight_agreement, axis=1)
    styled = styled.applymap(highlight_margin, subset=["Margin"])

    return styled

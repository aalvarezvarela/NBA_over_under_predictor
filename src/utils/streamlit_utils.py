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
    Render games as nice cards with team logos in two-column layout.

    Args:
        df: DataFrame with upcoming game predictions
    """
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
                    game_dt_madrid = game_dt.tz_localize("UTC").tz_convert(
                        "Europe/Madrid"
                    )
                else:
                    game_dt_madrid = game_dt.tz_convert("Europe/Madrid")
                game_time = game_dt_madrid.strftime("%I:%M %p")
                game_date = game_dt_madrid.strftime("%b %d, %Y")

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

                # Determine styling based on agreement
                if both_agree:
                    border_color = "#4CAF50"
                    agree_icon = "✅"
                else:
                    border_color = "#FF9800"
                    agree_icon = "⚠️"

                # Create card with header using st.html for proper rendering
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

                # Stats section with border to match header
                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 2px solid {border_color}; border-top: none; 
                             border-radius: 0 0 12px 12px; padding: 15px; margin-top: -5px; 
                             background: white;">
                        """,
                        unsafe_allow_html=True,
                    )

                    # Agreement indicator
                    agree_text = "Models Agree" if both_agree else "Models Disagree"
                    st.markdown(
                        f"""
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span style="font-size: 1.2rem;">{agree_icon}</span>
                            <span style="font-size: 1rem; font-weight: 600; margin-left: 8px;">
                                {agree_text}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Compact stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("O/U Line", f"{ou_line:.1f}")
                    with col2:
                        st.metric("Predicted", f"{predicted_total:.1f}")
                    with col3:
                        st.metric("Margin", f"{margin:+.1f}")

                    col1, col2 = st.columns(2)
                    with col1:
                        reg_icon = "🔵" if regressor_pred == "Under" else "🔴"
                        st.metric("Regressor", f"{reg_icon} {regressor_pred}")
                    with col2:
                        clf_icon = "🔵" if classifier_pred == "Under" else "🔴"
                        st.metric("Classifier", f"{clf_icon} {classifier_pred}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Over Odds", f"{over_odds:.2f}")
                    with col2:
                        st.metric("Under Odds", f"{under_odds:.2f}")

                    # Close the stats section div
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

    # Game time - convert to Madrid timezone
    game_times = pd.to_datetime(df["game_time"])
    if game_times.dt.tz is None:
        game_times = game_times.dt.tz_localize("UTC")
    display_df["Game Time (Madrid)"] = game_times.dt.tz_convert(
        "Europe/Madrid"
    ).dt.strftime("%Y-%m-%d %H:%M")

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
    display_df = display_df.sort_values("Game Time (Madrid)").reset_index(drop=True)

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

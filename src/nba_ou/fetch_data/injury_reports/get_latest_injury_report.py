import os
import re
from datetime import datetime

import pandas as pd
import pymupdf  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from nba_ou.config.request_headers import HEADERS_BROWSER_LIKE
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

date_pattern = re.compile(r"^\d{2}/\d{2}/\d{4}$")  # e.g. 03/11/2025
time_pattern = re.compile(r"^\d{2}:\d{2} \(ET\)$")  # e.g. 07:00 (ET)
matchup_pattern = re.compile(r"^[A-Z]{3}@[A-Z]{3}$")  # e.g. BKN@CLE
team_pattern = re.compile(r"^[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)+$")

status_pattern = re.compile(r"^[A-Z][a-zA-Z]*$")
player_name_pattern = re.compile(r"^[A-Z][a-zA-Z'.\- ]+, [A-Z][a-zA-Z'.\-]+")


def create_robust_session():
    """Create a requests session with retry strategy and browser-like headers."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Comprehensive headers to mimic a real browser
    session.headers.update(HEADERS_BROWSER_LIKE)

    return session


def get_latest_pdf(nba_injury_report_url, save_path="latest_report_today.pdf"):
    session = create_robust_session()
    try:
        # Fetch the webpage content with improved headers and retry logic
        response = session.get(nba_injury_report_url, timeout=30, allow_redirects=True)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all PDF links (ignoring case)
        pdf_links = [
            link["href"]
            for link in soup.find_all("a", href=True)
            if link["href"].lower().endswith(".pdf")
        ]

        if not pdf_links:
            print("No PDF links found.")
            return

        # Filter only 'Injury-Report' links (case insensitive)
        injury_reports = [link for link in pdf_links if "injury-report" in link.lower()]

        # Extract timestamps from filenames
        timestamp_regex = re.compile(r"(\d{1,2})(AM|PM)", re.IGNORECASE)
        timestamps = [
            (link, match.group(1), match.group(2).upper())
            for link in injury_reports
            if (match := timestamp_regex.search(link))
        ]

        if not timestamps:
            print("No valid timestamps found in PDF links.")
            raise ValueError("No valid timestamps found in PDF links.")

        # Convert timestamps to 24-hour format for proper sorting
        def convert_to_24h(hour, period):
            hour = int(hour)
            if period == "AM" and hour == 12:
                return 0  # 12 AM is 00:00 in 24-hour format
            elif period == "PM" and hour != 12:
                return hour + 12  # Convert PM times correctly
            return hour

        # timestamps = [
        #     (link, convert_to_24h(hour, period)) for link, hour, period in timestamps
        # ]

        # Find the latest timestamp
        # latest_pdf_url, latest_timestamp = max(timestamps, key=lambda x: x[1])
        latest_pdf_url = injury_reports[-1]
        print(f"The latest injury report link is: {latest_pdf_url}")

        # If the URL is relative, make it absolute
        if latest_pdf_url.startswith("/"):
            base_url = re.match(r"https?://[^/]+", nba_injury_report_url).group(0)
            latest_pdf_url = base_url + latest_pdf_url

        print(f"Downloading latest PDF: {latest_pdf_url}")

        # Download the PDF with improved headers and retry logic
        pdf_response = session.get(latest_pdf_url, timeout=30, allow_redirects=True)
        pdf_response.raise_for_status()

        # Save the PDF to disk
        with open(save_path, "wb") as file:
            file.write(pdf_response.content)

        print(f"PDF saved as {save_path}")

    except requests.exceptions.Timeout:
        print("Timeout error: The request took too long")
        raise
    except requests.exceptions.ConnectionError:
        print(f"Connection error: Unable to connect to {nba_injury_report_url}")
        raise
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e.response.status_code} - {e.response.reason}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        raise
    finally:
        session.close()


def classify_token(token, category):
    """Given a single token and the current partial row dict,
    decide which column it belongs to using regex patterns.
    The logic:
      1. If it matches date_pattern => 'Game Date'
      2. Else if time_pattern => 'Game Time'
      3. Else if matchup_pattern => 'Matchup'
      4. Else if team_pattern => 'Team'  (at least two words)
      5. Else if status_pattern => 'Current Status', IF not already filled
         (This might be too permissive, you can refine further)
      6. Else if 'Player Name' not yet assigned => 'Player Name'
      7. Else => append to 'Reason'
    """
    token = token.strip()
    if not token:
        return None

    if token == "NOT YET SUBMITTED":
        return category == "Reason"
    if category == "Player Name":
        if player_name_pattern.match(token):
            return True
    if category == "Game Date":
        if date_pattern.match(token):
            return True
    if category == "Game Time":
        if time_pattern.match(token):
            return True
    if category == "Matchup":
        if matchup_pattern.match(token):
            return True
    if category == "Team":
        if team_pattern.match(token) or token == "Philadelphia 76ers":
            if "," not in token:
                return True
    if category == "Current Status":
        if status_pattern.match(token):
            return True
    return False


def read_injury_report(pdf_path):
    # Path to your PDF fileC
    doc = pymupdf.open(pdf_path)
    col_names = [
        "Game Date",
        "Game Time",
        "Matchup",
        "Team",
        "Player Name",
        "Current Status",
        "Reason",
    ]

    dict_results_template = {col: None for col in col_names}

    data = []  # List to store extracted rows

    for page in doc:
        blocks = page.get_text("blocks")  # Extract text with bounding boxes
        page.get_text("html")
        joinwith_next = None
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]  # Extract bounding box and text
            text = text.strip()
            if not text:
                continue
            text_elements = [t.strip() for t in text.split("\n")]
            if joinwith_next:
                text_elements = joinwith_next + text_elements
                joinwith_next = None

            if text_elements[0] == "Game Date" and text_elements[1] == "Game Time":
                # skip header
                continue

            if len(text_elements) == 2 and any(
                status in text
                for status in [
                    "Out",
                    "Questionable",
                    "Probable",
                    "Doubtful",
                    "Available",
                ]
            ):
                joinwith_next = text_elements.copy()
                continue

            if not text_elements or (len(text_elements) < 2):
                continue  # Skip empty or incomplete lines

            if len(text_elements) >= 3 and (
                ";" in text_elements[-2] or "-" in text_elements[-2]
            ):
                text_elements[-2] = (
                    f"{text_elements[-2]} {text_elements[-1]}"  # Merge last two elements
                )
                text_elements.pop()

            if len(text_elements) >= 4 and (
                ";" in text_elements[-3] or "-" in text_elements[-3]
            ):
                text_elements[-3] = (
                    f"{text_elements[-3]} {text_elements[-2]} {text_elements[-1]}"  # Merge last three elements
                )
                text_elements.pop()
                text_elements.pop()

            current_line = text_elements.copy()
            results = dict_results_template.copy()
            if len(current_line) != 7:
                last_matched_col = None
                for element in current_line:
                    for category in col_names:
                        # Skip already filled values
                        if element in results.values():
                            break

                        # Enforce logical sequence: after "Player Name", only allow "Current Status" or "Reason"
                        if last_matched_col == "Player Name" and category not in [
                            "Current Status",
                            "Reason",
                        ]:
                            continue

                        if classify_token(element, category):
                            results[category] = element
                            last_matched_col = category
                            break

                    # Fallback if Reason is still empty
                    if not results["Reason"]:
                        results["Reason"] = current_line[-1]
            if len(current_line) == 7:
                results = {col: current_line[i] for i, col in enumerate(col_names)}

            # append the current line to data
            data.append(results)

    df = pd.DataFrame(data)

    # ffil except in player name, reason, current Status
    columns_to_ffill = ["Matchup", "Team", "Game Date", "Game Time"]
    df[columns_to_ffill] = df[columns_to_ffill].ffill()
    return df


def retrieve_injury_report_as_df(nba_injury_reports_url, reports_path):
    """
    Retrieves the latest injury report PDF from the NBA website,
    extracts the data, and saves it to a CSV file.
    """
    game_date = datetime.now().strftime("%Y-%m-%d")
    # Ensure the reports directory exists
    os.makedirs(reports_path, exist_ok=True)
    # Get the latest PDF
    pdf_name = f"latest_report_{game_date}.pdf"
    pdf_path = os.path.join(reports_path, pdf_name)
    get_latest_pdf(nba_injury_reports_url, pdf_path)

    # Read the PDF and extract data
    df_report = read_injury_report(pdf_path)

    return df_report

#!/usr/bin/env python3
"""
Test script to read an injury report PDF from a given path and display it as a DataFrame.

Usage:
    python test_read_injury_report.py --path /path/to/injury_report.pdf
    python test_read_injury_report.py -p /path/to/injury_report.pdf
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path to import nba_ou modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
    read_injury_report,
)


def test_read_injury_report(pdf_path: str) -> pd.DataFrame:
    """
    Read an injury report PDF and return it as a DataFrame.

    Args:
        pdf_path: Path to the injury report PDF file

    Returns:
        DataFrame containing the injury report data
    """
    # Check if file exists
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Reading injury report from: {pdf_path}")

    # Read the injury report
    df = read_injury_report(pdf_path)

    print(f"\nSuccessfully read injury report!")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\n" + "=" * 80)
    print("Preview of injury report data:")
    print("=" * 80)

    # Display the DataFrame with better formatting
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)

    print(df.to_string(index=False))

    return df



pdf_path = "/home/adrian_alvarez/Downloads/Injury-Report_2026-02-20_01_30PM.pdf"
df = test_read_injury_report(pdf_path)

  
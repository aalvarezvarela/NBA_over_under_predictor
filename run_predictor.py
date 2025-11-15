#!/usr/bin/env python
"""
Convenience wrapper to run the NBA predictor from project root.

Usage:
    python run_predictor.py
    python run_predictor.py -d 2025-01-15
"""

if __name__ == "__main__":
    from src.nba_predictor import main

    main()

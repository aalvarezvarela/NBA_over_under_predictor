import argparse

from nba_ou.modeling.model_registry import promote_prediction_models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote configured prediction model bundles from staging to production."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the S3 moves. Without this flag the script runs in dry-run mode.",
    )
    args = parser.parse_args()

    promote_prediction_models(dry_run=not args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# All-Star Voting Maintenance

Use this section to keep the all-star voting scrape files and Supabase table up to
date.

## Scrape One Season

```bash
.venv/bin/python scripts/all_star_voting/manage_all_star_voting.py scrape-season 2027
```

By default this writes:

```text
data/all_star_voting/2027/all_conferences.csv
```

The command refuses to overwrite an existing season file unless you pass:

```bash
--overwrite
```

Use `--headed` if Basketball Reference needs browser inspection.

## Build Combined Input From Available Seasons

```bash
.venv/bin/python scripts/all_star_voting/manage_all_star_voting.py build-input --overwrite-output
```

This combines every available:

```text
data/all_star_voting/<year>/all_conferences.csv
```

into:

```text
data/all_star_voting/all_star_voting_combined.csv
```

Use `--start-year` / `--end-year` to limit the all-star game years included.

## Upload To Supabase

```bash
.venv/bin/python scripts/all_star_voting/manage_all_star_voting.py upload
```

Default behavior:

- Reuses `data/all_star_voting/all_star_voting_combined.csv` if it exists.
- Combines available scrape files first when the combined CSV does not exist.
- Prepares player IDs, `season_year`, `team_name`, and vote percentages.
- Creates or alters `nba_all_star_voting.nba_all_star_voting` if needed.
- Upserts rows without dropping the table.

Pass `--overwrite-output` when you intentionally want to rebuild the combined CSV
from the per-season scrape files before upload.

Use this only when you intentionally want to replace the table:

```bash
--drop-existing
```

Use this to validate the current available scrape data without uploading:

```bash
--prepare-only
```

Use this to skip rows whose player_id or team cannot be resolved instead of
aborting the whole upload:

```bash
--skip-unresolved
```

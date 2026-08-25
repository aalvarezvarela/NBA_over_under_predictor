# Skills

`experiments` is repo-specific: it describes `training_pipeline/`,
`experiments/` and `artifacts/experiments/` in this repository.

The other three are **portable**. They were distilled from this NBA system but
are written as sport-agnostic decisions with the NBA implementation as evidence
and an explicit MLB translation in each section:

- `odds-data-architecture` — SBR ingestion, tick storage, snapshots
- `sports-data-architecture` — entities, identity, availability, context data
- `feature-engineering` — temporal rules and the eleven feature families

To bootstrap another sport, copy those three directories into the new repo's
`.claude/skills/` unchanged. Each ends with an ordered build checklist. Port
`experiments` too once the new repo has a training pipeline — the evaluation
discipline in it (seed noise before ranking, CV vs holdout, the silent-no-op
table) is sport-independent even though the file paths are not.

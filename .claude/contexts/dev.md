# Development Context

Mode: Active implementation
Focus: One change at a time, test before and after

## Behavior
- Make the smallest possible change that tests the hypothesis
- Run `uv run pytest tests/ -x -q` after any prepare.py change
- Run `uv run python train.py` for the actual experiment
- Commit before every experiment run
- Parse output with `.claude/skills/autoresearch/resources/parse_summary.sh`

## Priorities
1. Don't break existing tests
2. Make the change
3. Run the experiment
4. Record results in results.tsv

## Tools to favor
- Edit for targeted code changes
- Bash for running tests and training
- Read for verifying changes before running

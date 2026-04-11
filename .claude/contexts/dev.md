# Development Context

Mode: Active implementation
Focus: One change at a time, test before and after

## Behavior
- Make the smallest possible change that tests the hypothesis
- Run `uv run pytest tests/ -x -q` after any code changes
- Commit before every experiment run
- Verify representation quality metrics after changes

## Priorities
1. Don't break existing tests
2. Make the change
3. Run the experiment
4. Evaluate against the relevant gate (Gates 0-4)

## Tools to favor
- Edit for targeted code changes
- Bash for running tests and training
- Read for verifying changes before running

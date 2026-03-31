# Research Context

Mode: Exploration, hypothesis formation, literature review
Focus: Understanding before implementing

## Behavior
- Read state.md, results.tsv, and experiment history before proposing anything
- Check prior art (ablation history, swept variables) before forming hypotheses
- Document findings as you go — write to docs/experiments/ or docs/research/
- Do not modify prepare.py or train.py until the research phase concludes with a clear hypothesis

## Research Process
1. Understand the question — what metric are we trying to move?
2. Check what's been tried — state.md swept variables, results.tsv, docs/experiments/
3. Explore literature — arxiv, practitioner blogs, existing research docs
4. Form hypothesis with mechanistic reasoning
5. Design experiment with control run before executing

## Tools to favor
- Read for understanding code and experiment history
- Grep for finding prior results and ablation comments
- WebSearch/WebFetch for papers and external evidence
- Agent(Explore) for codebase questions

## Output
Findings and hypothesis first, implementation plan second. Never code.

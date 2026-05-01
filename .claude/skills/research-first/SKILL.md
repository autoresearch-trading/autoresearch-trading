---
name: research-first
description: Research before implementing active Pacifica paper-trading changes: internal history, current full-fidelity docs, and external evidence where needed.
---

# Research-First Protocol — active full-fidelity branch

Before implementing new strategy logic, eligibility gates, overlays, execution assumptions, or model changes:

1. Check `docs/NEXT_SESSION_HANDOFF.md` and `CLAUDE.md`.
2. Search current experiment docs under `docs/experiments/`.
3. Search current code under `scripts/` and `tests/`.
4. Search historical docs only as context; do not revive old Goal-A/representation-learning assumptions by default.
5. Use external research only when a decision needs outside evidence.

Focus on non-HFT, post-cost, causally testable ideas.

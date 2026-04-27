# Goal-A Survivors at Realistic Directional Accuracy

Cells that satisfy `frac_pos_acc_X > 0.55` AND `headroom_X_median_bps > 0`, where X is the directional accuracy regime.

**Cost band**: 2 × 6 bp fee + 2 × |slip| (round-trip). **Edge math**: expected gross PnL per round trip = (2p − 1) × |edge|. Headroom = expected gross − cost. See README §Survivors at realistic directional accuracy for the modelling caveats.


## Accuracy = 0.550 (55%)

_Survivor count: **0** of 300 cells._

_No cells survive at this accuracy._


### Top 5 cells by median headroom@0.55 (gate-aware)

| symbol | size | horizon | edge_med | slip_med | headroom_med | frac_pos | gate? |
|--------|------|---------|----------|----------|--------------|----------|-------|
| HYPE | $1k | H500 | 62.31 bp | 1.14 bp | -8.17 bp | 15.6% | fail |
| PUMP | $1k | H500 | 100.50 bp | 3.27 bp | -9.26 bp | 24.3% | fail |
| SOL | $1k | H500 | 39.92 bp | 0.58 bp | -9.51 bp | 8.2% | fail |
| ENA | $1k | H500 | 70.85 bp | 2.43 bp | -9.61 bp | 17.3% | fail |
| BNB | $1k | H500 | 35.26 bp | 0.61 bp | -9.89 bp | 7.0% | fail |


## Accuracy = 0.575 (57.5%)

_Survivor count: **0** of 300 cells._

_No cells survive at this accuracy._


### Top 5 cells by median headroom@0.575 (gate-aware)

| symbol | size | horizon | edge_med | slip_med | headroom_med | frac_pos | gate? |
|--------|------|---------|----------|----------|--------------|----------|-------|
| PUMP | $1k | H500 | 100.50 bp | 3.27 bp | -4.43 bp | 40.2% | fail |
| HYPE | $1k | H500 | 62.31 bp | 1.14 bp | -5.08 bp | 31.2% | fail |
| ENA | $1k | H500 | 70.85 bp | 2.43 bp | -6.22 bp | 31.9% | fail |
| HYPE | $10k | H500 | 62.31 bp | 2.25 bp | -7.16 bp | 25.0% | fail |
| SOL | $1k | H500 | 39.92 bp | 0.58 bp | -7.53 bp | 19.0% | fail |


## Accuracy = 0.600 (60%)

_Survivor count: **0** of 300 cells._

_No cells survive at this accuracy._


### Top 5 cells by median headroom@0.6 (gate-aware)

| symbol | size | horizon | edge_med | slip_med | headroom_med | frac_pos | gate? |
|--------|------|---------|----------|----------|--------------|----------|-------|
| PUMP | $1k | H500 | 100.50 bp | 3.27 bp | 0.51 bp | 51.1% | fail |
| HYPE | $1k | H500 | 62.31 bp | 1.14 bp | -2.00 bp | 43.7% | fail |
| ENA | $1k | H500 | 70.85 bp | 2.43 bp | -2.72 bp | 43.3% | fail |
| PUMP | $10k | H500 | 100.50 bp | 5.53 bp | -3.48 bp | 43.2% | fail |
| HYPE | $10k | H500 | 62.31 bp | 2.25 bp | -4.07 bp | 37.8% | fail |

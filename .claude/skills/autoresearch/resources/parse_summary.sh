#!/bin/bash
# Extract PORTFOLIO SUMMARY fields from train.py log as key=value pairs
# Usage: bash parse_summary.sh run_v5-sanity.log
#
# Parsing notes:
# - symbols_passing prints "3/25" — split on "/" to get integer
# - win_rate and profit_factor are conditionally printed (only when wr > 0)
#   treat missing values as 0.0
grep -E "^(sortino|sharpe|calmar|cvar_95|symbols_passing|num_trades|max_drawdown|win_rate|profit_factor):" "$1"

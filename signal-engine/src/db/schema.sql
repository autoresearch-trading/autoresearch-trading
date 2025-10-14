-- QuestDB schema for neurosymbolic trading bot signal engine

CREATE TABLE IF NOT EXISTS signals (
    ts TIMESTAMP,
    recv_ts TIMESTAMP,
    symbol SYMBOL,
    signal_type SYMBOL,
    value DOUBLE,
    confidence DOUBLE,
    direction SYMBOL,
    price DOUBLE,
    spread_bps INT,
    bid_depth DOUBLE,
    ask_depth DOUBLE,
    metadata STRING
) TIMESTAMP(ts) PARTITION BY DAY;

CREATE TABLE IF NOT EXISTS trades_processed (
    ts TIMESTAMP,
    symbol SYMBOL,
    trade_id STRING,
    side SYMBOL,
    price DOUBLE,
    qty DOUBLE,
    is_large BOOLEAN
) TIMESTAMP(ts) PARTITION BY DAY;

CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    ts TIMESTAMP,
    symbol SYMBOL,
    bid1_price DOUBLE, bid1_qty DOUBLE,
    bid2_price DOUBLE, bid2_qty DOUBLE,
    bid3_price DOUBLE, bid3_qty DOUBLE,
    bid4_price DOUBLE, bid4_qty DOUBLE,
    bid5_price DOUBLE, bid5_qty DOUBLE,
    ask1_price DOUBLE, ask1_qty DOUBLE,
    ask2_price DOUBLE, ask2_qty DOUBLE,
    ask3_price DOUBLE, ask3_qty DOUBLE,
    ask4_price DOUBLE, ask4_qty DOUBLE,
    ask5_price DOUBLE, ask5_qty DOUBLE,
    mid_price DOUBLE,
    spread_bps INT
) TIMESTAMP(ts) PARTITION BY DAY;

CREATE TABLE IF NOT EXISTS paper_trades (
    ts TIMESTAMP,
    symbol SYMBOL,
    trade_id SYMBOL,
    side SYMBOL,
    entry_price DOUBLE,
    qty DOUBLE,
    stop_loss DOUBLE,
    take_profit DOUBLE,
    exit_price DOUBLE,
    exit_ts TIMESTAMP,
    pnl DOUBLE,
    pnl_pct DOUBLE,
    cvd_value DOUBLE,
    tfi_value DOUBLE,
    ofi_value DOUBLE,
    regime STRING
) TIMESTAMP(ts) PARTITION BY DAY;

CREATE TABLE IF NOT EXISTS regime_log (
    ts TIMESTAMP,
    symbol SYMBOL,
    regime SYMBOL,
    atr DOUBLE,
    spread_bps INT,
    funding_rate DOUBLE,
    should_trade BOOLEAN
) TIMESTAMP(ts) PARTITION BY HOUR;

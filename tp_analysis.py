import csv

MAX_N = 20
COST_PCT = 0.0005  # optional transaction cost

def pf(s):
    try:
        return float(s)
    except:
        return None

def candle_cols(n):
    return {"high": f"High_{n}", "low": f"Low_{n}", "close": f"Close_{n}"}

def first_event(entry, stop, tp_price, row):
    last_close = None
    for n in range(1, MAX_N+1):
        cols = candle_cols(n)
        high = pf(row.get(cols["high"], ""))
        low  = pf(row.get(cols["low"], ""))
        close = pf(row.get(cols["close"], ""))
        if high is None or low is None:
            break
        last_close = close if close is not None else last_close
        sl_hit = (low < stop)
        tp_hit = (high >= tp_price)
        if sl_hit and tp_hit:
            return ("SL", stop)
        elif sl_hit:
            return ("SL", stop)
        elif tp_hit:
            return ("TP", tp_price)
    return ("NONE", last_close)

def trade_pnl(entry, outcome, price):
    if outcome == "TP":
        pnl = (price - entry) / entry
    elif outcome == "SL":
        pnl = (price - entry) / entry
    else:
        pnl = (price - entry) / entry if price else 0.0
    return pnl - COST_PCT

def evaluate(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    for r in rows:
        entry = pf(r.get("Open_1"))
        stop  = pf(r.get("SignalLow"))
        if not entry or not stop:
            continue
        risk_pct = (entry - stop) / entry * 100
        if risk_pct >= 1.0:  # skip trades with risk >= 1%
            continue
        tp_price = entry * 1.07
        outcome, price = first_event(entry, stop, tp_price, r)
        pnl = trade_pnl(entry, outcome, price)
        results.append((outcome, pnl))

    total_return = sum(p for _, p in results) * 100
    avg_return = (total_return / len(results)) if results else 0
    tp_hits = sum(1 for o, _ in results if o == "TP")
    sl_hits = sum(1 for o, _ in results if o == "SL")
    none = sum(1 for o, _ in results if o == "NONE")

    print(f"Trades taken: {len(results)}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Average per trade: {avg_return:.2f}%")
    print(f"TP hits: {tp_hits}, SL hits: {sl_hits}, None: {none}")

if __name__ == "__main__":
    evaluate("next_20_candles_30m (2).csv")
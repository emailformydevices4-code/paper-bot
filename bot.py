"""
Paper-Trading Bot for NSE stocks (virtual money, free data, runs on GitHub Actions)

Every weekday after market close it:
  1. Downloads free daily prices (Yahoo Finance) for your watchlist
  2. Fills yesterday's queued virtual orders at TODAY'S OPEN (realistic delay)
  3. Every 28 days re-searches ~600 strategy settings per stock using
     walk-forward testing (settings chosen on old data, graded on newer data)
  4. Only trusts a stock if its strategy passes strict checks
  5. Queues new BUY/SELL orders for tomorrow, applies stop-losses
  6. Updates the virtual account in state.json and messages you on Telegram

Nothing here places real orders. Not financial advice.
"""
import os
import sys
import json
import time
import urllib.parse
import urllib.request
import datetime as dt

import numpy as np
import pandas as pd

# ======================= SETTINGS (edit freely) =======================
START_CAPITAL = 10000.0          # virtual money (rupees)
MIN_PRICE = 10.0                 # ignore stocks priced below this
MIN_MEDIAN_TRADED_VALUE = 500000 # skip stocks trading less than ~Rs 5 lakh/day
MIN_HISTORY = 900                # need ~3.5 years of daily data
MAX_POSITIONS = 4
MAX_POSITION_PCT = 0.25          # max 25% of account in one stock
MAX_PCT_OF_DAILY_VOLUME = 0.01   # never buy more than 1% of a day's volume
STOP_LOSS = 0.15                 # sell if price falls 15% below your buy price
MAX_ROUNDTRIP_FEE_PCT = 0.04     # skip a buy if buy+sell fees exceed 4% of it
REOPT_EVERY_DAYS = 28            # how often to re-run the strategy search

# Costs. CHECK YOUR BROKER'S CURRENT CHARGES and edit these.
BROKERAGE_PCT = 0.001            # 0.1% of order value ...
BROKERAGE_MIN = 5.0              # ... but at least Rs 5
BROKERAGE_MAX = 20.0             # ... and at most Rs 20
DP_CHARGE_SELL = 15.34           # depository charge on every sell
SLIPPAGE = 0.003                 # 0.3% worse price than the chart (thin stocks)

# Strictness of the exam a strategy must pass on data it never saw
OOS_FOLDS = 3                    # 3 unseen test windows ...
FOLD_DAYS = 252                  # ... of about 1 year each
MIN_OOS_SHARPE = 0.5
MIN_OOS_SIGNALS = 6              # at least ~3 round trips in the test windows
MIN_PROFITABLE_FOLDS = 2         # profitable in at least 2 of 3 windows

STATE_FILE = os.getenv("STATE_FILE", "state.json")
WATCHLIST_FILE = os.getenv("WATCHLIST_FILE", "watchlist.txt")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# =====================================================================


# ---------------------------- costs ----------------------------------
def fees(value, side):
    """Approximate Indian delivery-trade charges in rupees (excluding slippage)."""
    brokerage = min(max(BROKERAGE_PCT * value, BROKERAGE_MIN), BROKERAGE_MAX)
    stt = 0.001 * value
    exch = 0.0000297 * value
    sebi = 0.000001 * value
    stamp = 0.00015 * value if side == "BUY" else 0.0
    gst = 0.18 * (brokerage + exch + sebi)
    dp = DP_CHARGE_SELL if side == "SELL" else 0.0
    return brokerage + stt + exch + sebi + stamp + gst + dp


# cost per buy or sell as a fraction, measured on a typical order size
_REF = START_CAPITAL * MAX_POSITION_PCT
COST_SIDE = (fees(_REF, "BUY") + fees(_REF, "SELL")) / 2 / _REF + SLIPPAGE


# ---------------------------- data -----------------------------------
def load(sym):
    import yfinance as yf
    t = sym if ("." in sym or sym.startswith("^")) else sym + ".NS"
    for _ in range(3):
        try:
            df = yf.download(t, period="10y", interval="1d", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df):
                return df
        except Exception:
            pass
        time.sleep(2)
    return None


def read_watchlist():
    out = []
    if os.path.exists(WATCHLIST_FILE):
        for line in open(WATCHLIST_FILE):
            s = line.split("#")[0].strip().upper()
            if s:
                out.append(s)
    return out


def eligible(df):
    if len(df) < MIN_HISTORY:
        return False, f"only {len(df)} days of history"
    price = float(df["Close"].iloc[-1])
    if price < MIN_PRICE:
        return False, f"price Rs {price:.2f} is below Rs {MIN_PRICE:.0f}"
    tv = float((df["Close"] * df["Volume"]).tail(60).median())
    if tv < MIN_MEDIAN_TRADED_VALUE:
        return False, f"too illiquid (median Rs {tv / 1e5:.1f} lakh/day)"
    return True, ""


# ------------------------- strategies --------------------------------
def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)


def make_signal(close, high, low, kind, p, cache):
    """1 = want to hold the stock, 0 = want to be out. Uses only past data."""
    def sma(n):
        k = ("s", n)
        if k not in cache:
            cache[k] = close.rolling(n).mean()
        return cache[k]

    if kind == "MA":                      # trend: fast average above slow average
        fast, slow = p
        return (sma(fast) > sma(slow)).astype(float)
    pos = pd.Series(np.nan, index=close.index)
    if kind == "RSI":                     # mean reversion: buy dips, sell rallies
        n, lo, hi = p
        k = ("r", n)
        if k not in cache:
            cache[k] = rsi(close, n)
        r = cache[k]
        pos[r < lo] = 1.0
        pos[r > hi] = 0.0
    else:                                 # BRK: breakout of N-day high, exit on M-day low
        n, m = p
        pos[close > high.rolling(n).max().shift(1)] = 1.0
        pos[close < low.rolling(m).min().shift(1)] = 0.0
    return pos.ffill().fillna(0.0)


def grid():
    for fast in range(3, 51, 3):
        for slow in range(20, 201, 5):
            if fast < slow:
                yield "MA", (fast, slow)
    for n in (7, 14, 21):
        for lo in (20, 25, 30, 35):
            for hi in (55, 60, 65, 70, 75):
                yield "RSI", (n, lo, hi)
    for n in (10, 20, 40, 55, 80, 100):
        for m in (5, 10, 20):
            if m < n:
                yield "BRK", (n, m)


def stats(r):
    r = np.asarray(r, dtype=float)
    if len(r) < 30 or r.std() == 0:
        return {"ret": 0.0, "sharpe": 0.0, "dd": 0.0}
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return {"ret": float(eq[-1] - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "dd": float((eq / peak - 1).min())}


def sharpe_cols(R):
    m, s = R.mean(0), R.std(0)
    return np.where(s > 0, m / np.where(s > 0, s, 1) * np.sqrt(252), 0.0)


def search(df, today):
    """Backtest every setting, grade the best-picking PROCESS on unseen windows."""
    close, high, low = df["Close"], df["High"], df["Low"]
    ret = close.pct_change().fillna(0.0)
    cache, combos, Rl, Pl = {}, [], [], []
    for kind, p in grid():
        sig = make_signal(close, high, low, kind, p, cache)
        pos = sig.shift(1).fillna(0.0)
        r = pos * ret - COST_SIDE * pos.diff().abs().fillna(0.0)
        combos.append((kind, p))
        Rl.append(r.values)
        Pl.append(pos.values)
    R, P, n = np.column_stack(Rl), np.column_stack(Pl), len(ret)
    bh_all = ret.values

    oos, bh_oos, fold_rets, signals = [], [], [], 0
    for k in range(OOS_FOLDS):
        te = n - (OOS_FOLDS - 1 - k) * FOLD_DAYS
        ts = te - FOLD_DAYS
        j = int(np.argmax(sharpe_cols(R[200:ts])))   # pick using ONLY older data
        seg = R[ts:te, j]
        oos.append(seg)
        bh_oos.append(bh_all[ts:te])
        fold_rets.append(stats(seg)["ret"])
        signals += int(np.abs(np.diff(P[ts - 1:te, j])).sum())
    o, b = stats(np.concatenate(oos)), stats(np.concatenate(bh_oos))
    profitable = sum(1 for x in fold_rets if x > 0)

    problems = []
    if o["sharpe"] < MIN_OOS_SHARPE:
        problems.append(f"unseen Sharpe {o['sharpe']:.2f} < {MIN_OOS_SHARPE}")
    if profitable < MIN_PROFITABLE_FOLDS:
        problems.append(f"profitable in only {profitable}/{OOS_FOLDS} windows")
    if signals < MIN_OOS_SIGNALS:
        problems.append("too few trades to judge")
    if not (o["ret"] > b["ret"] or o["dd"] > b["dd"]):
        problems.append("no better than just holding")

    jf = int(np.argmax(sharpe_cols(R[200:])))        # final pick uses all data
    kind, p = combos[jf]
    return {"kind": kind, "params": list(p), "passed": not problems,
            "reason": "; ".join(problems) if problems else "passed all checks",
            "oos": {**o, "signals": signals}, "buy_hold": b,
            "folds": [round(x, 3) for x in fold_rets],
            "tested": len(combos), "optimized_on": today}


def live_signal(df, strat):
    cache = {}
    sig = make_signal(df["Close"], df["High"], df["Low"], strat["kind"], tuple(strat["params"]), cache)
    return int(sig.iloc[-1])


# ------------------------- paper account ------------------------------
def new_state():
    return {"start_capital": START_CAPITAL, "cash": START_CAPITAL, "positions": {},
            "pending": [], "trades": [], "equity_curve": [], "strategies": {},
            "last_data_date": None, "nifty_start": None}


def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return new_state()


def last_close(data, sym, fallback):
    return float(data[sym]["Close"].iloc[-1]) if sym in data else fallback


def equity(state, data):
    v = state["cash"]
    for s, p in state["positions"].items():
        v += p["qty"] * last_close(data, s, p["avg_price"])
    return v


def fill_pending(state, data, msgs):
    still = []
    for o in state["pending"]:
        sym, df = o["symbol"], data.get(o["symbol"])
        bars = df[df.index > pd.Timestamp(o["signal_date"])] if df is not None else None
        if bars is None or bars.empty:
            still.append(o)
            continue
        fill_date, bar = bars.index[0], bars.iloc[0]
        if o["side"] == "BUY":
            price = float(bar["Open"]) * (1 + SLIPPAGE)
            budget = min(state["cash"], equity(state, data) * MAX_POSITION_PCT)
            adv = float(df.loc[:fill_date].iloc[:-1]["Volume"].tail(20).mean())
            qty = min(int(budget // price), int(MAX_PCT_OF_DAILY_VOLUME * adv))
            while qty > 0 and qty * price + fees(qty * price, "BUY") > state["cash"]:
                qty -= 1
            val = qty * price
            if qty < 1 or (fees(val, "BUY") + fees(val, "SELL")) / val > MAX_ROUNDTRIP_FEE_PCT:
                msgs.append(f"SKIPPED buy {sym}: order too small, fees would eat the profit")
                continue
            f = fees(val, "BUY")
            state["cash"] -= val + f
            state["positions"][sym] = {"qty": qty, "avg_price": price, "buy_fees": f,
                                       "entry_date": fill_date.date().isoformat()}
            state["trades"].append({"date": fill_date.date().isoformat(), "symbol": sym,
                                    "side": "BUY", "qty": qty, "price": round(price, 2),
                                    "fees": round(f, 2), "pnl": None, "reason": o["reason"]})
            msgs.append(f"BOUGHT {qty} {sym} @ Rs {price:.2f} (fees Rs {f:.0f})")
        else:
            pos = state["positions"].get(sym)
            if not pos:
                continue
            price = float(bar["Open"]) * (1 - SLIPPAGE)
            val = pos["qty"] * price
            f = fees(val, "SELL")
            pnl = (val - f) - (pos["qty"] * pos["avg_price"] + pos["buy_fees"])
            state["cash"] += val - f
            del state["positions"][sym]
            state["trades"].append({"date": fill_date.date().isoformat(), "symbol": sym,
                                    "side": "SELL", "qty": pos["qty"], "price": round(price, 2),
                                    "fees": round(f, 2), "pnl": round(pnl, 2), "reason": o["reason"]})
            msgs.append(f"SOLD {pos['qty']} {sym} @ Rs {price:.2f} | P/L Rs {pnl:+.0f} ({o['reason']})")
    state["pending"] = still


def queue_signals(state, data, watch, last_date, msgs):
    pend = {(o["symbol"]) for o in state["pending"]}
    n_buys = sum(1 for o in state["pending"] if o["side"] == "BUY")
    for sym in sorted(set(watch) | set(state["positions"])):
        df = data.get(sym)
        if df is None or df.index[-1].date().isoformat() != last_date or sym in pend:
            continue
        strat = state["strategies"].get(sym)
        ok, _ = eligible(df)
        want = int(bool(strat and strat.get("passed") and ok and live_signal(df, strat)))
        pos = state["positions"].get(sym)
        px = float(df["Close"].iloc[-1])
        reason = None
        if pos and px <= pos["avg_price"] * (1 - STOP_LOSS):
            reason = "stop-loss"
        elif pos and not want:
            reason = "exit signal" if (strat and strat.get("passed")) else "strategy no longer trusted"
        if reason:
            state["pending"].append({"symbol": sym, "side": "SELL", "signal_date": last_date, "reason": reason})
            msgs.append(f"QUEUED SELL {sym} for tomorrow's open ({reason})")
        elif want and not pos and len(state["positions"]) + n_buys < MAX_POSITIONS:
            state["pending"].append({"symbol": sym, "side": "BUY", "signal_date": last_date, "reason": "entry signal"})
            n_buys += 1
            msgs.append(f"QUEUED BUY {sym} for tomorrow's open ({strat['kind']} {strat['params']})")


def optimize_if_due(state, data, watch, last_date, force, msgs):
    for sym in watch:
        df = data.get(sym)
        if df is None:
            continue
        old = state["strategies"].get(sym)
        age = 10 ** 6
        if old:
            age = (dt.date.fromisoformat(last_date) - dt.date.fromisoformat(old["optimized_on"])).days
        if not (force or age >= REOPT_EVERY_DAYS):
            continue
        ok, why = eligible(df)
        if not ok:
            state["strategies"][sym] = {"passed": False, "reason": why, "optimized_on": last_date}
            continue
        res = search(df, last_date)
        state["strategies"][sym] = res
        if res["passed"] and not (old and old.get("passed")):
            msgs.append(f"{sym}: strategy {res['kind']}{res['params']} PASSED the unseen-data exam "
                        f"(return {res['oos']['ret']:.0%} vs hold {res['buy_hold']['ret']:.0%})")


# ------------------------- reporting ----------------------------------
def summary(state):
    ec = state["equity_curve"]
    eq = ec[-1]["equity"] if ec else state["cash"]
    lines = [f"Paper account: Rs {eq:,.0f} ({eq / state['start_capital'] - 1:+.1%}) | cash Rs {state['cash']:,.0f}"]
    if ec and ec[-1].get("nifty") and state.get("nifty_start"):
        lines.append(f"Nifty 50 over same period: {ec[-1]['nifty'] / state['nifty_start'] - 1:+.1%}")
    sells = [t for t in state["trades"] if t["side"] == "SELL"]
    wins = sum(1 for t in sells if t["pnl"] > 0)
    paid = sum(t["fees"] for t in state["trades"])
    lines.append(f"Closed trades: {len(sells)} (wins {wins}) | total fees paid Rs {paid:,.0f}")
    for s, p in state["positions"].items():
        lines.append(f"Holding {p['qty']} {s} @ Rs {p['avg_price']:.2f} since {p['entry_date']}")
    trusted = [s for s, v in state["strategies"].items() if v.get("passed")]
    lines.append(f"Stocks passing the exam: {', '.join(trusted) if trusted else 'none yet'}")
    return "\n".join(lines)


def notify(text):
    print(text)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            body = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": text[:4000]}).encode()
            urllib.request.urlopen(url, body, timeout=15)
        except Exception as e:
            print("Telegram failed:", e)


def report():
    s = load_state()
    print(summary(s))
    print("\nLast 10 trades:")
    for t in s["trades"][-10:]:
        print(t)
    print("\nStrategies:")
    for sym, v in s["strategies"].items():
        print(sym, "PASS" if v.get("passed") else "FAIL", "-", v.get("reason"))


# ----------------------------- main -----------------------------------
def main():
    force = os.getenv("FORCE_REOPT", "").lower() == "true" or "--reoptimize" in sys.argv
    state, watch, msgs = load_state(), read_watchlist(), []
    symbols = sorted(set(watch) | set(state["positions"]))
    data = {}
    for s in symbols:
        df = load(s)
        if df is None:
            msgs.append(f"{s}: could not download data (skipped)")
        else:
            data[s] = df
    if not data:
        notify("Bot: no price data could be downloaded today.")
        return
    last_date = max(df.index[-1] for df in data.values()).date().isoformat()
    if state["last_data_date"] == last_date and not force:
        print("No new trading day since the last run. Nothing to do.")
        return

    fill_pending(state, data, msgs)
    optimize_if_due(state, data, watch, last_date, force, msgs)
    queue_signals(state, data, watch, last_date, msgs)

    nifty = load("^NSEI")
    nclose = float(nifty["Close"].iloc[-1]) if nifty is not None else None
    if state["nifty_start"] is None and nclose:
        state["nifty_start"] = nclose
    row = {"date": last_date, "equity": round(equity(state, data), 2), "nifty": nclose}
    if state["equity_curve"] and state["equity_curve"][-1]["date"] == last_date:
        state["equity_curve"][-1] = row
    else:
        state["equity_curve"].append(row)
    state["last_data_date"] = last_date
    json.dump(state, open(STATE_FILE, "w"), indent=1)

    friday = dt.date.fromisoformat(last_date).weekday() == 4
    if msgs or friday:
        head = f"Paper bot | market date {last_date}"
        body = "\n".join(msgs) if msgs else "No trades today."
        notify(f"{head}\n{body}\n\n{summary(state)}" if friday else f"{head}\n{body}")
    else:
        print("Quiet day: no orders, no news.")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        main()

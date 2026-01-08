import warnings # non-fatal warnings (like “this will change in future versions”)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans # clustering algorithm; groups days into regimes
from sklearn.preprocessing import StandardScaler # normalizes features so one feature doesn’t dominate

import yfinance as yf

INTERACTIVE_MODE = True
SHOW_PLOTS = False   # we don't want graphs at this point, so we skip them

DATA_PATH = "ai_etf_downside_risk_data.xlsx"

N_REGIMES = 3                      # calm / stressed / crisis
TAIL_ALPHA = 0.95                  # CVaR at 95%: average loss on the worst 5% days
WEIGHT_GRID_STEP = 0.1             # When choosing allocations, the script tries weights in steps of 10%: 0.0, 0.1, … 1.0
TRADING_DAYS = 252

MARKET_TICKER = "SPY"              # SPY represents the S&P 500. Used to compute a 20-day market return feature

# Practical guideline threshold, not used for clustering, just printed as a human “rule of thumb”
PRACTICAL_VIX_LEVEL = 25.0

def cvar(returns, alpha=0.95): #define a function called cvar
    r = np.asarray(returns) #convert returns into a numpy array
    r = r[np.isfinite(r)] #Drop NaN and +- inf values.
    if len(r) == 0:
        return np.nan #if no usable returns exist, return NaN
    r_sorted = np.sort(r) # sort returns from most negative to most positive
    tail_n = int(np.ceil((1 - alpha) * len(r_sorted))) #if alpha=0.95, (1-alpha)=0.05 -> worst 5%
    tail_n = max(tail_n, 1) #ceil ensures at least enough observations
    return r_sorted[:tail_n].mean() # max(...,1) ensures at least 1 point

def sortino_ratio(returns, rf=0.0): #define sortino function
    r = np.asarray(returns) #convert to array and drop NaNs
    r = r[np.isfinite(r)]
    if len(r) < 5: 
        return np.nan # too few points -> unreliable, return NaN
    excess = r - rf # subtract risk-free rate (here rf=0, so basically unchanged)
    downside = excess[excess < 0] #inly negative returnas
    if len(downside) < 3:
        return np.nan # if there aren’t enough negative points, downside volatility is not stable
    ds = downside.std(ddof=1)
    if ds == 0: #compute downside standard deviation and avoid dividing by zero
        return np.nan
    return excess.mean() / ds # Sortino = mean return / downside volatility

def max_drawdown(returns): #define max drawdown function
    r = np.asarray(returns)
    r = r[np.isfinite(r)] #convert and clean
    if len(r) == 0:
        return np.nan # if empty return NaN 
    equity = (1 + r).cumprod() #turn daily returns into a “portfolio value over time” starting from $1
    peak = np.maximum.accumulate(equity) #running maximum: for each day, what was the highest value so far?
    dd = (equity - peak) / peak # drawdown is how far below the peak you are (negative number)
    return dd.min() # the most negative value is the worst drawdown

def annualized_return(daily_returns, trading_days=252): # annual return using log/compounding
    r = np.asarray(daily_returns) #log1p(r) = log(1+r) handles compounding properly; 
    #multiply by 252 -> annual log return;exp(...) - 1 converts back to percent return
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    return np.exp(np.log1p(r).mean() * trading_days) - 1

# annualized volatility: daily std * sqrt(252).
def annualized_vol(daily_returns, trading_days=252):
    r = np.asarray(daily_returns)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(trading_days) # Vol scales with sqr(time)

def perf_summary(returns, name): #Creates a dictionary of key stats used in output.
	#Annual return
	#Annual vol
	#Sortino
	#Max drawdown
	#CVaR95
    return {
        "name": name,
        "ann_return": annualized_return(returns, TRADING_DAYS),
        "ann_vol": annualized_vol(returns, TRADING_DAYS),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "cvar_95": cvar(returns, TAIL_ALPHA),
    }

def find_sheet(xls, keywords, preferred=None):
    sheets = xls.sheet_names
    if preferred and preferred in sheets:
        return preferred
    for s in sheets:
        low = s.lower()
        if any(k in low for k in keywords):
            return s
    raise ValueError(f"Could not find a suitable sheet. Available sheets: {sheets}")
# If “Daily_Returns” exists, it uses it, otherwise searches any sheet name containing "return"

def find_col(df, keywords, preferred=None):
    cols = list(df.columns)
    if preferred and preferred in cols:
        return preferred
    for c in cols:
        low = str(c).lower()
        if any(k in low for k in keywords):
            return c
    return None
#same idea but for columns (Date, Ticker, is_ai)

print("Loading data from Excel...") 

xls = pd.ExcelFile(DATA_PATH) # Opens the workbook

returns_sheet = find_sheet(xls, keywords=["return"], preferred="Daily_Returns")
funds_sheet   = find_sheet(xls, keywords=["fund", "summary", "meta", "info"], preferred="Fund_Summary")
#finds sheet names

daily = pd.read_excel(xls, returns_sheet)
funds = pd.read_excel(xls, funds_sheet)
# reads both sheets into pandas

date_col = find_col(daily, keywords=["date"], preferred="Date")
if date_col is None:
    raise ValueError(f"No date-like column found in returns sheet. Columns: {daily.columns}")
# finds Date column, if not found, stops
daily[date_col] = pd.to_datetime(daily[date_col])
daily = daily.sort_values(date_col).set_index(date_col)
#convert to datetime, sort and make Date the index

# keep only numeric columns (tickers), keep ETF return columns
daily = daily.select_dtypes(include=[np.number])

ticker_col = find_col(funds, keywords=["ticker"], preferred="Ticker")
if ticker_col is None:
    raise ValueError(f"No ticker-like column found in fund summary. Columns: {funds.columns}")

ai_flag_col = find_col(funds, keywords=["is_ai", "ai"], preferred="is_ai")
if ai_flag_col is None:
    raise ValueError(f"No AI-flag column found in fund summary. Columns: {funds.columns}")
#find ticker and AI flag columns

funds[ticker_col] = funds[ticker_col].astype(str)
#ensure tickers are strings to match return columns

common = sorted(set(daily.columns).intersection(set(funds[ticker_col])))
if len(common) == 0:
    raise ValueError("No common tickers between daily returns and fund summary.")
#only keep tickers found in both places

daily = daily[common]
#removes any return columns without metadata

ai_tickers = funds.loc[funds[ai_flag_col] == True, ticker_col].tolist() # build lists based on is_ai
trad_tickers = funds.loc[funds[ai_flag_col] == False, ticker_col].tolist()

ai_tickers = [t for t in ai_tickers if t in common]
trad_tickers = [t for t in trad_tickers if t in common]
#only keep tickers that truly exist in daily returns

print(f"Tickers in returns: {len(common)}")
print(f"AI ETFs: {len(ai_tickers)} | Traditional ETFs: {len(trad_tickers)}")

if len(ai_tickers) == 0 or len(trad_tickers) == 0:
    raise ValueError("Need at least 1 AI and 1 Traditional ETF. Check your AI flag column.")

#must have both groups or allocation makes no sense



# BUILD AI/TRAD PORTFOLIO RETURNS (EQUAL-WEIGHT)

print("Building equal-weight AI and Traditional portfolios...")

ret_ai = daily[ai_tickers].mean(axis=1, skipna=True).rename("ret_ai")
ret_tr = daily[trad_tickers].mean(axis=1, skipna=True).rename("ret_trad")

#Reduce 82(83) ETFs into two daily return series:
#      an equal-weight AI portfolio
#      an equal-weight Traditional portfolio
# So later not allocating among 82 things; allocating between two baskets



# DOWNLOAD MACRO FEATURES (YAHOO ONLY)

print("Downloading macro features (Yahoo only, Python 3.13 safe)...")

start = daily.index.min().date()
end   = daily.index.max().date()
#	use the same date range as our ETF data

# We use proxies available on Yahoo:
# ^VIX  = volatility (fear)
# SPY   = market return (S&P 500)
# ^TNX  = 10Y yield (scaled by 10)
# ^IRX  = 13-week T-bill yield (short rate, scaled by 100)
# HYG   = High-yield bond ETF (credit stress proxy via returns)
tickers = ["^VIX", MARKET_TICKER, "^TNX", "^IRX", "HYG"]

# Download from Yahoo
raw = yf.download(tickers, start=start, end=end, progress=False, group_by="column", auto_adjust=False)

if raw.empty:
    raise RuntimeError("Yahoo download returned empty data. Check internet or tickers.")

# yfinance may return:
# 1 MultiIndex columns: ('Adj Close', 'SPY') etc.
# 2 Single-level columns for single ticker
# do bothj

# if multi index, pick price field level safely, Use “Adj Close” if available, else fall back to “Close”
# Then rename columns to friendly names
if isinstance(raw.columns, pd.MultiIndex):
    fields = raw.columns.get_level_values(0).unique().tolist()
    price_field = "Adj Close" if "Adj Close" in fields else "Close"
    prices = raw[price_field].copy()
else:
    # single ticker case: just use Adj Close if exists, else Close
    if "Adj Close" in raw.columns:
        prices = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
    elif "Close" in raw.columns:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise RuntimeError(f"Could not find 'Adj Close' or 'Close' in Yahoo data columns: {list(raw.columns)}")

# now 'prices' should be a DataFrame with columns as tickers (or renamed tickers)
prices = prices.rename(columns={
    "^VIX": "VIX",
    MARKET_TICKER: "SPY",
    "^TNX": "TNX_10Y",
    "^IRX": "IRX_3M",
    "HYG": "HYG",
})

# ensure expected columns exist (some tickers may fail)
needed = ["VIX", "SPY", "TNX_10Y", "IRX_3M", "HYG"]
missing = [c for c in needed if c not in prices.columns]
if missing:
    raise RuntimeError(
        f"Missing expected Yahoo series: {missing}\n"
        f"Available columns: {list(prices.columns)}\n"
        "Sometimes Yahoo blocks a ticker temporarily; try rerunning or change proxies."
    )


# SPY daily returns + 20 day compounded return
spy_ret = prices["SPY"].pct_change()
spy_ret_20 = (1 + spy_ret).rolling(20).apply(np.prod, raw=True) - 1
spy_ret_20 = spy_ret_20.rename("sp500_ret_20d")
#pct_change() gives daily return, rolling product creates compounded 20-day return

# convert yields:
# TNX is 10x the yield (e.g., 45.2 means 4.52%)
tnx = (prices["TNX_10Y"] / 10.0).rename("y10")

# IRX is usually already percentish; dividing by 100 makes it decimal
irx = (prices["IRX_3M"] / 100.0).rename("y3m")

yield_slope = (tnx - irx).rename("yield_slope_10y_3m")
#slope < 0 means inverted curve (tight conditions)


# credit stress proxy: HYG return (more negative on credit stress days): HYG dros - credit stressed
credit_stress = prices["HYG"].pct_change().rename("credit_stress")

vix = prices["VIX"].rename("VIX")

features = pd.concat([vix, spy_ret_20, yield_slope, credit_stress], axis=1)
features = features.reindex(daily.index).ffill().dropna()

# reindex to ETF date index.
# forward fill missing macro days (common because of weekends/holidays)
# drop early NaNs (especially from rolling 20-day return)

print("Feature sample:")
print(features.head())

# KMEANS REGIME CLUSTERING
print(f"Clustering regimes with KMeans (k={N_REGIMES})...")

X = features.values
Xz = StandardScaler().fit_transform(X)
#convert features table to a matrix X. standardize each column to mean=0, std=1

kmeans = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=25)
cluster = kmeans.fit_predict(Xz)
#fnd 3 clusters in the standardized feature space, then output cluster labels (0/1/2) for each date

reg = pd.Series(cluster, index=features.index, name="cluster") # make it a date indexed series

# Label clusters by avg VIX (low -> calm, high -> crisis)
cluster_vix = features.groupby(reg)["VIX"].mean().sort_values()
ordered_clusters = cluster_vix.index.tolist()


names = ["calm", "stressed", "crisis"]
cluster_to_regime = {cl: names[min(i, len(names)-1)] for i, cl in enumerate(ordered_clusters)}

regime = reg.map(cluster_to_regime).rename("regime")
# convert 0/1/2  to calm/stressed/crisis resp
print("Cluster -> regime mapping (by avg VIX):")
for cl in ordered_clusters:
    print(f"  cluster {cl} | avg VIX={cluster_vix.loc[cl]:.2f} -> {cluster_to_regime[cl]}")



# PREP BACKTEST DATA (NO LOOKAHEAD)

print("Preparing backtest dataset (shift regime by 1 day to avoid lookahead)...")

df = pd.concat([ret_ai, ret_tr, features, regime], axis=1).dropna()
#combine everything into one dataset:
# AI return
# Traditional returns
# macro features
# regime label

df["regime_lag"] = df["regime"].shift(1)
#we only know today’s regime after seeing today’s macro data
# so to avoid cheating, we use yesterday’s regime to set today’s portfolio weights
# that’s what shift(1) does here

# drop rows where lag is missing
df = df.dropna(subset=["regime_lag"])


print("Regime counts (lagged):")
print(df["regime_lag"].value_counts())


# PER-REGIME METRICS: AI vs TRAD

print("\nPer-regime metrics (AI vs Traditional):")

regimes_order = ["calm", "stressed", "crisis"]
present = [r for r in regimes_order if r in set(df["regime_lag"])]
# ensures we only evaluate regimes that exist

rows = []
for r in present:
    sub = df[df["regime_lag"] == r]
    for label, series in [("AI", sub["ret_ai"]), ("Traditional", sub["ret_trad"])]:
        rows.append({
            "regime": r,
            "group": label,
            "mean_daily": series.mean(),
            "sortino": sortino_ratio(series.values),
            "cvar_95": cvar(series.values, TAIL_ALPHA),
        })

metrics_df = pd.DataFrame(rows)
print(metrics_df)

# we take all days that were classified as regime r:
# then compute:
# mean daily return
# Sortino
# CVaR95
# for AI and Traditional



# OPTIMAL ALLOCATION PER REGIME (MAX SORTINO)

print("\nChoosing optimal AI weight per regime (maximize Sortino)...")

weights = np.round(np.arange(0, 1 + WEIGHT_GRID_STEP, WEIGHT_GRID_STEP), 2)
# list of candidate weights (0.0, 0.1, ..., 1.0)
alloc = {}
for r in present:
    sub = df[df["regime_lag"] == r]
    best = {"w_ai": None, "sortino": -1e18, "cvar_95": None, "mean_daily": None}
#for each regime
    for w in weights:
        # mix = w*AI + (1-w)*Traditional
        mix = w * sub["ret_ai"].values + (1 - w) * sub["ret_trad"].values #	build mixed returns:
        s = sortino_ratio(mix) #compute sortino
        if np.isnan(s): #skip invalid
            continue
        if s > best["sortino"]: #keep the best w 
            best = {
                "w_ai": float(w),
                "sortino": float(s),
                "cvar_95": float(cvar(mix, TAIL_ALPHA)),
                "mean_daily": float(np.nanmean(mix)),
            }

    alloc[r] = best # store best allocaion for regime r


for r in present:
    b = alloc[r]
    print(f"  {r.upper():8s} -> AI {b['w_ai']:.2f} | Sortino {b['sortino']:.3f} | CVaR95 {b['cvar_95']:.4f}")

      # calm - best AI weight
    # stressed - best AI weight
    # crisis - best AI weight


# BACKTEST DYNAMIC VS STATIC
print("\nBacktesting strategy ...")

#  knobs to reduce turnover 
N_ENTER = 2          # require 2 consecutive crisis days to ENTER
N_EXIT  = 7          # require 7 consecutive non crisis days to EXIT (stickier)
WEEKLY_REBAL = True  # trade only weekly
MAX_STEP = 0.20      # max change in AI weight per rebalance (20% per trade)

# target AI weight based on regime allocation table
df["w_ai_target"] = df["regime_lag"].map(lambda r: alloc[r]["w_ai"])

# binary crisis indicator
is_crisis = (df["regime_lag"] == "crisis").astype(int)

# enter crisis only after N_ENTER consecutive crisis days.
# Exit crisis only after N_EXIT consecutive non crisis days.
crisis_run = is_crisis.groupby((is_crisis != is_crisis.shift()).cumsum()).cumcount() + 1
noncrisis_run = (1 - is_crisis).groupby(((1 - is_crisis) != (1 - is_crisis).shift()).cumsum()).cumcount() + 1

# smooth state machine: “sticky crisis” logic
state = []
in_crisis = False
for i in range(len(df)):
    if not in_crisis:
        # can enter?
        if is_crisis.iat[i] == 1 and crisis_run.iat[i] >= N_ENTER:
            in_crisis = True
    else:
        # can exit?
        if is_crisis.iat[i] == 0 and noncrisis_run.iat[i] >= N_EXIT:
            in_crisis = False
    state.append(in_crisis)

# store smoothed crisis state
df["crisis_smooth"] = pd.Series(state, index=df.index)

# Converts smoothed crisis state into a smoothed target weight:
# if crisis_smooth => use the crisis allocation
# else => use the best non crisis allocation (usually calm/stressed, so 0)
w_crisis = alloc["crisis"]["w_ai"] if "crisis" in alloc else 0.0
# pick "stressed" if present else "calm" else 0
noncr = "stressed" if "stressed" in alloc else ("calm" if "calm" in alloc else None)
w_noncrisis = alloc[noncr]["w_ai"] if noncr else 0.0

# smoothed target weight (only depends on crisis vs non-crisis)
df["w_ai_smooth_target"] = np.where(df["crisis_smooth"], w_crisis, w_noncrisis)

# rebalance schedule: Fridays only if weekly, otherwise every day
if WEEKLY_REBAL:
    rebalance_day = (df.index.weekday == 4)  # Fri
else:
    rebalance_day = np.ones(len(df), dtype=bool)
 
 
# build actual dynamic weights w[t], limited by MAX_STEP
w = np.zeros(len(df), dtype=float)
w[0] = df["w_ai_smooth_target"].iat[0]

for t in range(1, len(df)):
    w_prev = w[t-1]

    if rebalance_day[t]:
        target = float(df["w_ai_smooth_target"].iat[t])
        delta = target - w_prev
        # limit how much we can change in one rebalance
        delta_limited = np.clip(delta, -MAX_STEP, MAX_STEP)
        w[t] = w_prev + delta_limited
    else:
        # hold weight between rebalances
        w[t] = w_prev

# store dynamic weights
df["w_ai_dyn"] = w
df["w_tr_dyn"] = 1 - df["w_ai_dyn"]

# compute dynamic strategy daily returns
df["ret_dyn"] = df["w_ai_dyn"] * df["ret_ai"] + df["w_tr_dyn"] * df["ret_trad"]
# static benchmark = 50/50 every day
df["ret_static"] = 0.5 * df["ret_ai"] + 0.5 * df["ret_trad"]

# turnover = sum of daily absolute weight changes
turnover_daily = df["w_ai_dyn"].diff().abs().dropna()
total_turnover = turnover_daily.sum()
annual_turnover = turnover_daily.mean() * TRADING_DAYS
# summarize performance
perf_dyn = perf_summary(df["ret_dyn"].values, "Dynamic Regime Strategy (Turnover-Reduced)")
perf_sta = perf_summary(df["ret_static"].values, "Static 50/50 Benchmark")

print("\n=== Performance Summary ===")
for p in [perf_dyn, perf_sta]:
    print(f"\n{p['name']}")
    print(f"  Annual return:   {p['ann_return']:.4f}")
    print(f"  Annual vol:      {p['ann_vol']:.4f}")
    print(f"  Sortino:         {p['sortino']:.4f}")
    print(f"  Max drawdown:    {p['max_drawdown']:.4f}")
    print(f"  CVaR 95%:        {p['cvar_95']:.4f}")

print("\n=== Turnover (Dynamic) ===")
print(f"Total turnover (sum |Δw_ai|): {total_turnover:.2f}")
print(f"Approx annual turnover:       {annual_turnover:.2f}x")

# BUY / HOLD / SELL for an ETF ticker

# helper: compute compounded return over last 20 days
def _last_20d_compounded(series):
    s = series.dropna()
    if len(s) < 20:
        return np.nan
    return (1.0 + s.tail(20)).prod() - 1.0

def suggest_action_for_ticker(ticker: str):
    t = str(ticker).strip().upper() # normalize user input

    # ensure ticker exists
    if t not in common:
        return {
            "ok": False,
            "msg": f"Ticker '{t}' not found in your dataset. Use 'list' to see available tickers."
        }

    # group membership
    group = "AI" if t in ai_tickers else "Traditional"

    # latest state of model
    latest_date = df.index[-1]
    latest_regime_lag = df["regime_lag"].iloc[-1]
    w_ai_now = float(df["w_ai_dyn"].iloc[-1])
    w_ai_prev = float(df["w_ai_dyn"].iloc[-2]) if len(df) >= 2 else w_ai_now
    delta_w = w_ai_now - w_ai_prev

    # rotation-based signal:
    # - If AI weight increased -> BUY AI / SELL Traditional
    # - If AI weight decreased -> SELL AI / BUY Traditional
    # - If unchanged -> HOLD
    # BASE SIGNAL (momentum vs group basket, regime-aware)
        
    REL_THRESH = 0.01  # 1% threshold for “meaningful” outperformance

    # preferred group depends on regime:
    # treat crisis as favoring AI, otherwise favor Traditional    
    preferred_group = "AI" if latest_regime_lag == "crisis" else "Traditional"

    # ticker momentum over last 20 trading days
    ticker_mom20 = _last_20d_compounded(daily[t])

    # group basket momentum over last 20 trading days
    if group == "AI":
        basket = daily[ai_tickers].mean(axis=1, skipna=True)
    else:
        basket = daily[trad_tickers].mean(axis=1, skipna=True)

    basket_mom20 = _last_20d_compounded(basket)

    base_signal = "HOLD"
    mom_note = "n/a"

    # only compute signal if both momentums are valid numbers
    if np.isfinite(ticker_mom20) and np.isfinite(basket_mom20):
        gap = ticker_mom20 - basket_mom20
        mom_note = f"{ticker_mom20:+.2%} (ticker) vs {basket_mom20:+.2%} (group), gap {gap:+.2%}"

        if group != preferred_group:
            # if we r in the “wrong” group for this regime, be conservative
            base_signal = "SELL" if gap < -REL_THRESH else "HOLD"
        else:
            # if we r in the preferred group, buy winners / sell losers
            if gap > REL_THRESH:
                base_signal = "BUY"
            elif gap < -REL_THRESH:
                base_signal = "SELL"
            else:
                base_signal = "HOLD"

    final_signal = base_signal

    # strength of signal based on how much the model weight moved today
    eps = 1e-9
    abs_dw = abs(delta_w)
    if abs_dw >= (MAX_STEP * 0.9):
        strength = "strong"
    elif abs_dw >= (MAX_STEP * 0.3):
        strength = "moderate"
    elif abs_dw > eps:
        strength = "weak"
    else:
        strength = "flat"

    # rotation direction (is the model shifting toward AI or Traditional today?)
    rotation = "toward AI" if delta_w > eps else ("toward Traditional" if delta_w < -eps else "no rotation")

    return {
        "ok": True,
        "ticker": t,
        "group": group,
        "date": str(latest_date.date()),
        "regime_lag": latest_regime_lag,
        "w_ai_now": w_ai_now,
        "w_ai_prev": w_ai_prev,
        "delta_w": delta_w,
        "rotation": rotation,
        "base_signal": base_signal,
        "final_signal": final_signal,
        "strength": strength,
        "momentum_20d_note": mom_note,
    }


def print_available_tickers(n=82):
    # show up to n tickers (sorted)
    ticks = sorted(list(common))
    print(f"\nAvailable tickers ({len(ticks)} total):")
    print(", ".join(ticks[:n]) + (" ..." if len(ticks) > n else ""))


def interactive_signal_loop():
    print("\n==============================")
    print("Interactive ETF Signal (Model)")
    print("==============================")
    print("Type an ETF ticker (e.g., QQQ, SPY...), or:")
    print("  - 'list' to show tickers")
    print("  - 'state' to show latest model state")
    print("  - 'quit' to exit\n")

    while True:
        user_in = input("Enter ticker / command: ").strip()
        if not user_in:
            continue

        cmd = user_in.strip().lower()

        if cmd in ("quit", "exit", "q"):
            print("Bye.")
            break

        if cmd == "list":
            print_available_tickers()
            continue

        if cmd == "state":
            latest_date = df.index[-1]
            print(f"\nLatest date in backtest: {latest_date.date()}")
            print(f"Latest lagged regime:    {df['regime_lag'].iloc[-1]}")
            print(f"AI weight (prev -> now): {df['w_ai_dyn'].iloc[-2]:.2f} -> {df['w_ai_dyn'].iloc[-1]:.2f}")
            print(f"Rotation today:          "
                  f"{'toward AI' if df['w_ai_dyn'].iloc[-1] > df['w_ai_dyn'].iloc[-2] else ('toward Traditional' if df['w_ai_dyn'].iloc[-1] < df['w_ai_dyn'].iloc[-2] else 'no rotation')}")
            print("")
            continue

        # otherwise treat as ticker
        out = suggest_action_for_ticker(user_in)
        if not out["ok"]:
            print(out["msg"])
            continue

        print("\n--------------------------------")
        print(f"Ticker:     {out['ticker']}   | Group: {out['group']}")
        print(f"Date:       {out['date']}     | Regime (lagged): {out['regime_lag']}")
        print(f"Model w_AI:  {out['w_ai_prev']:.2f} -> {out['w_ai_now']:.2f}  (Δ {out['delta_w']:+.2f}, {out['strength']})")
        print(f"Rotation:   {out['rotation']}")
        print(f"Signal:     {out['final_signal']}  (base: {out['base_signal']})")
        print(f"Momentum20: {out['momentum_20d_note']}")
        print("--------------------------------\n")


interactive_signal_loop()

# PLOTS (not used here)
print("\nPlotting cumulative returns and regime timeline...")

cum_dyn = (1 + df["ret_dyn"]).cumprod() # growth of $1 invested
cum_sta = (1 + df["ret_static"]).cumprod()

#Regime timeline:
	#maps calm/stressed/crisis to 0/1/2 and plots

plt.figure(figsize=(10, 5))
plt.plot(cum_dyn.index, cum_dyn, label="Dynamic")
plt.plot(cum_sta.index, cum_sta, label="Static 50/50")
plt.title("Cumulative Growth of $1")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

reg_num_map = {"calm": 0, "stressed": 1, "crisis": 2}
reg_num = df["regime_lag"].map(reg_num_map)

plt.figure(figsize=(10, 2.5))
plt.plot(reg_num.index, reg_num.values)
plt.yticks([0, 1, 2], ["calm", "stressed", "crisis"])
plt.title("Detected Regime Over Time (lagged, no lookahead)")
plt.grid(True)
plt.tight_layout()
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# PRACTICAL INTERPRETATION
print("\n=== Practical Interpretation ===")

macro_cols = ["VIX", "sp500_ret_20d", "yield_slope_10y_3m", "credit_stress"]
# median macro levels per regime (lagged)
by_regime = df.groupby("regime_lag")[macro_cols].median().loc[present]

print("\nMedian macro levels by regime (lagged):")
print(by_regime)

print("\nRule of thumb (heuristic):")
print(f"- If VIX > {PRACTICAL_VIX_LEVEL:.0f} AND yield slope (10Y-3M) < 0, markets are typically stressed/inverted.")
print("- In those conditions, your model tends to shift toward the regime(s) with higher AI weight.")

print("\nYour learned regime allocations:")
for r in present:
    print(f"  {r.upper():8s}: AI {alloc[r]['w_ai']:.0%} / Traditional {1-alloc[r]['w_ai']:.0%}")

print("\nDone.")


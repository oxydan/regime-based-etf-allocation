import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "regime_results.pkl"
OUT_DIR = "plots_out"

def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def compute_cum(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()

def compute_drawdown(returns: pd.Series) -> pd.Series:
    equity = compute_cum(returns)
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd

def to_regime_numeric(regime_series: pd.Series):
    # stable mapping for k=3
    reg_map = {"calm": 0, "stressed": 1, "crisis": 2}
    return regime_series.map(reg_map)

def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}' not found in df. Available: {list(df.columns)}")
    return df[col]

def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Could not find '{RESULTS_PATH}'.\n"
            "Run your main backtest script first (with the saving block added)."
        )

    with open(RESULTS_PATH, "rb") as f:
        res = pickle.load(f)

    df = res["df"].copy()
    alloc = res["alloc"]
    metrics_df = res["metrics_df"].copy()

    ensure_out_dir(OUT_DIR)

    #required series from df
    ret_dyn = safe_series(df, "ret_dyn")
    ret_static = safe_series(df, "ret_static")
    w_ai_dyn = safe_series(df, "w_ai_dyn")
    regime_lag = safe_series(df, "regime_lag")

    # Plot 1: Cumulative performance
    #This is the standard “headline” plot. It makes the main claim visible: 
    #the dynamic strategy slightly outperforms the static benchmark
    cum_dyn = compute_cum(ret_dyn)
    cum_sta = compute_cum(ret_static)

    plt.figure(figsize=(11, 5))
    plt.plot(cum_dyn.index, cum_dyn, label="Dynamic regime strategy")
    plt.plot(cum_sta.index, cum_sta, label="Static 50/50 benchmark")
    plt.title("Cumulative Growth of $1")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_cumulative_growth.png"), dpi=150)
    plt.show()

    # Plot 2: Drawdowns (risk visibility)
    #downside risk and regime switching. 
    # Drawdown shows the lived experience of risk over time and complements CVaR/Sortino
    dd_dyn = compute_drawdown(ret_dyn)
    dd_sta = compute_drawdown(ret_static)

    plt.figure(figsize=(11, 4.5))
    plt.plot(dd_dyn.index, dd_dyn, label="Dynamic drawdown")
    plt.plot(dd_sta.index, dd_sta, label="Static 50/50 drawdown")
    plt.title("Drawdown Over Time")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_drawdowns.png"), dpi=150)
    plt.show()

    # Plot 3: Regime timeline + AI weight (mechanism)
    #This explains the mechanism: when the model is in calm/stressed/crisis
    # how the allocation actually moves. It makes the strategy interpretable
    reg_num = to_regime_numeric(regime_lag)

    plt.figure(figsize=(11, 4.8))
    plt.plot(w_ai_dyn.index, w_ai_dyn.values, label="AI weight (dynamic)")
    plt.title("Dynamic AI Weight With Regime Timeline")
    plt.xlabel("Date")
    plt.ylabel("AI weight")

    # Add regime as a light background band (using numeric levels)
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(reg_num.index, reg_num.values, alpha=0.35, linewidth=1.0, label="Regime (0 calm, 1 stressed, 2 crisis)")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["calm", "stressed", "crisis"])
    ax.grid(True)

    # legends for both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_weight_and_regime.png"), dpi=150)
    plt.show()

    # Plot 4: Per-regime Sortino (why allocations differ)
    #This directly explains why optimal weights are what they are 
    # (crisis favored AI, calm/stressed favored traditional)
    # It visually justifies the allocation rule
    pivot_sort = metrics_df.pivot(index="regime", columns="group", values="sortino").reindex(["calm", "stressed", "crisis"])
    pivot_sort = pivot_sort.dropna(how="all")

    fig, ax = plt.subplots(figsize=(9.5, 5))
    pivot_sort.plot(kind="bar", ax=ax)

    ax.set_title("Per-Regime Sortino: AI Basket vs Traditional Basket")
    ax.set_xlabel("Regime (lagged)")
    ax.set_ylabel("Sortino ratio")
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_sortino_by_regime.png"), dpi=150)
    plt.show()
    # Plot 5: Turnover and trading intensity (cost proxy)
    #since turnover is a practical constraint 
    # (report annual turnover), this plot shows when trading 
    # intensity happens and whether it clusters in stress periods
    
    # daily abs change in AI weight
    w_change = w_ai_dyn.diff().abs()
    w_change = w_change.dropna()

    rolling_turnover_63d = w_change.rolling(63).sum()  # ~quarter of trading days

    plt.figure(figsize=(11, 4.8))
    plt.plot(w_change.index, w_change.values, label="Daily |Δ AI weight|")
    plt.plot(rolling_turnover_63d.index, rolling_turnover_63d.values, label="Rolling 63D turnover sum")
    plt.title("Turnover Profile (Trading Intensity)")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_turnover_profile.png"), dpi=150)
    plt.show()
    # Print a quick allocation recap
    print("\nAllocation table (learned per regime):")
    for r in ["calm", "stressed", "crisis"]:
        if r in alloc:
            print(f"  {r.upper():8s}: w_AI = {alloc[r]['w_ai']:.2f} | Sortino = {alloc[r]['sortino']:.3f} | CVaR = {alloc[r]['cvar_95']:.4f}")

    print(f"\nSaved 5 plots to: {OUT_DIR}/")

if __name__ == "__main__":
    main()
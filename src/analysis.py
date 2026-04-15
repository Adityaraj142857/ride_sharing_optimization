"""
=============================================================================
analysis.py
=============================================================================
Module  : Member 4 — Analysis, OR Concepts & Presentation
Project : Revenue Maximization for Ride-Sharing using LP & Price Discretization
=============================================================================
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

log        = logging.getLogger(__name__)
FIGURES_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")

PRICE_LEVELS = [100, 150, 200]
PRICE_NAMES  = ["Low (₹100)", "Medium (₹150)", "High (₹200)"]


# ── Sensitivity Analysis ──────────────────────────────────────────────────────

def sensitivity_analysis_demand(lp_input: pd.DataFrame,
                                 solve_fn,
                                 deltas: list = [-0.3, -0.15, 0, 0.15, 0.3]) -> pd.DataFrame:
    """
    Vary total demand by ±δ% and observe revenue impact.

    Parameters
    ----------
    lp_input  : base LP input dataframe
    solve_fn  : callable — build_and_solve from optimization.py
    deltas    : list of fractional changes to apply

    Returns
    -------
    pd.DataFrame  [delta_pct, revenue]
    """
    log.info("Running demand sensitivity analysis …")
    records = []
    demand_cols = ["demand_low", "demand_medium", "demand_high"]

    for delta in deltas:
        df_mod = lp_input.copy()
        df_mod[demand_cols] = (lp_input[demand_cols] * (1 + delta)).round().astype(int)
        result = solve_fn(df_mod)
        records.append({
            "delta_pct": delta * 100,
            "revenue"  : result["objective"],
        })
        log.info("  Δ demand = %+.0f%%  →  Revenue = ₹%.2f",
                 delta * 100, result["objective"])

    return pd.DataFrame(records)


def sensitivity_analysis_supply(lp_input: pd.DataFrame,
                                 solve_fn,
                                 deltas: list = [-0.3, -0.15, 0, 0.15, 0.3]) -> pd.DataFrame:
    """Same as above but varies driver supply."""
    log.info("Running supply sensitivity analysis …")
    records = []
    for delta in deltas:
        df_mod = lp_input.copy()
        df_mod["drivers"] = (lp_input["drivers"] * (1 + delta)).round().astype(int)
        result = solve_fn(df_mod)
        records.append({
            "delta_pct": delta * 100,
            "revenue"  : result["objective"],
        })
    return pd.DataFrame(records)


# ── Graphical Method (2-variable LP) ─────────────────────────────────────────

def plot_graphical_lp(zone: int = 1, time_slot: str = "Peak",
                      demand_low: int = 120, demand_high: int = 60,
                      drivers: int = 80, save: bool = True) -> None:
    """
    Graphical LP for the simplified 2-price-level case in a single zone.

    Variables : x1 = rides at ₹100,  x2 = rides at ₹200
    Objective : max Z = 100·x1 + 200·x2
    Constraints:
        x1 ≤ demand_low   (demand constraint, price 1)
        x2 ≤ demand_high  (demand constraint, price 2)
        x1 + x2 ≤ drivers (driver supply)
        x1, x2 ≥ 0
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    x1 = np.linspace(0, max(demand_low, drivers) * 1.15, 500)

    # Feasible region boundaries
    x2_demand1 = np.full_like(x1, demand_high)    # x2 ≤ demand_high (horizontal)
    x2_drivers = np.clip(drivers - x1, 0, None)   # x1 + x2 ≤ drivers

    # Fill feasible region
    x2_upper = np.minimum(x2_demand1, x2_drivers)
    x2_upper = np.clip(x2_upper, 0, None)
    ax.fill_between(x1, 0, x2_upper,
                    where=x1 <= demand_low,
                    alpha=0.2, color="#1976D2", label="Feasible Region")

    # Plot constraints
    ax.axvline(demand_low, color="#E53935", linestyle="--", lw=1.8,
               label=f"x₁ ≤ {demand_low}  (demand, price low)")
    ax.axhline(demand_high, color="#43A047", linestyle="--", lw=1.8,
               label=f"x₂ ≤ {demand_high}  (demand, price high)")
    ax.plot(x1, x2_drivers, color="#7B1FA2", lw=1.8,
            label=f"x₁ + x₂ ≤ {drivers}  (driver supply)")

    # Objective contour at optimal
    opt_x1 = min(demand_low, drivers)
    opt_x2 = min(demand_high, max(0, drivers - opt_x1))
    Z_opt  = 100 * opt_x1 + 200 * opt_x2
    x2_obj = (Z_opt - 100 * x1) / 200
    ax.plot(x1, x2_obj, "k--", lw=1.2, alpha=0.6,
            label=f"Z = {Z_opt:,}  (isoprofit line)")

    # Optimal point
    ax.plot(opt_x1, opt_x2, "r*", markersize=16, zorder=5,
            label=f"Optimal: ({opt_x1}, {opt_x2})  Z=₹{Z_opt:,}")

    ax.set_xlim(0, max(demand_low, drivers) * 1.15)
    ax.set_ylim(0, max(demand_high, drivers) * 1.15)
    ax.set_xlabel("x₁  (rides at ₹100)", fontsize=11)
    ax.set_ylabel("x₂  (rides at ₹200)", fontsize=11)
    ax.set_title(f"Graphical LP — Zone {zone} | {time_slot}\n"
                 f"max Z = 100·x₁ + 200·x₂", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.35)

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"graphical_lp_zone{zone}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)
    plt.show()


# ── Result visualization ──────────────────────────────────────────────────────

def plot_revenue_by_zone(result: dict, save: bool = True) -> None:
    """Bar chart of total revenue per zone from the optimal solution."""
    rides = result["rides"]
    zone_rev = (
        rides.groupby("zone")["revenue"].sum()
             .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(zone_rev)))
    bars = ax.bar([f"Z{z}" for z in zone_rev.index], zone_rev.values,
                  color=colors, edgecolor="white")
    ax.set_xlabel("Zone", fontsize=11)
    ax.set_ylabel("Revenue (₹)", fontsize=11)
    ax.set_title("Optimal Revenue by Zone", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    for bar, val in zip(bars, zone_rev.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01, f"₹{val:,.0f}",
                ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "revenue_by_zone.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)
    plt.show()


def save_results(result: dict) -> None:
    """Export rides + prices tables to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result["rides"].to_csv(RESULTS_DIR / "optimal_rides.csv", index=False)
    result["prices"].to_csv(RESULTS_DIR / "selected_prices.csv", index=False)
    if result.get("shadow_prices"):
        shadow_df = pd.DataFrame(
            [{"zone": z, "time_slot": t, "shadow_price": pi}
             for (z, t), pi in result["shadow_prices"].items()]
        ).sort_values(["zone", "time_slot"])
        shadow_df.to_csv(RESULTS_DIR / "shadow_prices.csv", index=False)
    log.info("Results saved to %s", RESULTS_DIR)
"""
=============================================================================
demand_modeling.py
=============================================================================
Module  : Member 2 — Demand Modeling & Pricing Logic
Project : Revenue Maximization for Ride-Sharing using LP & Price Discretization
=============================================================================
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

log = logging.getLogger(__name__)

PRICE_LEVELS   = [100, 150, 200]
PRICE_LABELS   = ["Low (₹100)", "Medium (₹150)", "High (₹200)"]
DEMAND_FACTORS = [1.00, 0.75, 0.50]
FIGURES_DIR    = Path("outputs/figures")


def get_price_demand_table(base_demand: int) -> pd.DataFrame:
    """
    Return a DataFrame showing demand at each price level for a given base demand.

    Parameters
    ----------
    base_demand : int
        Baseline demand at the lowest price point.

    Returns
    -------
    pd.DataFrame  with columns [price, demand, revenue]
    """
    records = []
    for price, factor, label in zip(PRICE_LEVELS, DEMAND_FACTORS, PRICE_LABELS):
        d = round(base_demand * factor)
        records.append({
            "price_label" : label,
            "price"       : price,
            "demand_factor": factor,
            "demand"      : d,
            "revenue"     : price * d,
        })
    return pd.DataFrame(records)


def plot_demand_curve(lp_input: pd.DataFrame, zone: int, time_slot: str,
                      save: bool = True) -> None:
    """
    Plot the price–demand curve for a given zone × time slot.
    """
    row_df = lp_input[(lp_input["zone"] == zone) & (lp_input["time_slot"] == time_slot)]
    if row_df.empty:
        log.warning("No data found for zone=%s, time_slot=%s", zone, time_slot)
        return
    if len(row_df) > 1:
        log.warning(
            "Multiple rows found for zone=%s, time_slot=%s; using the first row.",
            zone,
            time_slot,
        )
    row = row_df.iloc[0]

    demands  = [int(row["demand_low"]),
                int(row["demand_medium"]),
                int(row["demand_high"])]
    revenues = [p * d for p, d in zip(PRICE_LEVELS, demands)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"Zone {zone}  |  {time_slot} Hours — Price–Demand Analysis",
                 fontsize=13, fontweight="bold", y=1.01)

    # Left: demand curve
    ax1 = axes[0]
    ax1.plot(PRICE_LEVELS, demands, "o-", color="#1565C0", linewidth=2.2,
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax1.fill_between(PRICE_LEVELS, demands, alpha=0.12, color="#1565C0")
    ax1.set_xlabel("Price Level (₹)", fontsize=11)
    ax1.set_ylabel("Demand (trip requests)", fontsize=11)
    ax1.set_title("Demand Curve", fontsize=11)
    ax1.set_xticks(PRICE_LEVELS)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for x, y in zip(PRICE_LEVELS, demands):
        ax1.annotate(f"{y}", xy=(x, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9)

    # Right: revenue bar chart
    ax2 = axes[1]
    bars = ax2.bar(PRICE_LABELS, revenues,
                   color=["#43A047", "#FDD835", "#E53935"], edgecolor="white",
                   width=0.55)
    ax2.set_xlabel("Price Level", fontsize=11)
    ax2.set_ylabel("Revenue (₹)", fontsize=11)
    ax2.set_title("Revenue at Each Price Level", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, rev in zip(bars, revenues):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(revenues) * 0.01,
                 f"₹{rev:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"demand_curve_zone{zone}_{time_slot.lower()}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved figure → %s", path)
    plt.show()


def plot_all_zones_heatmap(lp_input: pd.DataFrame, save: bool = True) -> None:
    """
    Heatmap of demand_low values across all zones × time slots.
    """
    pivot = lp_input.pivot_table(
        index="zone", columns="time_slot", values="demand_low", aggfunc="sum"
    )
    fig, ax = plt.subplots(figsize=(7, max(4, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Zone {z}" for z in pivot.index], fontsize=9)
    plt.colorbar(im, ax=ax, label="Demand (trips)")
    ax.set_title("Baseline Demand Heatmap — All Zones", fontsize=12, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, str(int(pivot.values[i, j])),
                    ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "demand_heatmap_all_zones.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved figure → %s", path)
    plt.show()
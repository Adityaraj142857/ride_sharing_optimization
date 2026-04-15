"""
=============================================================================
generate_ampl_data.py
=============================================================================
Converts data/processed/lp_input.csv  →  ampl/data.dat
Run this after data_preparation.py has produced lp_input.csv.

Usage:
    python generate_ampl_data.py
=============================================================================
"""

import pandas as pd
from pathlib import Path

PRICE_MAP = {
    "low"   : 100,
    "medium": 150,
    "high"  : 200,
}

INPUT_CSV  = Path("data/processed/lp_input.csv")
OUTPUT_DAT = Path("ampl/data.dat")


def zone_key(zone: int, time_slot: str) -> str:
    """Create a clean zone identifier for AMPL (no spaces)."""
    return f"{zone}_{time_slot.lower().replace('-', '')}"


def generate(csv_path: Path = INPUT_CSV, dat_path: Path = OUTPUT_DAT) -> None:
    df = pd.read_csv(csv_path)
    dat_path.parent.mkdir(parents=True, exist_ok=True)

    zones = sorted(set(zone_key(r.zone, r.time_slot) for _, r in df.iterrows()))

    lines = []
    lines.append("# =============================================================")
    lines.append("# data.dat  —  Auto-generated from data/processed/lp_input.csv")
    lines.append("# =============================================================")
    lines.append("")

    # PRICE_LEVELS set
    lines.append("set PRICE_LEVELS := low medium high ;")
    lines.append("")

    # ZONES set
    lines.append("set ZONES :=")
    for z in zones:
        lines.append(f"    {z}")
    lines.append(";")
    lines.append("")

    # price parameter
    lines.append("param price :=")
    for name, val in PRICE_MAP.items():
        lines.append(f"    {name:<8} {val}")
    lines.append(";")
    lines.append("")

    # S (driver supply) parameter
    lines.append("param S :=")
    for _, row in df.iterrows():
        key = zone_key(row.zone, row.time_slot)
        lines.append(f"    {key:<25} {int(row.drivers)}")
    lines.append(";")
    lines.append("")

    # D (demand) parameter  — 2-D table format
    lines.append("param D :")
    lines.append("           low        medium     high  :=")
    for _, row in df.iterrows():
        key = zone_key(row.zone, row.time_slot)
        lines.append(
            f"    {key:<25} {int(row.demand_low):<10} "
            f"{int(row.demand_medium):<10} {int(row.demand_high)}"
        )
    lines.append(";")

    dat_path.write_text("\n".join(lines) + "\n")
    print(f"Written: {dat_path}  ({len(df)} zones)")
    print("\nFirst 5 zones in data.dat:")
    for z in zones[:5]:
        row = df.iloc[zones.index(z)]
        print(f"  {z}: drivers={int(row.drivers)}, "
              f"D_low={int(row.demand_low)}, "
              f"D_med={int(row.demand_medium)}, "
              f"D_high={int(row.demand_high)}")


if __name__ == "__main__":
    generate()

"""
=============================================================================
main.py  —  Entry point for the full project pipeline
=============================================================================
Usage
-----
    python main.py --data data/raw/yellow_tripdata_2023-01.csv

Steps
-----
    1. Data Preparation   (Member 1)
    2. Demand Modeling    (Member 2)
    3. Optimization       (Member 3)
    4. Analysis & Output  (Member 4)
=============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from src.data_preparation import run_pipeline
from src.demand_modeling   import plot_demand_curve, plot_all_zones_heatmap
from src.optimization      import build_and_solve, print_solution_summary
from src.analysis          import (plot_graphical_lp, plot_revenue_by_zone,
                                   sensitivity_analysis_demand, save_results)


def main(data_path: str) -> None:
    # ── Step 1 ────────────────────────────────────────────────────────────────
    print("\n🔧  Step 1 / 4  —  Data Preparation")
    lp_input = run_pipeline(data_path)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    print("\n📊  Step 2 / 4  —  Demand Modelling")
    top_zone = int(lp_input.sort_values("demand_low", ascending=False)["zone"].iloc[0])
    plot_demand_curve(lp_input, zone=top_zone, time_slot="Peak")
    plot_all_zones_heatmap(lp_input)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    print("\n⚙️   Step 3 / 4  —  Optimization")
    result = build_and_solve(lp_input)
    print_solution_summary(result)

    # ── Step 4 ────────────────────────────────────────────────────────────────
    print("\n📈  Step 4 / 4  —  Analysis")
    plot_graphical_lp(zone=top_zone, time_slot="Peak",
                      demand_low=int(lp_input[lp_input["zone"] == top_zone]["demand_low"].max()),
                      demand_high=int(lp_input[lp_input["zone"] == top_zone]["demand_high"].max()),
                      drivers=int(lp_input[lp_input["zone"] == top_zone]["drivers"].max()))
    plot_revenue_by_zone(result)
    save_results(result)

    print("\n✅  All steps complete.  Outputs saved to outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ride-Sharing Revenue Optimization Pipeline"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to raw NYC TLC trip data CSV or Parquet file"
    )
    args = parser.parse_args()
    main(args.data)
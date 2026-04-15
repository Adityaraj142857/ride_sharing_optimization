"""
=============================================================================
data_preparation.py
=============================================================================
Module  : Member 1 — Data Engineering & Preprocessing
Project : Revenue Maximization for Ride-Sharing using LP & Price Discretization
Dataset : NYC TLC Yellow Taxi Trip Records
=============================================================================
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
PEAK_MORNING   = (7,  10)   # 07:00 – 10:59
PEAK_EVENING   = (17, 20)   # 17:00 – 20:59
PRICE_LEVELS   = [100, 150, 200]          # ₹ / ride (Low, Medium, High)
DEMAND_FACTORS = [1.00, 0.75, 0.50]      # elasticity multipliers per price level
TOP_N_ZONES    = 10                       # keep only the busiest zones for LP model
SLOT_DURATION_SEC = 4 * 3600             # assumed slot length for 'concurrent' method (4 hrs)
SUPPLY_FACTOR = 0.60                      # scale ride-capacity below "observed" (forces trade-offs)

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ── Helper functions ──────────────────────────────────────────────────────────

def _classify_time_slot(hour: int) -> str:
    """Return 'Peak' or 'Off-Peak' for a given hour (0–23)."""
    if PEAK_MORNING[0] <= hour <= PEAK_MORNING[1]:
        return "Peak"
    if PEAK_EVENING[0] <= hour <= PEAK_EVENING[1]:
        return "Peak"
    return "Off-Peak"


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing."""
    required = {
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "total_amount",
        "trip_distance",
        "VendorID",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")


# ── Main pipeline functions ───────────────────────────────────────────────────

def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load a single month of NYC TLC Yellow Taxi data.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV (or Parquet) file.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified dataframe.
    """
    filepath = Path(filepath)
    log.info("Loading raw data from: %s", filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, low_memory=False)

    log.info("Raw data shape: %s rows × %s cols", *df.shape)
    _validate_dataframe(df)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data-quality filters to the raw trip records.

    Rules applied
    -------------
    - Drop rows with null pickup/dropoff datetimes or location IDs
    - Keep only trips with fare_amount  > 0
    - Keep only trips with trip_distance > 0
    - Keep only trips with trip_duration between 1 min and 4 hours
    - Remove statistical outliers in fare_amount (IQR method)
    """
    log.info("Cleaning data …")
    n_before = len(df)

    # Parse datetimes
    df["tpep_pickup_datetime"]  = pd.to_datetime(df["tpep_pickup_datetime"],  errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

    # Drop nulls in critical columns
    df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime",
                       "PULocationID", "DOLocationID"], inplace=True)

    # Positive fares & distances
    df = df[(df["fare_amount"] > 0) & (df["trip_distance"] > 0)]

    # Trip duration filter
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df = df[(df["trip_duration_min"] >= 1) & (df["trip_duration_min"] <= 240)]

    # Fare outlier removal (IQR)
    Q1, Q3 = df["fare_amount"].quantile(0.25), df["fare_amount"].quantile(0.75)
    IQR    = Q3 - Q1
    df     = df[(df["fare_amount"] >= Q1 - 1.5 * IQR) &
                (df["fare_amount"] <= Q3 + 1.5 * IQR)]

    n_after = len(df)
    log.info("Rows removed during cleaning: %s  (%.1f%%)",
             n_before - n_after, 100 * (n_before - n_after) / n_before)
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive time/zone features required by the LP model.

    New columns
    -----------
    hour        : int   – pickup hour (0–23)
    time_slot   : str   – 'Peak' | 'Off-Peak'
    zone        : int   – pickup location ID (alias for PULocationID)
    """
    log.info("Engineering features …")
    df["hour"]      = df["tpep_pickup_datetime"].dt.hour
    df["time_slot"] = df["hour"].apply(_classify_time_slot)
    df["zone"]      = df["PULocationID"].astype(int)
    return df


def compute_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trip counts as a proxy for demand.

    Assumption (document in report)
    --------------------------------
    Demand_{i,t} = total number of trip requests originating in
                   zone i during time slot t.

    Returns
    -------
    pd.DataFrame  with columns [zone, time_slot, demand]
    """
    log.info("Computing demand per zone × time slot …")
    demand = (
        df.groupby(["zone", "time_slot"])
          .size()
          .reset_index(name="demand")
    )
    return demand


def estimate_drivers(
    df: pd.DataFrame,
    method: str = "trip_count",
    *,
    trips_per_driver_per_slot: int = 6,
) -> pd.DataFrame:
    """
    Estimate driver (supply) availability per zone × time slot.

    ⚠  Driver count is NOT directly available in the NYC TLC dataset.
       This function uses one of two proxy methods (document in report).

    Parameters
    ----------
    method : 'trip_count' | 'concurrent' | 'unique_vendor'
        'trip_count'   — estimate required drivers from observed trip volume:
                         drivers ≈ trips_in_slot / trips_per_driver_per_slot.
                         This aligns supply units with demand units (trips/slot).
                         RECOMMENDED method.
        'concurrent'   — Little's Law approximation for simultaneously active
                         vehicles in a slot of fixed duration:
                         drivers ≈ (trips × avg_trip_duration_sec) / slot_duration_sec
                         Assumes each time slot spans SLOT_DURATION_SEC seconds.
                         ⚠  Old (buggy) formula was 3600 / avg_duration_sec which
                            yields hourly turnovers (~5), NOT a driver count.
        'unique_vendor'— count of unique VendorIDs per group (very rough proxy;
                         VendorID is not a driver identifier in TLC data).

    trips_per_driver_per_slot : int
        Only used when method='trip_count'. Assumed average number of trips a
        single driver can serve in one time slot (e.g., ~6 trips per 4-hour slot).

    Notes on units (important)
    --------------------------
    The optimization variable x represents "rides served per time slot".
    Therefore the driver-side constraint must be expressed as a *ride capacity*
    per slot, not a raw driver headcount.

    This function returns:
    - drivers_count : estimated number of active drivers in the zone × slot
    - drivers       : ride capacity per slot (drivers_count × trips_per_driver_per_slot)

    Returns
    -------
    pd.DataFrame  with columns [zone, time_slot, drivers, drivers_count]
    """
    log.info("Estimating driver supply (method='%s') …", method)

    if method == "trip_count":
        if trips_per_driver_per_slot <= 0:
            raise ValueError("trips_per_driver_per_slot must be a positive integer.")

        trips = (
            df.groupby(["zone", "time_slot"])
            .size()
            .reset_index(name="trips")
        )
        # Use ceil so we don't systematically under-supply capacity.
        trips["drivers_count"] = np.ceil(trips["trips"] / trips_per_driver_per_slot).astype(int)
        trips["drivers"] = (trips["drivers_count"] * trips_per_driver_per_slot).astype(int)
        agg = trips[["zone", "time_slot", "drivers", "drivers_count"]]

    elif method == "concurrent":
        # Little's Law: L = λ × W
        #   L (concurrent drivers) = trip_rate × avg_service_time
        #   trip_rate = trips / slot_duration_sec
        #   → drivers ≈ (trips × avg_duration_sec) / SLOT_DURATION_SEC
        #
        # BUG FIXED: old formula was (3600 / avg_duration_sec) which gives
        # hourly turnovers per vehicle (~5 for a ~700 s trip), NOT driver count.
        # That caused the LP to be capped at ~5 drivers → ~5 rides served per zone.
        df["trip_duration_sec"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds()

        grp = df.groupby(["zone", "time_slot"])
        agg = grp["trip_duration_sec"].mean().reset_index(name="avg_duration_sec")
        trip_counts = grp.size().reset_index(name="trips")
        agg = agg.merge(trip_counts, on=["zone", "time_slot"])

        # concurrent drivers via Little's Law; clip at 1 to avoid zero supply
        agg["drivers"] = (
            (agg["trips"] * agg["avg_duration_sec"]) / SLOT_DURATION_SEC
        ).clip(lower=1).round().astype(int)
        agg["drivers_count"] = agg["drivers"]

    elif method == "unique_vendor":
        agg = (
            df.groupby(["zone", "time_slot"])["VendorID"]
            .nunique()
            .reset_index(name="drivers")
        )
        agg["drivers_count"] = agg["drivers"]
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'trip_count', 'concurrent', or 'unique_vendor'."
        )

    return agg[["zone", "time_slot", "drivers", "drivers_count"]]


# def discretize_demand(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Apply demand-elasticity model for each price level.

#     Assumption (document in report)
#     --------------------------------
#     Use a *zone-specific constant-elasticity* demand curve:

#         D(p) = D(₹100) · (p / 100)^(-e_zone)

#     Where e_zone is assigned by zone demand tier:
#     - high-demand zones (inelastic)   → lower e (price increases can raise revenue)
#     - low-demand zones  (elastic)     → higher e (price increases reduce revenue)

#     Input  : DataFrame with column 'demand'
#     Output : Same DataFrame + demand_low, demand_medium, demand_high columns
#     """
#     log.info("Applying demand discretization across price levels …")
#     zone_totals = df.groupby("zone")["demand"].sum()
#     if len(zone_totals) >= 3:
#         q_low = zone_totals.quantile(0.33)
#         q_high = zone_totals.quantile(0.66)
#     else:
#         q_low = zone_totals.min()
#         q_high = zone_totals.max()

#     def _elasticity_for_zone(z: int) -> float:
#         tot = float(zone_totals.loc[z])
#         if tot >= q_high:
#             return 0.80
#         if tot <= q_low:
#             return 1.40
#         return 1.10

#     e = df["zone"].map(_elasticity_for_zone).astype(float)
#     base = df["demand"].astype(float)
#     p0 = 100.0

#     df["demand_low"] = base.round().astype(int)
#     df["demand_medium"] = (base * (150.0 / p0) ** (-e)).round().astype(int)
#     df["demand_high"] = (base * (200.0 / p0) ** (-e)).round().astype(int)

#     df["demand_medium"] = np.minimum(df["demand_low"], df["demand_medium"]).clip(lower=0).astype(int)
#     df["demand_high"] = np.minimum(df["demand_medium"], df["demand_high"]).clip(lower=0).astype(int)
#     return df

def discretize_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply demand-elasticity model for each price level.

    Improved Logic
    --------------
    - Introduces zone-level heterogeneity
    - Uses demand-based elasticity (high-demand zones are less price-sensitive)
    - Prevents trivial 'always-medium' solution

    Output : Same DataFrame + demand_low, demand_medium, demand_high columns
    """
    log.info("Applying improved demand discretization …")

    # Normalize demand to create elasticity variation
    max_demand = df["demand"].max()
    min_demand = df["demand"].min()

    # Avoid division by zero
    if max_demand == min_demand:
        df["elasticity_factor"] = 0.5
    else:
        df["elasticity_factor"] = (
            (df["demand"] - min_demand) / (max_demand - min_demand)
        )

    # Interpretation:
    # High-demand zones → elasticity_factor ≈ 1 → less sensitive
    # Low-demand zones → elasticity_factor ≈ 0 → more sensitive

    # Low price → full demand
    df["demand_low"] = df["demand"]

    # Medium price → moderate drop
    # Low-demand groups become much more price-sensitive, so ₹100 can be optimal.
    df["demand_medium"] = (
        df["demand"] * (0.35 + 0.40 * df["elasticity_factor"])  # 0.35 .. 0.75
    ).round().astype(int)

    # High price → stronger drop
    # High-demand groups remain relatively inelastic, so ₹200 can be optimal.
    df["demand_high"] = (
        df["demand"] * (0.15 + 0.50 * df["elasticity_factor"])  # 0.15 .. 0.65
    ).round().astype(int)

    return df.drop(columns=["elasticity_factor"])


def filter_top_zones(df: pd.DataFrame, n: int = TOP_N_ZONES) -> pd.DataFrame:
    """
    Keep only the top-N busiest zones to keep the LP tractable.
    """
    log.info("Filtering to top %s zones by total demand …", n)
    top_zones = (
        df.groupby("zone")["demand"]
          .sum()
          .nlargest(n)
          .index
    )
    return df[df["zone"].isin(top_zones)].reset_index(drop=True)


def build_lp_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the final LP-ready table.

    Columns
    -------
    zone, time_slot, drivers,
    demand_low   (D_{i,j=1}),
    demand_medium(D_{i,j=2}),
    demand_high  (D_{i,j=3})
    """
    log.info("Building LP input table …")
    demand  = compute_demand(df)
    drivers = estimate_drivers(df, method="trip_count", trips_per_driver_per_slot=6)
    merged = demand.merge(drivers, on=["zone", "time_slot"], how="left")
    # Filter to top-N zones BEFORE discretising to avoid computing unused columns.
    merged = filter_top_zones(merged)
    merged = discretize_demand(merged)

    # Tighten supply so it is meaningfully binding (otherwise pricing can be trivial).
    merged["drivers"] = np.ceil(merged["drivers"].astype(float) * SUPPLY_FACTOR).clip(lower=1).astype(int)

    # Reorder columns for clarity
    cols = ["zone", "time_slot", "drivers", "drivers_count",
            "demand_low", "demand_medium", "demand_high"]
    merged = merged[cols].sort_values(["zone", "time_slot"]).reset_index(drop=True)
    log.info("LP input table shape: %s rows × %s cols", *merged.shape)
    return merged


def save_processed(df: pd.DataFrame, filename: str = "lp_input.csv") -> Path:
    """
    Save the processed dataset to data/processed/.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    log.info("Saved processed data → %s", out_path)
    return out_path


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_pipeline(raw_filepath: str | Path) -> pd.DataFrame:
    """
    End-to-end data preparation pipeline.

    Steps
    -----
    1. Load raw NYC TLC data
    2. Clean & validate
    3. Engineer features
    4. Build LP input (demand + drivers + price discretization)
    5. Save to data/processed/

    Returns
    -------
    pd.DataFrame  — LP-ready aggregated table
    """
    log.info("=" * 60)
    log.info("Starting Data Preparation Pipeline")
    log.info("=" * 60)

    df       = load_raw_data(raw_filepath)
    df       = clean_data(df)
    df       = engineer_features(df)
    lp_input = build_lp_input(df)
    save_processed(lp_input)

    log.info("Pipeline complete ✓")
    log.info("\n%s", lp_input.head(10).to_string(index=False))
    return lp_input


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_preparation.py <path_to_raw_csv_or_parquet>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
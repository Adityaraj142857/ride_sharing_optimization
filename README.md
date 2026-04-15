# Revenue Maximization for Ride-Sharing
### Using Price Discretization and Linear / Integer Programming

---

## Project Overview

This project formulates and solves a **Mixed-Integer Linear Program (MILP)**
to maximize revenue for a ride-sharing platform by optimally selecting
**discrete price levels** across different zones and time periods.

**Dataset** : NYC TLC Yellow Taxi Trip Records  
**Source**  : https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

---

## Team Division

| Member | Role | Module |
|--------|------|--------|
| Member 1 | Data Engineering & Preprocessing | `src/data_preparation.py` |
| Member 2 | Demand Modelling & Pricing Logic | `src/demand_modeling.py` |
| Member 3 | Optimization Modelling (CORE) | `src/optimization.py` |
| Member 4 | Analysis, OR Concepts & Presentation | `src/analysis.py` |

---

## Directory Structure

```
ride_sharing_optimization/
│
├── data/
│   ├── raw/                  ← Place downloaded NYC TLC .csv / .parquet here
│   ├── processed/            ← Auto-generated LP input table (lp_input.csv)
│   └── external/             ← Zone lookup tables, shapefiles, etc.
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py   ← Member 1
│   ├── demand_modeling.py    ← Member 2
│   ├── optimization.py       ← Member 3
│   └── analysis.py           ← Member 4
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_demand_modeling.ipynb
│   ├── 03_optimization_model.ipynb
│   └── 04_analysis_visualization.ipynb
│
├── outputs/
│   ├── figures/              ← All plots saved here
│   └── results/              ← optimal_rides.csv, selected_prices.csv
│
├── reports/
│   └── assets/               ← Images for final report
│
├── main.py                   ← Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (Jan 2023 recommended)
# https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet

# 3. Place file in data/raw/
mv yellow_tripdata_2023-01.parquet data/raw/

# 4. Run full pipeline
python main.py --data data/raw/yellow_tripdata_2023-01.parquet
```

---

## Mathematical Model

**Decision Variables**

| Symbol | Type | Description |
|--------|------|-------------|
| x_{i,j} | ℝ ≥ 0 | Rides served in zone i at price level j |
| y_{i,j} | {0,1} | 1 if price level j is selected for zone i |

**Objective**

```
max Z = Σ_i Σ_j  p_j · x_{i,j}
```

**Constraints**

| # | Constraint | Meaning |
|---|-----------|---------|
| C1 | x_{i,j} ≤ D_{i,j} · y_{i,j} | Cannot exceed demand; rides only if price selected |
| C2 | Σ_j x_{i,j} ≤ S_i | Total rides ≤ available drivers in zone i |
| C3 | Σ_j y_{i,j} = 1 | Exactly one price level per zone |
| C4 | x_{i,j} ≥ 0, y_{i,j} ∈ {0,1} | Non-negativity & binary |

---

## Key Assumptions (Document in Report)

1. Demand = number of trip requests per zone per time slot
2. Driver supply approximated from mean trip duration (concurrent trips proxy)
3. Demand decreases with price: Low→100%, Medium→75%, High→50%
4. Prices are discretized: ₹100, ₹150, ₹200
5. One price level is applied uniformly across an entire zone × time slot

---

## OR Concepts Covered

- Linear Programming (LP)
- Graphical Method (2-variable case)
- Simplex Method (LP relaxation)
- Duality (shadow price = value of adding one driver)
- Sensitivity Analysis (demand & supply variation)
- Integer Programming (binary price-selection variables)
- Transportation / Assignment Problem (driver allocation)
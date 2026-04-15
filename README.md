# 🚕 Revenue Maximization for Ride-Sharing

### Using Price Discretization, MILP & AMPL Optimization

---

## 📌 Project Overview

This project develops a **data-driven optimization framework** to maximize revenue for a ride-sharing platform by selecting **optimal price levels across zones and time slots**.

The problem is formulated as a **Mixed-Integer Linear Program (MILP)** and solved using:

* **Python (PuLP)** for modeling & prototyping
* **AMPL + HiGHS Solver** for industry-grade optimization

---

## 📊 Dataset

* **Source**: NYC TLC Yellow Taxi Trip Records
* **Link**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
* **Granularity**:

  * Zone-wise
  * Time-slot (Peak / Off-Peak)
  * Trip-level raw data → aggregated into demand/supply

---

## 🎯 Objective

Maximize total revenue:

```
max Z = Σ_i Σ_j  p_j · x_{i,j}  −  penalty · unmet_demand
```

---

## 🧠 Key Idea

* Lower prices → higher demand
* Higher prices → higher revenue per ride
* Limited drivers → supply constraint

👉 The model finds the **optimal trade-off between price and service capacity**

---

## ⚙️ Mathematical Model

### Decision Variables

| Variable | Type  | Meaning                                 |
| -------- | ----- | --------------------------------------- |
| x_{i,j}  | ℝ ≥ 0 | Rides served in zone i at price level j |
| y_{i,j}  | {0,1} | 1 if price level j is selected          |
| u_i      | ℝ ≥ 0 | Unserved demand (lost rides)            |

---

### Objective Function

```
maximize Z =
    Σ_i Σ_j (price[j] * x[i,j])
  − penalty * Σ_i u[i]
```

---

### Constraints

| #   | Constraint                              | Meaning              |
| --- | --------------------------------------- | -------------------- |
| C1  | x[i,j] ≤ D[i,j] * y[i,j]                | Cannot exceed demand |
| C1b | Σ_j x[i,j] + u[i] = Σ_j D[i,j] * y[i,j] | Demand balance       |
| C2  | Σ_j x[i,j] ≤ S[i]                       | Driver supply limit  |
| C3  | Σ_j y[i,j] = 1                          | One price per zone   |
| C4  | x ≥ 0, y ∈ {0,1}, u ≥ 0                 | Feasibility          |

---

## 💡 Important Modeling Improvement

Unlike basic models:

✔ Introduced **unserved demand penalty**
✔ Prevents unrealistic “ignore demand” solutions
✔ Mimics **customer churn / lost business**

---

## 🏗️ Project Architecture

```
ride_sharing_optimization/
│
├── ampl/
│   ├── model.mod              ← MILP formulation (AMPL)
│   ├── data.dat               ← Generated input data
│   ├── run.run                ← Solve + analysis script
│   └── generate_ampl_data.py
│
├── data/
│   ├── raw/                   ← NYC TLC parquet
│   ├── processed/             ← lp_input.csv
│   └── external/
│
├── src/
│   ├── data_preparation.py    ← Cleaning + feature engineering
│   ├── demand_modeling.py     ← Demand estimation (price elasticity)
│   ├── optimization.py        ← PuLP MILP model
│   └── analysis.py            ← Visualization & OR analysis
│
├── outputs/
│   ├── figures/
│   └── results/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Pipeline Flow

```
Raw Taxi Data
    ↓
Data Cleaning & Feature Engineering
    ↓
Demand Estimation (price-based)
    ↓
LP Input Table (zone × time)
    ↓
MILP Optimization
    ↓
Results + Visualization + Sensitivity
```

---

## 📈 Key Results & Insights

### 🔹 Demand Behavior

* Demand decreases with price (elastic)
* Highly sensitive at lower price levels

---

### 🔹 Pricing Strategy

* High-demand zones → higher prices (₹200)
* Moderate zones → medium prices (₹150)
* Low-demand zones → low prices (₹100)

---

### 🔹 Core Finding

> Revenue is constrained more by **driver availability** than demand.

---

### 🔹 Shadow Prices (Duality)

* Positive shadow price → drivers are scarce
* Zero shadow price → excess drivers

👉 Interpretation:

> Shadow price = **value of adding one extra driver**

---

### 🔹 Sensitivity Analysis

Tested impact of:

* Driver supply increase (+10%)
* Demand fluctuations

Observation:

* Revenue increases only in **constrained zones**
* No effect in surplus zones

---

## 📊 OR Concepts Implemented

* Linear Programming (LP)
* Mixed Integer Programming (MILP)
* Graphical Method (2-variable case)
* Duality (shadow pricing)
* Sensitivity Analysis
* Capacity Constraints
* Discrete Decision Modeling

---

## 🛠️ Tools & Technologies

| Tool                   | Purpose                  |
| ---------------------- | ------------------------ |
| Python (Pandas, NumPy) | Data processing          |
| PuLP                   | Optimization (prototype) |
| AMPL                   | Industrial optimization  |
| HiGHS Solver           | MILP solving             |
| Matplotlib             | Visualization            |

---

## ▶️ How to Run

### Python Pipeline

```bash
pip install -r requirements.txt
python main.py --data data/raw/yellow_tripdata_2025-01.parquet
```

---

### AMPL Optimization

```bash
cd ampl
ampl
include run.run;
```

---

## ⚠️ Assumptions

1. Demand derived from historical trip counts
2. Driver supply approximated using trip frequency
3. Demand decreases with price (discretized elasticity)
4. Fixed price per zone per time slot
5. No ride pooling or routing considered

---

## 🚀 Future Scope

* Real-time dynamic pricing
* Machine learning demand prediction
* Driver repositioning across zones
* Ride pooling optimization
* Reinforcement learning-based pricing

---

## 👥 Team Roles

| Member   | Contribution              |
| -------- | ------------------------- |
| Member 1 | Data preprocessing        |
| Member 2 | Demand modeling           |
| Member 3 | Optimization (core model) |
| Member 4 | Analysis & visualization  |

---

## 🏁 Final Conclusion

This project demonstrates how **optimization + data** can be used to design pricing strategies in ride-sharing systems.

> The optimal pricing policy is not driven purely by demand, but by the **interaction between demand and supply constraints**.

---

## 📌 Author Note

This project bridges **Operations Research theory** with **real-world platform economics**, providing a strong foundation for dynamic pricing systems used in modern ride-sharing platforms.

---

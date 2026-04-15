"""
=============================================================================
optimization.py
=============================================================================
Module  : Member 3 — Optimization Modelling (CORE)
Project : Revenue Maximization for Ride-Sharing using LP & Price Discretization
=============================================================================

Mathematical Model
------------------
Decision Variables:
    x_{i,j}  ∈ ℝ≥0   — rides served in zone i at price level j
    y_{i,j}  ∈ {0,1}  — 1 if price level j is selected for zone i

Objective (Maximize Revenue):
    Z = Σ_i Σ_j  p_j · x_{i,j}

Constraints:
    (C1) Demand     :  x_{i,j} ≤ D_{i,j} · y_{i,j}          ∀ i, j
    (C2) Drivers    :  Σ_j x_{i,j} ≤ S_i                     ∀ i
    (C3) One price  :  Σ_j y_{i,j} = 1                        ∀ i
    (C4) Non-neg    :  x_{i,j} ≥ 0,  y_{i,j} ∈ {0, 1}
=============================================================================
"""

import logging
import pandas as pd
import numpy as np

def safe_name(s):
    """Convert strings into solver-safe names"""
    return str(s).replace("-", "_").replace(" ", "_")

log = logging.getLogger(__name__)

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    log.warning("PuLP not installed. Run: pip install pulp")


PRICE_LEVELS  = [100, 150, 200]
PRICE_NAMES   = ["low", "medium", "high"]


# ── Model builder ─────────────────────────────────────────────────────────────

def build_and_solve(lp_input: pd.DataFrame,
                    solver_name: str = "CBC",
                    *,
                    unserved_penalty: float = 30.0,
                    compute_shadow_prices: bool = True) -> dict:
    """
    Build the Mixed-Integer Linear Program and solve it with PuLP.

    Parameters
    ----------
    lp_input     : processed DataFrame from data_preparation.py
    solver_name  : 'CBC' (default, free) | 'GLPK' | 'CPLEX'

    Returns
    -------
    dict with keys:
        status       : solver status string
        objective    : optimal revenue (₹)
        rides        : DataFrame — x_{i,j} values
        prices       : DataFrame — y_{i,j} values (selected price per zone)
        shadow_prices: dict — dual values of driver constraints (LP w/ fixed prices)
    """
    if not PULP_AVAILABLE:
        raise ImportError("Install PuLP first:  pip install pulp")

    log.info("Building MILP model …")

    # ── Index sets ────────────────────────────────────────────────────────────
    zones      = lp_input["zone"].unique().tolist()
    time_slots = lp_input["time_slot"].unique().tolist()
    groups     = list(lp_input[["zone", "time_slot"]].itertuples(index=False, name=None))
    J          = range(len(PRICE_LEVELS))          # j = 0,1,2

    # ── Parameters ────────────────────────────────────────────────────────────
    # D[i_t][j] : demand in zone-timeslot group (i,t) at price level j
    D = {}
    S = {}   # driver supply
    for _, row in lp_input.iterrows():
        key = (row["zone"], row["time_slot"])
        D[key] = {
            0: int(row["demand_low"]),
            1: int(row["demand_medium"]),
            2: int(row["demand_high"]),
        }
        S[key] = int(row["drivers"])

    # ── PuLP model ────────────────────────────────────────────────────────────
    model = pulp.LpProblem("RideSharing_Revenue_Maximization", pulp.LpMaximize)

    # Decision variables
    x = {(g, j): pulp.LpVariable(f"x_{g[0]}_{g[1]}_{j}", lowBound=0, cat="Continuous")
         for g in groups for j in J}
    y = {(g, j): pulp.LpVariable(f"y_{g[0]}_{g[1]}_{j}", cat="Binary")
         for g in groups for j in J}
    u = {g: pulp.LpVariable(f"u_{g[0]}_{g[1]}", lowBound=0, cat="Continuous")
         for g in groups}  # unmet demand (rides)

    # Objective
    model += (
        pulp.lpSum(PRICE_LEVELS[j] * x[(g, j)] for g in groups for j in J)
        - unserved_penalty * pulp.lpSum(u[g] for g in groups)
    ), "Revenue_minus_unserved_penalty"

    # C1 – Demand-linking constraint  x_{g,j} ≤ D_{g,j} · y_{g,j}
    for g in groups:
        for j in J:
            model += x[(g, j)] <= D[g][j] * y[(g, j)], f"C1_demand_{g[0]}_{g[1]}_{j}"

    # C1b – Accounting: served + unmet == realized demand at selected price
    for g in groups:
        model += (
            pulp.lpSum(x[(g, j)] for j in J) + u[g]
            == pulp.lpSum(D[g][j] * y[(g, j)] for j in J)
        ), f"C1b_balance_{g[0]}_{g[1]}"

    # C2 – Driver supply   Σ_j x_{g,j} ≤ S_g
    for g in groups:
        model += pulp.lpSum(x[(g, j)] for j in J) <= S[g], \
                 f"C2_drivers_{g[0]}_{g[1]}"

    # C3 – One price per zone-timeslot   Σ_j y_{g,j} = 1
    for g in groups:
        model += pulp.lpSum(y[(g, j)] for j in J) == 1, \
                 f"C3_oneprice_{g[0]}_{g[1]}"

    # ── Solve ─────────────────────────────────────────────────────────────────
    log.info("Solving with %s …", solver_name)
    if solver_name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=0)
    else:
        solver = pulp.getSolver(solver_name, msg=0)

    model.solve(solver)
    status = pulp.LpStatus[model.status]
    log.info("Solver status: %s", status)
    log.info("Optimal Revenue: ₹%.2f", pulp.value(model.objective))

    # ── Extract results ───────────────────────────────────────────────────────
    rides_records  = []
    prices_records = []

    for g in groups:
        for j in J:
            xval = pulp.value(x[(g, j)])
            yval = pulp.value(y[(g, j)])
            rides_records.append({
                "zone": g[0], "time_slot": g[1],
                "price_level": PRICE_NAMES[j],
                "price": PRICE_LEVELS[j],
                "rides_served": round(xval or 0),
                "revenue": round((xval or 0) * PRICE_LEVELS[j], 2),
                "price_selected": bool(round(yval or 0)),
            })

        # Best price for this group
        best_j = max(J, key=lambda j: pulp.value(y[(g, j)]) or 0)
        prices_records.append({
            "zone": g[0], "time_slot": g[1],
            "selected_price_level": PRICE_NAMES[best_j],
            "selected_price": PRICE_LEVELS[best_j],
        })

    rides_df  = pd.DataFrame(rides_records)
    prices_df = pd.DataFrame(prices_records)

    # Shadow prices (duality): CBC doesn't provide meaningful duals for a MILP.
    # We compute them from a continuous LP with y fixed to the optimal MILP choice.
    shadow = {}
    if compute_shadow_prices and status == "Optimal":

        chosen = {(g, j): int(round(pulp.value(y[(g, j)]) or 0)) for g in groups for j in J}    

        # ✅ LP is created HERE
        lp = pulp.LpProblem("RideSharing_FixedPrice_LP", pulp.LpMaximize)

        x_lp = {(g, j): pulp.LpVariable(f"xLP_{g[0]}_{g[1]}_{j}", lowBound=0) for g in groups for j in J} 
        u_lp = {g: pulp.LpVariable(f"uLP_{g[0]}_{g[1]}", lowBound=0) for g in groups}

        lp += (pulp.lpSum(PRICE_LEVELS[j] * x_lp[(g, j)] for g in groups for j in J)
               - unserved_penalty * pulp.lpSum(u_lp[g] for g in groups)), "LP_objective"

        # 👇👇👇 YOUR FIX GOES HERE ONLY
        for g in groups:
            for j in J:
                lp += x_lp[(g, j)] <= D[g][j] * chosen[(g, j)], f"LP_C1_{g[0]}_{g[1]}_{j}"

            lp += (
                pulp.lpSum(x_lp[(g, j)] for j in J) + u_lp[g]
                == pulp.lpSum(D[g][j] * chosen[(g, j)] for j in J)
            ), f"LP_balance_{g[0]}_{g[1]}"

            # ✅ THIS IS THE FIXED LINE
            cname = f"LP_C2_{safe_name(g[0])}_{safe_name(g[1])}"
            lp += pulp.lpSum(x_lp[(g, j)] for j in J) <= S[g], cname

        # solve AFTER constraints
        lp.solve(pulp.PULP_CBC_CMD(msg=0))

        # extract duals
        for g in groups:
            cname = f"LP_C2_{safe_name(g[0])}_{safe_name(g[1])}"
            c = lp.constraints.get(cname)
            if c and c.pi is not None:
                shadow[g] = c.pi

    return {
        "status"       : status,
        "objective"    : pulp.value(model.objective),
        "rides"        : rides_df,
        "prices"       : prices_df,
        "shadow_prices": shadow,
        "model"        : model,
    }


def print_solution_summary(result: dict) -> None:
    """Pretty-print the optimization results to console."""
    print("\n" + "=" * 60)
    print(f"  Solver Status  : {result['status']}")
    print(f"  Optimal Revenue: ₹{result['objective']:,.2f}")
    print("=" * 60)

    print("\n📌 Selected Prices per Zone:")
    print(result["prices"].to_string(index=False))

    print("\n📌 Rides Served (non-zero only):")
    active = result["rides"][result["rides"]["rides_served"] > 0]
    print(active[["zone", "time_slot", "price", "rides_served", "revenue"]]
          .to_string(index=False))

    if result["shadow_prices"]:
        print("\n📌 Shadow Prices (₹ gained per extra driver):")
        for (zone, slot), pi in sorted(result["shadow_prices"].items()):
            if pi and abs(pi) > 1e-6:
                print(f"  Zone {zone:>4}  {slot:<10}  π = ₹{pi:.2f}")
    print()
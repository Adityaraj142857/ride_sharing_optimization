# =============================================================================
# model.mod  —  Revenue Maximization for Ride-Sharing
# =============================================================================
# Mixed-Integer Linear Program (MILP)
#
# Decision Variables:
#   x[i,j]  : rides served in zone i at price level j  (continuous, >= 0)
#   y[i,j]  : 1 if price level j is chosen for zone i  (binary)
#
# Objective:
#   max Z = sum_{i,j}  price[j] * x[i,j]
#
# Constraints:
#   C1  demand linking  : x[i,j] <= D[i,j] * y[i,j]   for all i, j
#   C2  driver supply   : sum_j x[i,j] <= S[i]          for all i
#   C3  one price       : sum_j y[i,j] = 1               for all i
# =============================================================================

# ── Sets ──────────────────────────────────────────────────────────────────────
set ZONES;                 # Set of zone-timeslot groups  e.g. "142_Peak"
set PRICE_LEVELS;          # {low, medium, high}

# ── Parameters ────────────────────────────────────────────────────────────────
param price {PRICE_LEVELS} >= 0;          # Price (₹) at each level
param D     {ZONES, PRICE_LEVELS} >= 0;  # Demand: zone i at price level j
param S     {ZONES} >= 0;           
param penalty >= 0;
     # Driver supply in zone i

# ── Decision Variables ────────────────────────────────────────────────────────
var x {ZONES, PRICE_LEVELS} >= 0;        # Rides served
var y {ZONES, PRICE_LEVELS} binary;      # Price selection indicator
var u {ZONES} >= 0;

# ── Objective ─────────────────────────────────────────────────────────────────
maximize TotalRevenue:
    sum {i in ZONES, j in PRICE_LEVELS} price[j] * x[i,j]
    - penalty * sum {i in ZONES} u[i];

# ── Constraints ───────────────────────────────────────────────────────────────

# C1: Rides can only be served at a selected price level, and cannot exceed demand
subject to DemandLinking {i in ZONES, j in PRICE_LEVELS}:
    x[i,j] <= D[i,j] * y[i,j];

# C2: Total rides served in a zone cannot exceed available drivers
subject to DriverSupply {i in ZONES}:
    sum {j in PRICE_LEVELS} x[i,j] <= S[i];

# C3: Exactly one price level must be selected per zone
subject to OnePricePerZone {i in ZONES}:
    sum {j in PRICE_LEVELS} y[i,j] = 1;
    
# C4: Demand balance
subject to DemandBalance {i in ZONES}:
    sum {j in PRICE_LEVELS} x[i,j] + u[i]
    = sum {j in PRICE_LEVELS} D[i,j] * y[i,j];
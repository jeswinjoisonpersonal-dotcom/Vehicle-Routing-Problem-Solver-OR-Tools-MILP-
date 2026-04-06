"""
Evaluation Harness for CVRP Solver
====================================
Demonstrates: offline simulation, ablations, scenario analysis, KPI tracking.
"""

import time
import numpy as np
import pandas as pd
from main import CVRPSolver, generate_random_instance, VRPInstance

# ─── KPI Definitions ────────────────────────────────────────────────────────

KPI_THRESHOLDS = {
    "total_distance_km":      {"max": 2000, "unit": "km"},
    "avg_utilization_pct":    {"min": 50.0,  "unit": "%"},
    "solve_time_s":           {"max": 30.0,  "unit": "s"},
    "feasibility_rate":       {"min": 1.0,   "unit": "rate"},
}

def check_kpis(results: list[dict]) -> pd.DataFrame:
    records = []
    for r in results:
        passed = {
            "distance_ok":     r["total_distance_km"] <= KPI_THRESHOLDS["total_distance_km"]["max"],
            "utilization_ok":  r["avg_utilization_pct"] >= KPI_THRESHOLDS["avg_utilization_pct"]["min"],
            "time_ok":         r["solve_time_s"] <= KPI_THRESHOLDS["solve_time_s"]["max"],
            "feasible":        r["status"] in ("OPTIMAL", "FEASIBLE"),
        }
        records.append({**r, **passed, "all_passed": all(passed.values())})
    return pd.DataFrame(records)


# ─── Scenario Analysis ───────────────────────────────────────────────────────

def scenario_analysis(seeds: list[int] = None) -> pd.DataFrame:
    """Run solver across multiple random instances (Monte Carlo scenarios)."""
    if seeds is None:
        seeds = list(range(10))

    solver = CVRPSolver(time_limit_s=15, verbose=False)
    results = []

    scenarios = [
        {"num_customers": 8,  "num_vehicles": 2, "vehicle_capacity": 40},
        {"num_customers": 10, "num_vehicles": 3, "vehicle_capacity": 50},
        {"num_customers": 12, "num_vehicles": 4, "vehicle_capacity": 45},
    ]

    for scenario in scenarios:
        for seed in seeds:
            instance = generate_random_instance(**scenario, seed=seed)
            sol = solver.solve(instance)
            summary = sol.summary()
            results.append({
                "scenario":             f"C{scenario['num_customers']}_V{scenario['num_vehicles']}",
                "num_customers":        scenario["num_customers"],
                "num_vehicles":         scenario["num_vehicles"],
                "vehicle_capacity":     scenario["vehicle_capacity"],
                "seed":                 seed,
                "status":               summary["status"],
                "total_distance_km":    summary["total_distance_km"],
                "avg_utilization_pct":  summary["avg_utilization_pct"],
                "solve_time_s":         summary["solve_time_s"],
            })
            print(f"  [{scenario['num_customers']}C/{scenario['num_vehicles']}V seed={seed}]  "
                  f"dist={summary['total_distance_km']}km  "
                  f"util={summary['avg_utilization_pct']}%  "
                  f"status={summary['status']}")

    return pd.DataFrame(results)


# ─── Ablation: Capacity vs Distance ─────────────────────────────────────────

def ablation_capacity(seed: int = 42) -> pd.DataFrame:
    """Ablate vehicle capacity and measure impact on route distance."""
    solver = CVRPSolver(time_limit_s=10, verbose=False)
    capacities = [20, 30, 40, 50, 60, 80]
    rows = []
    print("\nAblation: Vehicle Capacity vs Total Distance")
    print("-" * 50)
    for cap in capacities:
        instance = generate_random_instance(num_customers=10, num_vehicles=3,
                                            vehicle_capacity=cap, seed=seed)
        sol = solver.solve(instance)
        s = sol.summary()
        rows.append({"capacity": cap, "distance_km": s["total_distance_km"],
                     "utilization_pct": s["avg_utilization_pct"], "status": s["status"]})
        print(f"  Cap={cap:3d}  dist={s['total_distance_km']:8.2f} km  "
              f"util={s['avg_utilization_pct']:5.1f}%  {s['status']}")
    return pd.DataFrame(rows)


# ─── Ablation: Fleet Size vs Distance ───────────────────────────────────────

def ablation_fleet_size(seed: int = 42) -> pd.DataFrame:
    """Ablate number of vehicles and measure distance/utilization trade-off."""
    solver = CVRPSolver(time_limit_s=10, verbose=False)
    fleet_sizes = [2, 3, 4, 5]
    rows = []
    print("\nAblation: Fleet Size vs Total Distance")
    print("-" * 50)
    for v in fleet_sizes:
        instance = generate_random_instance(num_customers=10, num_vehicles=v,
                                            vehicle_capacity=50, seed=seed)
        sol = solver.solve(instance)
        s = sol.summary()
        rows.append({"num_vehicles": v, "distance_km": s["total_distance_km"],
                     "utilization_pct": s["avg_utilization_pct"], "status": s["status"]})
        print(f"  Vehicles={v}  dist={s['total_distance_km']:8.2f} km  "
              f"util={s['avg_utilization_pct']:5.1f}%  {s['status']}")
    return pd.DataFrame(rows)


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CVRP Evaluation Harness")
    print("=" * 60)

    # 1. Scenario analysis
    print("\n[1] Scenario Analysis (3 configs × 5 seeds)")
    df_scenarios = scenario_analysis(seeds=list(range(5)))

    print("\nScenario Summary:")
    summary = df_scenarios.groupby("scenario").agg(
        mean_dist=("total_distance_km", "mean"),
        std_dist=("total_distance_km", "std"),
        mean_util=("avg_utilization_pct", "mean"),
        feasibility_rate=("status", lambda x: (x.isin(["OPTIMAL","FEASIBLE"])).mean()),
    ).round(2)
    print(summary.to_string())

    # 2. Ablations
    print("\n[2] Ablations")
    df_cap   = ablation_capacity()
    df_fleet = ablation_fleet_size()

    # 3. KPI check
    print("\n[3] KPI Acceptance Check")
    kpi_rows = df_scenarios.to_dict("records")
    kpi_df = check_kpis(kpi_rows)
    pass_rate = kpi_df["all_passed"].mean()
    print(f"  KPI pass rate: {pass_rate*100:.1f}%")
    print(f"  Threshold definitions: {KPI_THRESHOLDS}")

    # 4. Save
    df_scenarios.to_csv("scenario_results.csv", index=False)
    df_cap.to_csv("ablation_capacity.csv", index=False)
    df_fleet.to_csv("ablation_fleet.csv", index=False)
    print("\nResults saved: scenario_results.csv, ablation_capacity.csv, ablation_fleet.csv")

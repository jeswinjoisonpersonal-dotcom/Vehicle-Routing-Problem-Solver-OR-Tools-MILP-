"""
Vehicle Routing Problem Solver using OR-Tools CP-SAT
=====================================================
Demonstrates: MILP/CP-SAT, constraint programming, planning/routing/allocation.

Problem: A depot must dispatch K vehicles to serve N customers.
Each vehicle has a capacity. Minimize total travel distance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from ortools.sat.python import cp_model
import time
import json

# ─── Problem Definition ──────────────────────────────────────────────────────

@dataclass
class Location:
    id: int
    x: float
    y: float
    demand: int = 0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"Customer_{self.id}" if self.id > 0 else "Depot"

@dataclass
class VRPInstance:
    """Encapsulates a full CVRP instance."""
    locations: List[Location]
    num_vehicles: int
    vehicle_capacity: int
    depot_id: int = 0

    @property
    def customers(self) -> List[Location]:
        return [l for l in self.locations if l.id != self.depot_id]

    @property
    def depot(self) -> Location:
        return self.locations[self.depot_id]

    def distance(self, i: int, j: int) -> int:
        """Euclidean distance scaled to integer (CP-SAT requires integers)."""
        a, b = self.locations[i], self.locations[j]
        return int(np.hypot(a.x - b.x, a.y - b.y) * 100)

    def distance_matrix(self) -> np.ndarray:
        n = len(self.locations)
        D = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                D[i][j] = self.distance(i, j)
        return D

@dataclass
class VRPSolution:
    routes: List[List[int]]          # list of routes, each route = list of location ids
    total_distance: float
    vehicle_loads: List[int]
    solve_time_s: float
    status: str
    instance: VRPInstance

    def summary(self) -> Dict:
        return {
            "status": self.status,
            "total_distance_km": round(self.total_distance / 100, 2),
            "num_routes": len(self.routes),
            "vehicle_utilization_pct": [
                round(100 * load / self.instance.vehicle_capacity, 1)
                for load in self.vehicle_loads
            ],
            "avg_utilization_pct": round(
                100 * np.mean(self.vehicle_loads) / self.instance.vehicle_capacity, 1
            ),
            "solve_time_s": round(self.solve_time_s, 3),
            "routes": self.routes,
        }


# ─── Solver ──────────────────────────────────────────────────────────────────

class CVRPSolver:
    """
    Capacitated VRP solver using OR-Tools CP-SAT.

    Formulation:
      - Binary arc variables x[v][i][j]: vehicle v travels arc (i→j)
      - Integer load variables load[v]: total demand served by vehicle v
      - Subtour elimination via Miller-Tucker-Zemlin (MTZ) auxiliary variables
      - Objective: minimize total arc cost
    """

    def __init__(self, time_limit_s: int = 30, verbose: bool = True):
        self.time_limit_s = time_limit_s
        self.verbose = verbose

    def solve(self, instance: VRPInstance) -> VRPSolution:
        model = cp_model.CpModel()
        n = len(instance.locations)
        V = instance.num_vehicles
        C = instance.vehicle_capacity
        D = instance.distance_matrix()

        # ── Decision Variables ───────────────────────────────────────────────
        # x[v][i][j] = 1 if vehicle v travels arc i→j
        x = [[[model.NewBoolVar(f"x_{v}_{i}_{j}")
               for j in range(n)] for i in range(n)] for v in range(V)]

        # u[v][i] = position of node i in route for vehicle v (MTZ subtour elimination)
        u = [[model.NewIntVar(0, n, f"u_{v}_{i}") for i in range(n)] for v in range(V)]

        # load[v] = total demand carried by vehicle v
        load = [model.NewIntVar(0, C, f"load_{v}") for v in range(V)]

        # ── Constraints ──────────────────────────────────────────────────────

        # 1. Each customer visited exactly once (across all vehicles)
        for j in instance.customers:
            model.Add(sum(x[v][i][j.id] for v in range(V) for i in range(n) if i != j.id) == 1)

        # 2. Flow conservation: if vehicle enters a node, it must leave
        for v in range(V):
            for k in range(n):
                in_flow  = sum(x[v][i][k] for i in range(n) if i != k)
                out_flow = sum(x[v][k][j] for j in range(n) if j != k)
                model.Add(in_flow == out_flow)

        # 3. Each vehicle leaves depot at most once
        for v in range(V):
            model.Add(sum(x[v][0][j] for j in range(1, n)) <= 1)

        # 4. Capacity constraint
        for v in range(V):
            model.Add(load[v] == sum(
                instance.locations[j].demand * x[v][i][j]
                for i in range(n) for j in range(1, n) if i != j
            ))
            model.Add(load[v] <= C)

        # 5. No self-loops
        for v in range(V):
            for i in range(n):
                model.Add(x[v][i][i] == 0)

        # 6. MTZ subtour elimination
        for v in range(V):
            model.Add(u[v][0] == 0)
            for i in range(1, n):
                model.Add(u[v][i] >= 1)
                model.Add(u[v][i] <= n - 1)
            for i in range(1, n):
                for j in range(1, n):
                    if i != j:
                        model.Add(u[v][j] >= u[v][i] + 1 - n * (1 - x[v][i][j]))

        # ── Objective ────────────────────────────────────────────────────────
        total_dist = sum(
            D[i][j] * x[v][i][j]
            for v in range(V) for i in range(n) for j in range(n) if i != j
        )
        model.Minimize(total_dist)

        # ── Solve ─────────────────────────────────────────────────────────────
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        solver.parameters.log_search_progress = self.verbose

        t0 = time.time()
        status = solver.Solve(model)
        elapsed = time.time() - t0

        status_map = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.UNKNOWN: "UNKNOWN",
        }
        status_str = status_map.get(status, "UNKNOWN")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            routes = self._extract_routes(solver, x, n, V, instance.depot_id)
            loads  = [solver.Value(load[v]) for v in range(V)]
            dist   = solver.ObjectiveValue()
        else:
            routes, loads, dist = [], [], float("inf")

        return VRPSolution(
            routes=routes,
            total_distance=dist,
            vehicle_loads=loads,
            solve_time_s=elapsed,
            status=status_str,
            instance=instance,
        )

    def _extract_routes(self, solver, x, n, V, depot):
        routes = []
        for v in range(V):
            # trace route starting from depot
            route = []
            current = depot
            visited = set()
            while True:
                next_node = None
                for j in range(n):
                    if j != current and j not in visited:
                        if solver.Value(x[v][current][j]) == 1:
                            next_node = j
                            break
                if next_node is None or next_node == depot:
                    break
                route.append(next_node)
                visited.add(next_node)
                current = next_node
            if route:
                routes.append([depot] + route + [depot])
        return routes


# ─── Instance Generator ──────────────────────────────────────────────────────

def generate_random_instance(
    num_customers: int = 10,
    num_vehicles: int = 3,
    vehicle_capacity: int = 40,
    seed: int = 42,
) -> VRPInstance:
    rng = np.random.default_rng(seed)
    locations = [Location(id=0, x=50.0, y=50.0, demand=0, name="Depot")]
    for i in range(1, num_customers + 1):
        locations.append(Location(
            id=i,
            x=float(rng.integers(0, 100)),
            y=float(rng.integers(0, 100)),
            demand=int(rng.integers(5, 20)),
        ))
    return VRPInstance(locations=locations, num_vehicles=num_vehicles,
                       vehicle_capacity=vehicle_capacity)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CVRP Solver — OR-Tools CP-SAT")
    print("=" * 60)

    instance = generate_random_instance(num_customers=10, num_vehicles=3,
                                        vehicle_capacity=50, seed=7)

    print(f"\nInstance: {len(instance.customers)} customers, "
          f"{instance.num_vehicles} vehicles, capacity={instance.vehicle_capacity}")
    print(f"Total demand: {sum(c.demand for c in instance.customers)}")
    print(f"Locations:")
    for loc in instance.locations:
        print(f"  [{loc.id}] {loc.name:20s}  xy=({loc.x:.0f},{loc.y:.0f})  demand={loc.demand}")

    solver = CVRPSolver(time_limit_s=30, verbose=False)
    solution = solver.solve(instance)

    summary = solution.summary()
    print(f"\n{'─'*40}")
    print(f"Status         : {summary['status']}")
    print(f"Total Distance : {summary['total_distance_km']} km")
    print(f"Solve Time     : {summary['solve_time_s']} s")
    print(f"Avg Utilization: {summary['avg_utilization_pct']}%")
    print(f"\nRoutes:")
    for i, (route, util) in enumerate(zip(summary["routes"], summary["vehicle_utilization_pct"])):
        node_names = [instance.locations[n].name for n in route]
        print(f"  Vehicle {i+1} ({util}% capacity): {' → '.join(node_names)}")

    with open("solution.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSolution saved to solution.json")

"""
Route Visualization for CVRP Solution
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from main import generate_random_instance, CVRPSolver


def plot_solution(instance, solution, title="CVRP Solution"):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor("#0f0f1a")
    fig.patch.set_facecolor("#0f0f1a")

    locs = instance.locations
    colors = cm.tab10(np.linspace(0, 1, instance.num_vehicles))

    # Plot routes
    for v_idx, (route, color) in enumerate(zip(solution.routes, colors)):
        if not route:
            continue
        xs = [locs[i].x for i in route]
        ys = [locs[i].y for i in route]
        ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=4,
                label=f"Vehicle {v_idx+1} ({solution.vehicle_loads[v_idx]} units)", alpha=0.9)

    # Plot customers
    for loc in instance.customers:
        ax.scatter(loc.x, loc.y, s=80, color="white", zorder=5, edgecolors="#aaa", linewidth=0.5)
        ax.annotate(f"{loc.id}\n({loc.demand})", (loc.x, loc.y),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color="#cccccc")

    # Plot depot
    depot = instance.depot
    ax.scatter(depot.x, depot.y, s=200, marker="*", color="#ffdd57", zorder=6, label="Depot")
    ax.annotate("DEPOT", (depot.x, depot.y), textcoords="offset points", xytext=(8, 8),
                fontsize=9, color="#ffdd57", fontweight="bold")

    ax.legend(loc="lower right", facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=9)
    ax.set_title(f"{title}\nTotal Distance: {solution.total_distance/100:.1f} km | Status: {solution.status}",
                 color="white", fontsize=12)
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_xlabel("X Coordinate", color="#888")
    ax.set_ylabel("Y Coordinate", color="#888")

    plt.tight_layout()
    plt.savefig("routes.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("Route visualization saved to routes.png")
    plt.show()


if __name__ == "__main__":
    instance = generate_random_instance(num_customers=10, num_vehicles=3,
                                        vehicle_capacity=50, seed=7)
    solver = CVRPSolver(time_limit_s=30, verbose=False)
    solution = solver.solve(instance)
    plot_solution(instance, solution)

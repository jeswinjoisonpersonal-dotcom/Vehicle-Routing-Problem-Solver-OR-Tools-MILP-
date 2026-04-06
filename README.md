# Project 1: Vehicle Routing Problem Solver (OR-Tools + MILP)

## What It Does
Solves a capacitated Vehicle Routing Problem (CVRP) using Google OR-Tools CP-SAT solver.
Covers: planning, routing, allocation, resource utilization.

## Skills Demonstrated
- MILP / CP-SAT constraint programming
- OR-Tools solver stack
- Evaluation harness with KPIs
- Clean problem formulation from messy real-world inputs

## Setup
```bash
pip install -r requirements.txt
python main.py
python evaluate.py
```

## Files
- `main.py`        - Core solver + problem definition
- `evaluate.py`    - Evaluation harness (KPIs, ablations, scenario analysis)
- `visualize.py`   - Route visualization
- `requirements.txt`

"""Compares search algorithms on a weighted grid and on 8-puzzle instances; writes results/benchmarks.json.

    python examples/run_benchmarks.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from searchagents import EightPuzzle, GridProblem, a_star, breadth_first, greedy, random_grid, uniform_cost

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def row(r) -> dict:
    return {"found": r.found, "cost": round(r.cost, 3), "expanded": r.expanded, "generated": r.generated}


def main() -> None:
    res: dict = {"DATA": "synthetic grids (seeded) and 8-puzzle instances"}
    totals: dict[str, dict[str, float]] = {}
    for seed in range(20):
        rows = random_grid(40, 40, 0.25, seed, terrain=True)
        p = GridProblem(rows, (0, 0), (39, 39))
        runs = {"uniform-cost": uniform_cost(p), "A* (Manhattan)": a_star(p, p.manhattan), "greedy (Manhattan)": greedy(p, p.manhattan)}
        if not runs["uniform-cost"].found:
            continue
        for name, r in runs.items():
            t = totals.setdefault(name, {"instances": 0, "cost": 0.0, "expanded": 0})
            t["instances"] += 1; t["cost"] += r.cost; t["expanded"] += r.expanded
    res["weighted_40x40_grids_25pct_walls_totals_over_solvable_instances"] = {
        n: {"instances": int(t["instances"]), "mean_cost": round(t["cost"] / t["instances"], 2),
            "mean_expanded": round(t["expanded"] / t["instances"], 1)} for n, t in totals.items()}

    eight = {}
    for label, start in (("optimal 14 moves", (8, 1, 3, 4, 0, 2, 7, 6, 5)), ("31 moves (hardest)", (8, 6, 7, 2, 5, 4, 3, 0, 1))):
        puz = EightPuzzle(start)
        runs = {"BFS": breadth_first(puz), "A* (misplaced tiles)": a_star(puz, puz.misplaced), "A* (Manhattan)": a_star(puz, puz.manhattan)}
        eight[label] = {n: row(r) for n, r in runs.items()}
    res["eight_puzzle"] = eight
    (OUT / "benchmarks.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

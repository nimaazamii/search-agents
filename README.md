# search-agents

Classical search algorithms with a common problem interface: **BFS, depth-limited DFS, iterative deepening, uniform-cost, greedy best-first and A\***,
applied to weighted grid path finding (4- or 8-connected, terrain costs) and the **8-puzzle** (with solvability test and Manhattan / misplaced-tile heuristics).

## Installation and quick start
```bash
git clone https://github.com/nimaazamii/search-agents.git
cd search-agents
pip install -e ".[dev]"
pytest -q
```

```python
from searchagents import GridProblem, a_star, breadth_first
grid = ["....#...",
        ".##.#.#.",
        ".#..#.#.",
        ".#.##.#.",
        ".#....#.",
        ".####.#.",
        "........"]
p = GridProblem(grid, (0, 0), (0, 7))
r = a_star(p, p.manhattan)
print(r.cost, r.expanded, "".join(r.actions))
```
Output: `17.0 22 RRRDDLDDRRRUUUURR`  (cost, nodes expanded, move sequence)

## Verification (`pytest -q`, 12 tests)
* On 25 random 12x12 mazes BFS, uniform-cost and A* agree with an independent BFS (reachability and path length); on 20 weighted grids A* and uniform-cost equal an independent Dijkstra, and A* expands fewer nodes.
* Iterative deepening is optimal on small grids (exponential, so only small instances are tested); greedy search is shown to be non-optimal on a trap grid.
* With diagonal moves the Euclidean and Chebyshev heuristics stay optimal, while Manhattan is inadmissible and is checked only to be no better than optimal.
* 8-puzzle: parity-based solvability (an unsolvable instance is rejected by BFS as well), all optimal algorithms agree, Manhattan dominates misplaced tiles, plans are replayed to reach the goal,
and the known hardest instance `8 6 7 / 2 5 4 / 3 0 1` is solved with the optimal **31 moves**.

## Benchmarks (`python examples/run_benchmarks.py`; synthetic grids, seeded)
40x40 grids with 25% walls and terrain costs 1-5; only the 9 of 20 seeds that are solvable are counted:

| algorithm | instances | mean path cost | mean nodes expanded |
|---|---|---|---|
| uniform-cost | 9 | 101.33 | 1183.3 |
| A* (Manhattan) | 9 | 101.33 | 821.9 |
| greedy (Manhattan) | 9 | 153.33 | 120.9 |

Greedy is much faster but returns paths about 51% more expensive than the optimum on these grids.

8-puzzle (single instances, so these are examples, not averages):

| instance | algorithm | optimal moves | expanded | generated |
|---|---|---|---|---|
| optimal 14 moves | BFS | 14 | 3,633 | 5,961 |
| optimal 14 moves | A* (misplaced tiles) | 14 | 301 | 499 |
| optimal 14 moves | A* (Manhattan) | 14 | 76 | 128 |
| 31 moves (hardest) | BFS | 31 | 181,347 | 181,439 |
| 31 moves (hardest) | A* (misplaced tiles) | 31 | 143,839 | 162,463 |
| 31 moves (hardest) | A* (Manhattan) | 31 | 20,290 | 29,757 |

## Limitations and future work
Graph search with full re-opening (no IDA*/memory-bounded variants), iterative deepening uses only path-based cycle checking (exponential on open grids), no bidirectional search, 3x3 puzzle only.

## References
* Hart, Nilsson and Raphael (1968) A formal basis for the heuristic determination of minimum cost paths. IEEE Trans. SSC 4(2); Russell and Norvig, *Artificial Intelligence: A Modern Approach*.

## License
MIT, see `LICENSE`.

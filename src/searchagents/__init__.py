"""searchagents: BFS, DFS, iterative deepening, uniform-cost, greedy and A* on grids and the 8-puzzle."""
from .core import (SearchResult, a_star, best_first, breadth_first, depth_limited, greedy, iterative_deepening,
                   uniform_cost)
from .problems import EightPuzzle, GridProblem, random_grid

__version__ = "0.1.0"
__all__ = ["EightPuzzle", "GridProblem", "SearchResult", "a_star", "best_first", "breadth_first", "depth_limited", "greedy",
           "iterative_deepening", "random_grid", "uniform_cost"]

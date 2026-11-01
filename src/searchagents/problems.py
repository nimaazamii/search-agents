"""Concrete problems: weighted grid path finding and the sliding-tile puzzle."""
from __future__ import annotations

import math
import random
from collections.abc import Hashable, Iterable


class GridProblem:
    """4- or 8-connected grid; cells are '#' (wall), '.' (cost 1) or a digit 1-9 (terrain cost)."""

    def __init__(self, rows: list[str], start: tuple[int, int], goal: tuple[int, int], diagonal: bool = False) -> None:
        self.rows, self.start, self.goal, self.diagonal = rows, start, goal, diagonal
        for r, c in (start, goal):
            if self.cell_cost(r, c) is None:
                raise ValueError("start and goal must be free cells")

    def cell_cost(self, r: int, c: int) -> float | None:
        if not (0 <= r < len(self.rows) and 0 <= c < len(self.rows[0])) or self.rows[r][c] == "#":
            return None
        return 1.0 if self.rows[r][c] == "." else float(self.rows[r][c])

    def initial(self) -> tuple[int, int]:
        return self.start

    def is_goal(self, s: Hashable) -> bool:
        return s == self.goal

    def successors(self, s: tuple[int, int]) -> Iterable[tuple[str, tuple[int, int], float]]:
        steps = [("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1)]
        if self.diagonal:
            steps += [("UL", -1, -1), ("UR", -1, 1), ("DL", 1, -1), ("DR", 1, 1)]
        for name, dr, dc in steps:
            r, c = s[0] + dr, s[1] + dc
            cost = self.cell_cost(r, c)
            if cost is not None:
                yield name, (r, c), cost * (math.sqrt(2) if len(name) == 2 else 1.0)

    # heuristics (admissible when every cell costs at least 1)
    def manhattan(self, s: tuple[int, int]) -> float:
        return abs(s[0] - self.goal[0]) + abs(s[1] - self.goal[1])

    def euclidean(self, s: tuple[int, int]) -> float:
        return math.hypot(s[0] - self.goal[0], s[1] - self.goal[1])

    def chebyshev(self, s: tuple[int, int]) -> float:
        return float(max(abs(s[0] - self.goal[0]), abs(s[1] - self.goal[1])))


def random_grid(rows: int, cols: int, wall_prob: float, seed: int, terrain: bool = False) -> list[str]:
    rng = random.Random(seed)
    grid = []
    for r in range(rows):
        line = ""
        for c in range(cols):
            if (r, c) in ((0, 0), (rows - 1, cols - 1)):
                line += "."
            elif rng.random() < wall_prob:
                line += "#"
            else:
                line += str(rng.randint(1, 5)) if terrain and rng.random() < 0.4 else "."
        grid.append(line)
    return grid


class EightPuzzle:
    """3x3 sliding puzzle; a state is a tuple of 9 numbers with 0 as the blank."""
    GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def __init__(self, start: tuple[int, ...], goal: tuple[int, ...] = GOAL) -> None:
        if sorted(start) != list(range(9)) or sorted(goal) != list(range(9)):
            raise ValueError("states must be permutations of 0..8")
        self.start, self.goal = tuple(start), tuple(goal)

    def initial(self):
        return self.start

    def is_goal(self, s) -> bool:
        return s == self.goal

    def successors(self, s):
        z = s.index(0)
        r, c = divmod(z, 3)
        for name, dr, dc in (("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                t = list(s)
                t[z], t[nr * 3 + nc] = t[nr * 3 + nc], 0
                yield name, tuple(t), 1.0

    @staticmethod
    def inversions(state: tuple[int, ...]) -> int:
        tiles = [t for t in state if t]
        return sum(1 for i in range(len(tiles)) for j in range(i + 1, len(tiles)) if tiles[i] > tiles[j])

    def solvable(self) -> bool:
        """Reachability test: both states must have inversion counts of equal parity (odd board width)."""
        return self.inversions(self.start) % 2 == self.inversions(self.goal) % 2

    def misplaced(self, s) -> float:
        return float(sum(1 for a, b in zip(s, self.goal) if a and a != b))

    def manhattan(self, s) -> float:
        pos = {t: divmod(i, 3) for i, t in enumerate(self.goal)}
        return float(sum(abs(i // 3 - pos[t][0]) + abs(i % 3 - pos[t][1]) for i, t in enumerate(s) if t))

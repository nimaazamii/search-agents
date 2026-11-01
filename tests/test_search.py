import heapq
import random
import unittest
from collections import deque

from searchagents import (EightPuzzle, GridProblem, a_star, breadth_first, depth_limited, greedy, iterative_deepening,
                          random_grid, uniform_cost)

OPEN = ["....", "....", "...."]


def dijkstra(problem):
    start = problem.initial()
    dist, heap = {start: 0.0}, [(0.0, start)]
    while heap:
        d, s = heapq.heappop(heap)
        if d > dist[s]:
            continue
        if problem.is_goal(s):
            return d
        for _, n, c in problem.successors(s):
            if d + c < dist.get(n, 1e18):
                dist[n] = d + c; heapq.heappush(heap, (d + c, n))
    return None


def bfs_len(rows, start, goal):
    q, seen = deque([(start, 0)]), {start}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < len(rows) and 0 <= n[1] < len(rows[0]) and rows[n[0]][n[1]] != "#" and n not in seen:
                seen.add(n); q.append((n, d + 1))
    return None


class TestGrid(unittest.TestCase):
    def test_open_grid_optimal_algorithms_agree(self):
        p = GridProblem(OPEN, (0, 0), (2, 3))
        for r in (breadth_first(p), uniform_cost(p), a_star(p, p.manhattan)):
            self.assertTrue(r.found); self.assertEqual(r.cost, 5.0); self.assertEqual(len(r.actions), 5)
        self.assertEqual(a_star(p, p.manhattan).path[0], (0, 0)); self.assertEqual(a_star(p, p.manhattan).path[-1], (2, 3))

    def test_random_mazes_against_independent_references(self):
        for seed in range(25):
            rows = random_grid(12, 12, 0.3, seed)
            p = GridProblem(rows, (0, 0), (11, 11))
            expected = bfs_len(rows, (0, 0), (11, 11))
            for algo in (breadth_first, uniform_cost, lambda q: a_star(q, q.manhattan)):
                r = algo(p)
                self.assertEqual(r.found, expected is not None, seed)
                if expected is not None:
                    self.assertEqual(r.cost, expected)

    def test_iterative_deepening_is_optimal_on_small_grids(self):
        checked = 0
        for seed in range(60):
            rows = random_grid(5, 5, 0.2, seed)
            expected = bfs_len(rows, (0, 0), (4, 4))
            if expected is None or expected > 10:               # IDDFS is exponential: keep the instances small
                continue
            r = iterative_deepening(GridProblem(rows, (0, 0), (4, 4)), max_depth=expected)
            self.assertTrue(r.found)
            self.assertEqual(r.cost, expected)
            checked += 1
        self.assertGreater(checked, 10)
        p = GridProblem(["..#.", "..#."], (0, 0), (0, 3))
        self.assertFalse(iterative_deepening(p, max_depth=4).found)

    def test_weighted_terrain_a_star_equals_dijkstra_and_expands_less(self):
        total_ucs = total_astar = 0
        for seed in range(20):
            rows = random_grid(15, 15, 0.2, seed, terrain=True)
            p = GridProblem(rows, (0, 0), (14, 14))
            ref = dijkstra(p)
            u, a = uniform_cost(p), a_star(p, p.manhattan)
            self.assertEqual(u.found, ref is not None)
            if ref is not None:
                self.assertAlmostEqual(u.cost, ref); self.assertAlmostEqual(a.cost, ref)
                total_ucs += u.expanded; total_astar += a.expanded
        self.assertLess(total_astar, total_ucs)                       # an informed heuristic prunes the search

    def test_greedy_is_fast_but_not_always_optimal(self):
        rows = ["..9.", "..9.", "...."]
        p = GridProblem(rows, (0, 1), (0, 3))
        opt, g = a_star(p, p.manhattan), greedy(p, p.manhattan)
        self.assertLessEqual(opt.cost, g.cost)
        # a trap where the heuristic leads into a dead end
        trap = ["..#..", ".##..", "..#..", "....."]
        q = GridProblem(trap, (0, 0), (0, 4))
        self.assertTrue(greedy(q, q.manhattan).found)
        self.assertGreater(greedy(q, q.manhattan).cost, a_star(q, q.manhattan).cost - 1e-9)

    def test_diagonal_moves_and_heuristics_are_admissible(self):
        rows = random_grid(10, 10, 0.15, 3)
        p = GridProblem(rows, (0, 0), (9, 9), diagonal=True)
        ref = dijkstra(p)
        for h in (p.euclidean, p.chebyshev):
            r = a_star(p, h)
            self.assertAlmostEqual(r.cost, ref)
        # Manhattan overestimates with diagonal moves, so it can lose optimality; Euclidean never does
        self.assertGreaterEqual(a_star(p, p.manhattan).cost, ref - 1e-9)

    def test_unreachable_and_validation(self):
        p = GridProblem(["..#.", "..#.", "..#."], (0, 0), (0, 3))
        for algo in (breadth_first, uniform_cost, lambda q: a_star(q, q.manhattan), lambda q: iterative_deepening(q, 5)):
            self.assertFalse(algo(p).found)
        with self.assertRaises(ValueError):
            GridProblem(["#."], (0, 0), (0, 1))
        self.assertTrue(breadth_first(GridProblem(["."], (0, 0), (0, 0))).found)

    def test_depth_limited_respects_limit(self):
        p = GridProblem(OPEN, (0, 0), (2, 3))
        self.assertFalse(depth_limited(p, 4).found)
        self.assertTrue(depth_limited(p, 5).found)


class TestEightPuzzle(unittest.TestCase):
    def test_solvability_parity(self):
        self.assertTrue(EightPuzzle((1, 2, 3, 4, 5, 6, 7, 8, 0)).solvable())
        self.assertFalse(EightPuzzle((1, 2, 3, 4, 5, 6, 8, 7, 0)).solvable())             # one swapped pair
        rng = random.Random(0)
        for _ in range(20):
            s = list(range(9)); rng.shuffle(s)
            puz = EightPuzzle(tuple(s))
            if not puz.solvable():
                self.assertFalse(breadth_first(puz).found)
                break

    def test_heuristics_dominance_and_optimality(self):
        puz = EightPuzzle((8, 1, 3, 4, 0, 2, 7, 6, 5))
        self.assertTrue(puz.solvable())
        a_m, a_t, bfs = a_star(puz, puz.manhattan), a_star(puz, puz.misplaced), breadth_first(puz)
        self.assertEqual(a_m.cost, a_t.cost); self.assertEqual(a_m.cost, bfs.cost)           # all optimal
        self.assertLessEqual(a_m.expanded, a_t.expanded)                                      # Manhattan dominates misplaced tiles
        self.assertLess(a_m.expanded, bfs.expanded)
        s = puz.start
        for act in a_m.actions:                                                               # replay the plan
            s = next(n for a, n, _ in puz.successors(s) if a == act)
        self.assertEqual(s, puz.GOAL)

    def test_hardest_instances_have_31_moves(self):
        puz = EightPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1))                                       # a known 31-move instance
        self.assertEqual(a_star(puz, puz.manhattan).cost, 31.0)

    def test_heuristic_values_and_validation(self):
        puz = EightPuzzle((1, 2, 3, 4, 5, 6, 0, 7, 8))
        self.assertEqual(puz.manhattan(puz.start), 2.0)
        self.assertEqual(puz.misplaced(puz.start), 2.0)
        self.assertEqual(puz.manhattan(puz.GOAL), 0.0)
        with self.assertRaises(ValueError):
            EightPuzzle((1, 1, 2, 3, 4, 5, 6, 7, 8))


if __name__ == "__main__":
    unittest.main()

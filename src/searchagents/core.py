"""Uninformed and informed graph search over an abstract problem interface."""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Protocol


class Problem(Protocol):
    def initial(self) -> Hashable: ...
    def is_goal(self, state: Hashable) -> bool: ...
    def successors(self, state: Hashable) -> Iterable[tuple[object, Hashable, float]]: ...   # (action, next state, step cost)


@dataclass
class SearchResult:
    found: bool
    path: list[Hashable] = field(default_factory=list)      # states from start to goal
    actions: list[object] = field(default_factory=list)
    cost: float = float("inf")
    expanded: int = 0
    generated: int = 0
    max_frontier: int = 0


def _build(parent: dict, state, start) -> tuple[list, list]:
    states, actions = [state], []
    while state != start:
        prev, act = parent[state]
        states.append(prev); actions.append(act)
        state = prev
    return states[::-1], actions[::-1]


def breadth_first(problem: Problem) -> SearchResult:
    """BFS (graph search); optimal in number of steps."""
    start = problem.initial()
    if problem.is_goal(start):
        return SearchResult(True, [start], [], 0.0)
    frontier, parent, cost = deque([start]), {}, {start: 0.0}
    seen, expanded, generated, mx = {start}, 0, 1, 1
    while frontier:
        s = frontier.popleft(); expanded += 1
        for act, nxt, c in problem.successors(s):
            if nxt in seen:
                continue
            seen.add(nxt); parent[nxt] = (s, act); cost[nxt] = cost[s] + c; generated += 1
            if problem.is_goal(nxt):
                p, a = _build(parent, nxt, start)
                return SearchResult(True, p, a, cost[nxt], expanded, generated, mx)
            frontier.append(nxt); mx = max(mx, len(frontier))
    return SearchResult(False, expanded=expanded, generated=generated, max_frontier=mx)


def depth_limited(problem: Problem, limit: int) -> SearchResult:
    """Depth-first search with a depth limit (cycle check along the current path only)."""
    start = problem.initial()
    expanded = generated = 0
    best_frontier = 0

    def rec(state, path_states, actions, cost, depth):
        nonlocal expanded, generated, best_frontier
        if problem.is_goal(state):
            return SearchResult(True, path_states[:], actions[:], cost)
        if depth == limit:
            return None
        expanded += 1
        for act, nxt, c in problem.successors(state):
            if nxt in path_states:
                continue
            generated += 1
            path_states.append(nxt); actions.append(act)
            best_frontier = max(best_frontier, depth + 1)
            r = rec(nxt, path_states, actions, cost + c, depth + 1)
            if r:
                return r
            path_states.pop(); actions.pop()
        return None

    r = rec(start, [start], [], 0.0, 0)
    if r:
        r.expanded, r.generated, r.max_frontier = expanded, generated, best_frontier
        return r
    return SearchResult(False, expanded=expanded, generated=generated, max_frontier=best_frontier)


def iterative_deepening(problem: Problem, max_depth: int = 50) -> SearchResult:
    total_exp = total_gen = 0
    for d in range(max_depth + 1):
        r = depth_limited(problem, d)
        total_exp += r.expanded; total_gen += r.generated
        if r.found:
            r.expanded, r.generated = total_exp, total_gen
            return r
    return SearchResult(False, expanded=total_exp, generated=total_gen)


def best_first(problem: Problem, priority: Callable[[float, Hashable], float]) -> SearchResult:
    """Generic best-first graph search ordered by ``priority(g, state)``; re-opens states reached more cheaply."""
    start = problem.initial()
    counter = itertools.count()
    frontier = [(priority(0.0, start), next(counter), 0.0, start)]
    best_g, parent = {start: 0.0}, {}
    expanded, generated, mx = 0, 1, 1
    while frontier:
        _, _, g, s = heapq.heappop(frontier)
        if g > best_g.get(s, float("inf")):
            continue                                   # stale entry
        if problem.is_goal(s):
            p, a = _build(parent, s, start)
            return SearchResult(True, p, a, g, expanded, generated, mx)
        expanded += 1
        for act, nxt, c in problem.successors(s):
            ng = g + c
            if ng < best_g.get(nxt, float("inf")):
                best_g[nxt] = ng; parent[nxt] = (s, act); generated += 1
                heapq.heappush(frontier, (priority(ng, nxt), next(counter), ng, nxt))
                mx = max(mx, len(frontier))
    return SearchResult(False, expanded=expanded, generated=generated, max_frontier=mx)


def uniform_cost(problem: Problem) -> SearchResult:
    return best_first(problem, lambda g, s: g)


def greedy(problem: Problem, h: Callable[[Hashable], float]) -> SearchResult:
    """Greedy best-first: fast, not optimal."""
    return best_first(problem, lambda g, s: h(s))


def a_star(problem: Problem, h: Callable[[Hashable], float]) -> SearchResult:
    """A*: optimal when ``h`` is admissible (and, for this re-opening implementation, even if it is not consistent)."""
    return best_first(problem, lambda g, s: g + h(s))

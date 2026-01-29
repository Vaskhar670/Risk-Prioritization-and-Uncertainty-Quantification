from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class TriangularFuzzyNumber:
    a: float
    b: float
    c: float
    name: str = ""

    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        L = self.a + alpha * (self.b - self.a)
        R = self.c - alpha * (self.c - self.b)
        return (max(0.0, L), max(0.0, R))

    def centroid(self, steps: int = 200) -> float:
        xs = np.linspace(self.a, self.c, steps)
        mu = np.zeros_like(xs)
        left = xs <= self.b
        right = xs >= self.b
        if self.b != self.a:
            mu[left] = (xs[left] - self.a) / (self.b - self.a)
        else:
            mu[left] = 1.0
        if self.c != self.b:
            mu[right] = (self.c - xs[right]) / (self.c - self.b)
        else:
            mu[right] = 1.0
        mu = np.clip(mu, 0, 1)
        if mu.sum() == 0:
            return 0.0
        return float((xs * mu).sum() / mu.sum())

    def scaled(self, k: float) -> "TriangularFuzzyNumber":
        # scale about mode b
        return TriangularFuzzyNumber(self.b + (self.a - self.b) * k,
                                     self.b,
                                     self.b + (self.c - self.b) * k,
                                     name=f"{self.name}_scaled_{k:.2f}")

@dataclass
class IT2Triangular:
    """Interval Type-2 as UMF (outer) and LMF (inner) triangles."""
    umf: TriangularFuzzyNumber
    lmf: TriangularFuzzyNumber
    name: str = ""

    def alpha_outer(self, alpha: float) -> Tuple[float, float]:
        return self.umf.alpha_cut(alpha)

    def alpha_inner(self, alpha: float) -> Tuple[float, float]:
        return self.lmf.alpha_cut(alpha)

# -----------------------------
# Interval arithmetic
# -----------------------------

def interval_mul(x: Tuple[float, float], y: Tuple[float, float]) -> Tuple[float, float]:
    x1, x2 = x; y1, y2 = y
    cand = [x1*y1, x1*y2, x2*y1, x2*y2]
    return (min(cand), max(cand))

def prob_or(x: Tuple[float, float], y: Tuple[float, float]) -> Tuple[float, float]:
    """Probabilistic OR with interval bounds: p(A∪B)=p(A)+p(B)-p(A)p(B)."""
    x1, x2 = x; y1, y2 = y
    f = lambda a, b: a + b - a*b
    cand = [f(x1,y1), f(x1,y2), f(x2,y1), f(x2,y2)]
    return (min(cand), max(cand))

def combine_list(intervals: List[Tuple[float, float]], op) -> Tuple[float, float]:
    if not intervals:
        return (0.0, 0.0)
    cur = intervals[0]
    for nxt in intervals[1:]:
        cur = op(cur, nxt)
    return cur

# -----------------------------
# Fault tree nodes
# -----------------------------

class FTNode:
    def __init__(self, gate: Optional[str] = None, children: Optional[List["FTNode"]] = None,
                 value: Optional[IT2Triangular] = None, name: str = ""):
        self.gate = gate  # None for leaf, else "AND" / "OR"
        self.children = children or []
        self.value = value
        self.name = name or (value.name if value else gate)

    def evaluate_alpha(self, alpha: float) -> Tuple[float, float]:
        if self.gate is None:
            return self.value.alpha_outer(alpha)  # outer interval for conservative propagation
        child_intervals = [ch.evaluate_alpha(alpha) for ch in self.children]
        if self.gate == "AND":
            return combine_list(child_intervals, interval_mul)
        elif self.gate == "OR":
            return combine_list(child_intervals, prob_or)
        else:
            raise ValueError("Unknown gate")

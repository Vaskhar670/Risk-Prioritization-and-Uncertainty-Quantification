from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np


class FuzzyNumber:
    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        raise NotImplementedError


@dataclass
class TriangularFuzzyNumber(FuzzyNumber):
    a: float
    b: float
    c: float
    name: str = ""

    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        left = self.a + alpha * (self.b - self.a)
        right = self.c - alpha * (self.c - self.b)
        return (max(0.0, left), max(0.0, right))


# Interval Type-2 Triangular Fuzzy Number
@dataclass
class IT2Triangular:
    umf: TriangularFuzzyNumber
    lmf: TriangularFuzzyNumber
    name: str = ""

    def alpha(self, alpha: float, policy: str = "outer") -> Tuple[float, float]:
        """
        policy:
        - 'outer'   : pessimistic (UMF)
        - 'inner'   : optimistic (LMF)
        - 'average' : neutral
        """
        if policy == "outer":
            return self.umf.alpha_cut(alpha)
        elif policy == "inner":
            return self.lmf.alpha_cut(alpha)
        elif policy == "average":
            o = self.umf.alpha_cut(alpha)
            i = self.lmf.alpha_cut(alpha)
            return ((o[0] + i[0]) / 2, (o[1] + i[1]) / 2)
        else:
            raise ValueError("Unknown uncertainty policy")


def interval_mul(x: Tuple[float, float], y: Tuple[float, float]) -> Tuple[float, float]:
    x1, x2 = x
    y1, y2 = y
    values = [x1*y1, x1*y2, x2*y1, x2*y2]
    return (min(values), max(values))


def prob_or(x: Tuple[float, float], y: Tuple[float, float]) -> Tuple[float, float]:
    f = lambda a, b: a + b - a*b
    x1, x2 = x
    y1, y2 = y
    values = [f(x1,y1), f(x1,y2), f(x2,y1), f(x2,y2)]
    return (min(values), max(values))


GATE_OPERATORS: Dict[str, Callable] = {
    "AND": interval_mul,
    "OR": prob_or
}


def combine_intervals(intervals: List[Tuple[float, float]],
                      operator: Callable) -> Tuple[float, float]:
    result = intervals[0]
    for interval in intervals[1:]:
        result = operator(result, interval)
    return result


class FTNode:
    """
    Generic Fault Tree Node supporting IT2 fuzzy uncertainty.
    """

    def __init__(self,
                 gate: Optional[str] = None,
                 children: Optional[List["FTNode"]] = None,
                 value: Optional[IT2Triangular] = None,
                 name: str = ""):

        self.gate = gate
        self.children = children or []
        self.value = value
        self.name = name

    def evaluate_alpha(self,
                       alpha: float,
                       uncertainty_policy: str = "outer") -> Tuple[float, float]:

        # Leaf node (basic event)
        if self.gate is None:
            return self.value.alpha(alpha, uncertainty_policy)

        # Gate node
        if self.gate not in GATE_OPERATORS:
            raise ValueError(f"Unsupported gate: {self.gate}")

        operator = GATE_OPERATORS[self.gate]
        child_intervals = [
            child.evaluate_alpha(alpha, uncertainty_policy)
            for child in self.children
        ]

        return combine_intervals(child_intervals, operator)

def __run():
    pump_failure = IT2Triangular(
        umf=TriangularFuzzyNumber(0.02, 0.05, 0.08, "Pump_UMF"),
        lmf=TriangularFuzzyNumber(0.03, 0.05, 0.06, "Pump_LMF"),
        name="Pump Failure"
    )

    valve_failure = IT2Triangular(
        umf=TriangularFuzzyNumber(0.01, 0.03, 0.06, "Valve_UMF"),
        lmf=TriangularFuzzyNumber(0.02, 0.03, 0.04, "Valve_LMF"),
        name="Valve Failure"
    )

    sensor_failure = IT2Triangular(
        umf=TriangularFuzzyNumber(0.005, 0.01, 0.02, "Sensor_UMF"),
        lmf=TriangularFuzzyNumber(0.007, 0.01, 0.015, "Sensor_LMF"),
        name="Sensor Failure"
    )

    # --- Build fault tree ---
    pump_node = FTNode(value=pump_failure, name="Pump")
    valve_node = FTNode(value=valve_failure, name="Valve")
    sensor_node = FTNode(value=sensor_failure, name="Sensor")

    subsystem = FTNode(
        gate="AND",
        children=[pump_node, valve_node],
        name="Subsystem Failure"
    )

    top_event = FTNode(
        gate="OR",
        children=[subsystem, sensor_node],
        name="System Failure"
    )

    # --- Alpha-cut evaluation ---
    print("Alpha-cut based IT2 FFTA Results\n")

    for alpha in np.linspace(0, 1, 6):
        interval = top_event.evaluate_alpha(
            alpha,
            uncertainty_policy="outer"
        )
        print(f"α = {alpha:.2f} → Failure Probability ∈ {interval}")



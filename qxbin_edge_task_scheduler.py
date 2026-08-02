"""
QxBin Edge Tier — Battery-Aware Adaptive Task Prioritizer
========================================================
Use Case: Real-time uncertain decision making on edge / IoT / mobile devices.

A single personal cubit (Binary Probability Matrix) continuously evolves
under fractional exponents (bias**n, (1-bias)**m). The matrix encodes the
trade-off space between:
  - remaining battery
  - task urgency
  - estimated compute cost
  - environmental uncertainty (signal, thermal, etc.)

Measurement collapses the probability cloud into a concrete priority ranking
or "do / defer / drop" action. Zero external dependencies beyond NumPy.
Room temperature. Runs on any laptop, Raspberry Pi, or microcontroller-class
device that can execute Python.

Author: Rupesh Malpani | pikk.company | QxBin Framework
"""

import numpy as np
from typing import Dict, List, Tuple


class QxBinEdgeTaskScheduler:
    """
    Single-cubit QxBin engine specialized for edge task prioritization
    under uncertainty and energy constraints.
    """

    def __init__(self, grid_size: int = 5):
        self.grid_size = grid_size
        # Coordinate axes of the probability matrix:
        # rows  ~ battery / energy dimension
        # cols  ~ urgency / value dimension
        self.state = np.random.rand(grid_size, grid_size).astype(np.float64)
        self._normalize()
        self.history: List[float] = []

    def _normalize(self) -> None:
        s = self.state.sum()
        if s > 1e-12:
            self.state /= s

    def apply_superposition(
        self,
        battery_level: float = 0.65,   # 0..1 remaining energy
        urgency: float = 0.70,         # 0..1 task urgency / value
        n: int = 2,
        m: int = 1,
    ) -> np.ndarray:
        """
        Core QxBin fractional superposition update.

        bias is derived from the instantaneous battery × urgency product.
        Positive/negative exponents (n, m) steer the directed contribution
        of the probability cloud exactly as defined in the QxBin foundation.
        """
        # Composite bias: higher battery + higher urgency → stronger lean
        bias = np.clip(0.45 + 0.45 * (battery_level * urgency), 0.15, 0.92)

        frac = bias ** n
        tail = (1.0 - bias) ** m

        # Coordinate vectors (directed contributions)
        energy_vec = np.linspace(frac, tail, self.grid_size)
        urgency_vec = np.linspace(frac * 0.85 + 0.1, tail, self.grid_size)
        new_matrix = np.outer(energy_vec, urgency_vec)

        # Superposition blend
        self.state = 0.55 * self.state + 0.45 * new_matrix
        self._normalize()
        self.history.append(float(self.state.mean()))
        return self.state

    def measure_priority(self) -> Dict[str, float]:
        """
        Collapse the probability matrix into a classical action recommendation.
        Returns a structured priority score and discrete decision.
        """
        flat = self.state.flatten()
        # Weighted sample
        idx = np.random.choice(len(flat), p=flat)
        row, col = divmod(idx, self.grid_size)

        # Map grid coordinates back to human-readable scores
        energy_score = 1.0 - (row / (self.grid_size - 1))   # top rows = high energy
        urgency_score = col / (self.grid_size - 1)          # right cols = high urgency

        composite = 0.55 * energy_score + 0.45 * urgency_score

        if composite > 0.72:
            decision = "EXECUTE_NOW"
        elif composite > 0.42:
            decision = "DEFER_SHORT"
        else:
            decision = "DROP_OR_SLEEP"

        return {
            "composite_score": round(composite, 4),
            "energy_axis": round(energy_score, 4),
            "urgency_axis": round(urgency_score, 4),
            "decision": decision,
            "collapsed_cell": (int(row), int(col)),
        }

    def rank_tasks(
        self,
        tasks: List[Dict],
        battery_level: float,
    ) -> List[Tuple[str, Dict]]:
        """
        Rank a list of candidate tasks under current battery and uncertainty.
        Each task dict must contain: name, urgency (0-1), estimated_cost (0-1)
        """
        ranked = []
        for t in tasks:
            # Temporarily evolve for this task
            self.apply_superposition(
                battery_level=battery_level,
                urgency=t.get("urgency", 0.5),
                n=2 + int(t.get("estimated_cost", 0.3) * 2),
                m=1,
            )
            result = self.measure_priority()
            ranked.append((t["name"], result))

        # Sort by composite score descending
        ranked.sort(key=lambda x: x[1]["composite_score"], reverse=True)
        return ranked


# ------------------------------------------------------------------
# Demo — typical edge IoT / mobile scenario
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("QxBin Edge Tier — Battery-Aware Adaptive Task Scheduler")
    print("Room-temperature personal cubit on classical hardware")
    print("=" * 60)

    scheduler = QxBinEdgeTaskScheduler(grid_size=5)

    # Simulate a realistic edge device state
    battery = 0.38          # 38 % remaining
    ambient_uncertainty = 0.25

    candidate_tasks = [
        {"name": "sensor_upload",       "urgency": 0.85, "estimated_cost": 0.40},
        {"name": "local_ml_inference",  "urgency": 0.55, "estimated_cost": 0.70},
        {"name": "heartbeat_ping",      "urgency": 0.30, "estimated_cost": 0.10},
        {"name": "firmware_check",      "urgency": 0.15, "estimated_cost": 0.55},
        {"name": "user_notification",   "urgency": 0.75, "estimated_cost": 0.25},
    ]

    print(f"\nBattery remaining: {battery*100:.0f}%")
    print("Evolving QxBin probability matrix under fractional exponents...\n")

    ranked = scheduler.rank_tasks(candidate_tasks, battery_level=battery)

    print("Priority ranking (highest → lowest):")
    print("-" * 50)
    for i, (name, res) in enumerate(ranked, 1):
        print(f"{i}. {name:22s}  score={res['composite_score']:.3f}  →  {res['decision']}")

    print("\nFinal probability matrix (rounded):")
    print(np.round(scheduler.state, 3))
    print("\n✅ Edge cubit decision complete. Ready for real device integration.")

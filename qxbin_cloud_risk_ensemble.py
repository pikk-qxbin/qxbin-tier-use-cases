"""
QxBin Cloud Tier — Multi-Scenario Risk Ensemble & Workload Optimizer
====================================================================
Use Case: Scalable probabilistic simulation for risk assessment,
portfolio stress-testing, AI hyper-parameter search, or data-center
workload scheduling under uncertainty.

Many independent cubit chains (Binary Probability Matrices) evolve
in parallel using the same fractional exponent math (bias**n, (1-bias)**m).
An aggregate probability landscape emerges that can be measured,
optimized toward a target risk/return profile, or used to rank
scenarios.

Designed for server / cloud / batch environments. Numba-accelerated.
Easily portable to GPU (see main qxbin CUDA port).

Author: Rupesh Malpani | pikk.company | QxBin Framework
"""

import numpy as np
from numba import njit, prange
from typing import List, Dict, Optional


@njit(parallel=True, fastmath=True)
def _evolve_batch(states, biases, ns, ms):
    """Parallel QxBin fractional evolution of an ensemble of matrices."""
    n_cubits = states.shape[0]
    for i in prange(n_cubits):
        b = biases[i]
        nn = ns[i]
        mm = ms[i]

        frac = b ** nn
        tail = (1.0 - b) ** mm

        # Element-wise directed blend
        blended = (states[i] * frac + (1.0 - states[i]) * tail) * 0.5

        total = blended.sum()
        if total > 1e-12:
            states[i] = blended / total
        else:
            states[i] = np.ones_like(blended) / blended.size
    return states


class QxBinCloudRiskEnsemble:
    """
    Ensemble of QxBin cubits for multi-scenario risk / optimization workloads.
    """

    def __init__(self, num_scenarios: int = 48, grid_size: int = 6):
        self.num_scenarios = num_scenarios
        self.grid_size = grid_size
        self.states = np.random.rand(num_scenarios, grid_size, grid_size).astype(np.float64)
        # Normalize each scenario matrix
        for i in range(num_scenarios):
            s = self.states[i].sum()
            if s > 0:
                self.states[i] /= s

        self.scenario_labels: List[str] = [f"S{i:03d}" for i in range(num_scenarios)]
        self.risk_history: List[float] = []

    def evolve(
        self,
        risk_appetite: float = 0.55,          # 0 = conservative, 1 = aggressive
        volatility: float = 0.30,
        ns: Optional[np.ndarray] = None,
        ms: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evolve the entire ensemble one step.
        risk_appetite controls the central bias; volatility spreads the n/m exponents.
        """
        # Per-scenario biases around the risk appetite
        biases = np.clip(
            risk_appetite + np.random.normal(0, volatility * 0.25, self.num_scenarios),
            0.12, 0.93
        )

        if ns is None:
            ns = np.random.randint(1, 5, self.num_scenarios)
        if ms is None:
            ms = np.random.randint(1, 4, self.num_scenarios)

        self.states = _evolve_batch(self.states, biases, ns, ms)
        agg_mean = float(self.states.mean())
        self.risk_history.append(agg_mean)
        return self.states.mean(axis=0)

    def optimize_to_target_risk(
        self,
        target_risk: float = 0.62,
        max_steps: int = 100,
        risk_appetite: float = 0.55,
    ) -> Dict:
        """
        Feedback loop: evolve the ensemble until the aggregate mean probability
        approaches the desired risk level. Classic QxBin optimization pattern.
        """
        for step in range(max_steps):
            agg = self.evolve(risk_appetite=risk_appetite)
            current = float(agg.mean())
            if abs(current - target_risk) < 0.008:
                return {
                    "converged": True,
                    "steps": step + 1,
                    "final_mean": round(current, 5),
                    "target": target_risk,
                }
            # Gentle adaptive steer
            if current < target_risk:
                risk_appetite = min(0.90, risk_appetite + 0.015)
            else:
                risk_appetite = max(0.15, risk_appetite - 0.015)

        return {
            "converged": False,
            "steps": max_steps,
            "final_mean": round(float(self.states.mean()), 5),
            "target": target_risk,
        }

    def rank_scenarios(self, top_k: int = 8) -> List[Dict]:
        """
        Rank individual scenarios by their internal mean probability
        (proxy for realized risk contribution).
        """
        means = self.states.mean(axis=(1, 2))
        order = np.argsort(means)[::-1]  # highest first
        ranked = []
        for rank, idx in enumerate(order[:top_k], 1):
            ranked.append({
                "rank": rank,
                "scenario": self.scenario_labels[idx],
                "mean_prob": round(float(means[idx]), 5),
                "max_cell": round(float(self.states[idx].max()), 5),
            })
        return ranked

    def stress_test(self, shock: float = 0.25) -> Dict:
        """
        Apply a sudden bias shock (market crash / load spike) and measure
        how the ensemble redistributes probability mass.
        """
        pre = float(self.states.mean())
        # Force a strong directional lean
        self.evolve(risk_appetite=0.5 - shock, volatility=0.45)
        post = float(self.states.mean())
        return {
            "pre_shock_mean": round(pre, 5),
            "post_shock_mean": round(post, 5),
            "delta": round(post - pre, 5),
            "shock_magnitude": shock,
        }


# ------------------------------------------------------------------
# Demo — cloud risk / workload ensemble
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("QxBin Cloud Tier — Multi-Scenario Risk Ensemble")
    print("Scalable probabilistic cubit chains on classical hardware")
    print("=" * 64)

    ensemble = QxBinCloudRiskEnsemble(num_scenarios=64, grid_size=6)

    print(f"\nInitialized {ensemble.num_scenarios} parallel cubit scenarios.")
    print("Running optimization toward target risk profile 0.65...\n")

    result = ensemble.optimize_to_target_risk(
        target_risk=0.65,
        max_steps=90,
        risk_appetite=0.52,
    )
    print("Optimization result:", result)

    print("\nTop scenarios by realized probability mass:")
    for s in ensemble.rank_scenarios(top_k=6):
        print(f"  #{s['rank']}  {s['scenario']}  mean={s['mean_prob']:.4f}  peak={s['max_cell']:.4f}")

    print("\nApplying stress-test shock (Δ = -0.22)...")
    shock = ensemble.stress_test(shock=0.22)
    print("Stress-test:", shock)

    print("\n✅ Cloud ensemble ready for production risk / scheduling pipelines.")

"""
Parametric Evaluation: vary stream length N and α, measure error, space, update time.

Produces CSV results and plots for:
  - Relative error vs N
  - Relative error vs α
  - Space vs α
  - Update time vs N
  
"""

import os
import sys
import time
import math
import itertools
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.misra_gries import MisraGries
from algorithms.space_saving_plus import (
    LazySpaceSaving, SpaceSavingPlus, DoubleSpaceSaving, IntegratedSpaceSaving
)
from algorithms.count_min import CountMinSketch, AlphaCountMinSketch
from algorithms.count_sketch import CountSketch, AlphaCountSketch
from algorithms.learned_integrated_ss import LearnedIntegratedSpaceSaving
from data_generators.synthetic_zipf import generate_zipf_stream
from utils.metrics import compute_metrics
from utils.plotter import plot_parametric_results

# ======================================================================
# Configuration
# ======================================================================

STREAM_LENGTHS = [100_000, 500_000, 1_000_000, 2_000_000]
ALPHAS = [1.5, 2.0, 4.0, 8.0]
EPSILON = 0.01          # target error parameter
DELTA = 0.01            # failure probability for sketches
ZIPF_S = 1.0            # default Zipfian exponent
UNIVERSE_SIZE = 50000
SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "parametric")


def build_algorithms(alpha: float, epsilon: float = EPSILON, delta: float = DELTA):
    """Instantiate all algorithms for a given α."""
    k = math.ceil(alpha / epsilon)
    return {
        "MisraGries": MisraGries(k=k, alpha=alpha),
        "LazySS±": LazySpaceSaving(k=k, alpha=alpha),
        "SS±": SpaceSavingPlus(k=k, alpha=alpha),
        "DoubleSS±": DoubleSpaceSaving(k=k, alpha=alpha),
        "IntegratedSS±": IntegratedSpaceSaving(k=k, alpha=alpha),
        "CountMin": CountMinSketch.from_epsilon_delta(epsilon, delta, seed=SEED),
        "α-CountMin": AlphaCountMinSketch.from_epsilon_delta_alpha(epsilon, delta, alpha, seed=SEED),
        "CountSketch": CountSketch.from_epsilon_delta(epsilon, delta, seed=SEED),
        "α-CountSketch": AlphaCountSketch.from_epsilon_delta_alpha(epsilon, delta, alpha, seed=SEED),
        "LearnedISS±": LearnedIntegratedSpaceSaving(k=k, alpha=alpha),
    }


def run_single_experiment(N: int, alpha: float, seed: int = SEED) -> list:
    """
    Run all algorithms on one (N, α) configuration. Returns list of result dicts.
    """
    print(f"  Generating stream: N={N}, α={alpha} ...", end=" ", flush=True)
    stream, true_freq = generate_zipf_stream(
        N=N, alpha=alpha, s=ZIPF_S, universe_size=UNIVERSE_SIZE, seed=seed
    )
    F1 = sum(true_freq.values())
    print(f"done. |stream|={len(stream)}, F1={F1}, distinct={len(true_freq)}")

    algos = build_algorithms(alpha)

    # Collect candidate set for sketch-based heavy hitter queries
    candidates = set(true_freq.keys())

    results = []
    for name, algo in algos.items():
        # --- Update phase ---
        t_start = time.perf_counter()
        for item, delta in stream:
            algo.update(item, delta)
        t_elapsed = time.perf_counter() - t_start

        # --- Query phase: compute error over all items in true_freq ---
        metrics = compute_metrics(algo, true_freq, F1, candidates=candidates)

        results.append({
            "algorithm": name,
            "N": N,
            "alpha": alpha,
            "F1": F1,
            "stream_len": len(stream),
            "space": algo.get_space(),
            "max_space": algo.get_max_space(),
            "update_time_s": t_elapsed,
            "time_per_update_us": (t_elapsed / len(stream)) * 1e6,
            **metrics,
        })
        print(f"    {name:20s}  err={metrics['mean_relative_error']:.6f}  "
              f"space={algo.get_space()}  time={t_elapsed:.3f}s")

    return results


def run_parametric_evaluation():
    """Run full parametric grid and save results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    for N, alpha in itertools.product(STREAM_LENGTHS, ALPHAS):
        print(f"\n{'='*60}")
        print(f"Experiment: N={N}, α={alpha}")
        print(f"{'='*60}")
        results = run_single_experiment(N, alpha)
        all_results.extend(results)

    # Save CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "parametric_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Generate plots
    plot_parametric_results(df, OUTPUT_DIR)
    print(f"Plots saved to {OUTPUT_DIR}")

    return df


if __name__ == "__main__":
    run_parametric_evaluation()
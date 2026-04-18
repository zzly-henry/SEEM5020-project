"""
Advanced Evaluation: Learned-Integrated SpaceSaving± vs baseline algorithms.

Focuses on comparing the advanced design (LearnedISS±) against standard
IntegratedSS± and other methods, especially under Zipfian distributions
where the learned predictor should shine.

Evaluates:
  - Error reduction from learned hot-item allocation
  - Impact of fixed_ratio parameter
  - Impact of window_size parameter
  - Performance across Zipfian exponents and α values
  
"""

import os
import sys
import time
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.space_saving_plus import IntegratedSpaceSaving
from algorithms.learned_integrated_ss import LearnedIntegratedSpaceSaving
from algorithms.count_min import CountMinSketch
from algorithms.count_sketch import CountSketch
from algorithms.misra_gries import MisraGries

from data_generators.synthetic_zipf import generate_zipf_stream
from data_generators.synthetic_uniform import generate_uniform_stream

from utils.metrics import compute_metrics
from utils.plotter import plot_advanced_results

# ======================================================================
# Configuration
# ======================================================================

N_DEFAULT = 500_000
ALPHAS = [1.5, 2.0, 4.0, 8.0]
ZIPF_EXPONENTS = [1.0, 1.5, 2.0]
EPSILON = 0.01
DELTA = 0.01
SEED = 42
UNIVERSE_SIZE = 50000

# Learned-ISS± hyperparameter sweeps
FIXED_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.5]
WINDOW_SIZES = [200, 500, 1000, 2000, 5000]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "advanced")


# ======================================================================
# Experiment 1: LearnedISS± vs baselines across datasets
# ======================================================================

def experiment_baseline_comparison():
    """Compare LearnedISS± against baselines across Zipf exponents and α values."""
    print("\n" + "=" * 60)
    print("Experiment 1: Learned ISS± vs Baselines")
    print("=" * 60)

    results = []
    for s in ZIPF_EXPONENTS:
        for alpha in ALPHAS:
            print(f"\n  Zipf s={s}, α={alpha}")
            stream, true_freq = generate_zipf_stream(
                N=N_DEFAULT, alpha=alpha, s=s,
                universe_size=UNIVERSE_SIZE, seed=SEED
            )
            F1 = sum(true_freq.values())
            candidates = set(true_freq.keys())
            k = math.ceil(alpha / EPSILON)

            algos = {
                "IntegratedSS±": IntegratedSpaceSaving(k=k, alpha=alpha),
                "LearnedISS±": LearnedIntegratedSpaceSaving(
                    k=k, alpha=alpha, window_size=1000, fixed_ratio=0.2
                ),
                "MisraGries": MisraGries(k=k, alpha=alpha),
                "CountMin": CountMinSketch.from_epsilon_delta(EPSILON, DELTA, seed=SEED),
            }

            for algo_name, algo in algos.items():
                t_start = time.perf_counter()
                for item, delta in stream:
                    algo.update(item, delta)
                t_elapsed = time.perf_counter() - t_start

                metrics = compute_metrics(algo, true_freq, F1, candidates=candidates)
                results.append({
                    "experiment": "baseline_comparison",
                    "algorithm": algo_name,
                    "zipf_s": s,
                    "alpha": alpha,
                    "N": N_DEFAULT,
                    "F1": F1,
                    "space": algo.get_space(),
                    "update_time_s": t_elapsed,
                    **metrics,
                })
                print(f"    {algo_name:20s}  err={metrics['mean_relative_error']:.6f}")

    return results


# ======================================================================
# Experiment 2: Impact of fixed_ratio
# ======================================================================

def experiment_fixed_ratio():
    """Sweep fixed_ratio for LearnedISS± on Zipf s=1.5, α=2.0."""
    print("\n" + "=" * 60)
    print("Experiment 2: Impact of fixed_ratio")
    print("=" * 60)

    alpha = 2.0
    s = 1.5
    stream, true_freq = generate_zipf_stream(
        N=N_DEFAULT, alpha=alpha, s=s,
        universe_size=UNIVERSE_SIZE, seed=SEED
    )
    F1 = sum(true_freq.values())
    candidates = set(true_freq.keys())
    k = math.ceil(alpha / EPSILON)

    results = []
    for fr in FIXED_RATIOS:
        algo = LearnedIntegratedSpaceSaving(
            k=k, alpha=alpha, window_size=1000, fixed_ratio=fr
        )
        t_start = time.perf_counter()
        for item, delta in stream:
            algo.update(item, delta)
        t_elapsed = time.perf_counter() - t_start

        metrics = compute_metrics(algo, true_freq, F1, candidates=candidates)
        results.append({
            "experiment": "fixed_ratio_sweep",
            "algorithm": "LearnedISS±",
            "fixed_ratio": fr,
            "alpha": alpha,
            "zipf_s": s,
            "N": N_DEFAULT,
            "F1": F1,
            "space": algo.get_space(),
            "update_time_s": t_elapsed,
            **metrics,
        })
        print(f"  fixed_ratio={fr:.2f}  err={metrics['mean_relative_error']:.6f}  "
              f"fixed_slots={algo.k_fixed}  mutable_slots={algo.k_mutable}")

    # Also run baseline IntegratedSS± for reference
    algo = IntegratedSpaceSaving(k=k, alpha=alpha)
    for item, delta in stream:
        algo.update(item, delta)
    metrics = compute_metrics(algo, true_freq, F1, candidates=candidates)
    results.append({
        "experiment": "fixed_ratio_sweep",
        "algorithm": "IntegratedSS± (baseline)",
        "fixed_ratio": 0.0,
        "alpha": alpha,
        "zipf_s": s,
        "N": N_DEFAULT,
        "F1": F1,
        "space": algo.get_space(),
        "update_time_s": 0,
        **metrics,
    })

    return results


# ======================================================================
# Experiment 3: Impact of window_size
# ======================================================================

def experiment_window_size():
    """Sweep window_size for LearnedISS± on Zipf s=1.5, α=2.0."""
    print("\n" + "=" * 60)
    print("Experiment 3: Impact of window_size")
    print("=" * 60)

    alpha = 2.0
    s = 1.5
    stream, true_freq = generate_zipf_stream(
        N=N_DEFAULT, alpha=alpha, s=s,
        universe_size=UNIVERSE_SIZE, seed=SEED
    )
    F1 = sum(true_freq.values())
    candidates = set(true_freq.keys())
    k = math.ceil(alpha / EPSILON)

    results = []
    for ws in WINDOW_SIZES:
        algo = LearnedIntegratedSpaceSaving(
            k=k, alpha=alpha, window_size=ws, fixed_ratio=0.2
        )
        t_start = time.perf_counter()
        for item, delta in stream:
            algo.update(item, delta)
        t_elapsed = time.perf_counter() - t_start

        metrics = compute_metrics(algo, true_freq, F1, candidates=candidates)
        results.append({
            "experiment": "window_size_sweep",
            "algorithm": "LearnedISS±",
            "window_size": ws,
            "alpha": alpha,
            "zipf_s": s,
            "N": N_DEFAULT,
            "F1": F1,
            "space": algo.get_space(),
            "update_time_s": t_elapsed,
            **metrics,
        })
        print(f"  window_size={ws:5d}  err={metrics['mean_relative_error']:.6f}")

    return results


# ======================================================================
# Experiment 4: Learned vs Integrated on Uniform (should show less benefit)
# ======================================================================

def experiment_uniform_comparison():
    """Show that LearnedISS± has less advantage on uniform data."""
    print("\n" + "=" * 60)
    print("Experiment 4: Learned ISS± on Uniform Data")
    print("=" * 60)

    alpha = 2.0
    stream, true_freq = generate_uniform_stream(
        N=N_DEFAULT, alpha=alpha, universe_size=UNIVERSE_SIZE, seed=SEED
    )
    F1 = sum(true_freq.values())
    candidates = set(true_freq.keys())
    k = math.ceil(alpha / EPSILON)

    results = []
    for algo_name, algo in [
        ("IntegratedSS±", IntegratedSpaceSaving(k=k, alpha=alpha)),
        ("LearnedISS±", LearnedIntegratedSpaceSaving(k=k, alpha=alpha)),
    ]:
        t_start = time.perf_counter()
        for item, delta in stream:
            algo.update(item, delta)
        t_elapsed = time.perf_counter() - t_start

        metrics = compute_metrics(algo, true_freq, F1, candidates=candidates)
        results.append({
            "experiment": "uniform_comparison",
            "algorithm": algo_name,
            "dataset": "Uniform",
            "alpha": alpha,
            "N": N_DEFAULT,
            "F1": F1,
            "space": algo.get_space(),
            "update_time_s": t_elapsed,
            **metrics,
        })
        print(f"  {algo_name:20s}  err={metrics['mean_relative_error']:.6f}")

    return results


# ======================================================================
# Main
# ======================================================================

def run_advanced_evaluation():
    """Run all advanced experiments and save results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    all_results.extend(experiment_baseline_comparison())
    all_results.extend(experiment_fixed_ratio())
    all_results.extend(experiment_window_size())
    all_results.extend(experiment_uniform_comparison())

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "advanced_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_advanced_results(df, OUTPUT_DIR)
    print(f"Plots saved to {OUTPUT_DIR}")

    return df


if __name__ == "__main__":
    run_advanced_evaluation()
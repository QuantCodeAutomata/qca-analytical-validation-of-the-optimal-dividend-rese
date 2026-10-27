"""
Experiment 2: Numerical replication of policy regions and comparative statics.

Reproduces the paper's Section 6 / Figure 2-4 results:
  - Figure 2: w(z) and w'(z) for eta=0.1 vs eta≈0 (xi=1 baseline).
  - Figure 3: w(z) and w'(z) for xi=0, eta=0.1.
  - Figure 4: Threshold sensitivity w.r.t. xi, eta, p, sigma.

All results are saved to results/.

Run:
    python -m exp.exp2_numerical_replication
"""

from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import ModelParams
from src.solver import solve_qvi, make_grid
from src.diagnostics import classify_regions, check_structural_properties

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Grid settings (implementation assumption) ──────────────────
N_GRID = 600          # number of grid points
Z_MAX = 4.0           # upper truncation of yield domain
MAX_ITER = 4000
TOL = 1e-8


def _solve_scenario(
    params: ModelParams,
    label: str,
    N: int = N_GRID,
    z_max: float = Z_MAX,
    verbose: bool = False,
) -> dict:
    """
    Solve and classify a single parameter scenario.

    Returns dict with params, z_grid, w, w', J_tilde, regions, thresholds.
    """
    if not params.is_finite():
        print(f"  [SKIP] {label}: tilde_rho={params.tilde_rho:.4f} <= 0 (infinite regime)")
        return None

    z_grid = make_grid(z_max, N)
    result = solve_qvi(params, z_grid, max_iter=MAX_ITER, tol=TOL, verbose=verbose)
    w = result['w']
    w_prime = result['w_prime']
    J_tilde = result['J_tilde']
    hjb_resid = result['hjb_resid']

    regions = classify_regions(params, z_grid, w, w_prime, J_tilde, hjb_resid)
    tilde_rho = params.tilde_rho
    bar_z = params.bar_z

    print(f"  {label}: tilde_rho={tilde_rho:.5f}, bar_z={bar_z:.5f}, "
          f"z_*={regions.z_star}, z^*={regions.z_upper}, "
          f"z^**={regions.z_upper2}, z_0={regions.z_0_reset}, "
          f"z_lambda={regions.z_lambda}, C2={regions.has_second_continuation}")

    return {
        "label": label,
        "params": params,
        "z_grid": z_grid,
        "w": w,
        "w_prime": w_prime,
        "J_tilde": J_tilde,
        "hjb_resid": hjb_resid,
        "regions": regions,
        "tilde_rho": tilde_rho,
        "bar_z": bar_z,
    }


# ──────────────────────────────────────────────────────────────
# SCENARIO SET A: Figure 2  (eta=0.1 vs eta≈0, xi=1)
# ──────────────────────────────────────────────────────────────

def run_scenario_A() -> list[dict]:
    """
    Figure 2 style: eta=0.1 vs eta=0 with xi=1 baseline.

    mu=0.07, rho=0.08, sigma=0.25, p=0.5, xi=1, lambda=0.5
    """
    print("\n=== Scenario A: Figure 2 (eta=0.1 vs eta≈0) ===")

    base_kw = dict(mu=0.07, sigma=0.25, rho=0.08, p=0.5, lambda_=0.5, xi=1.0)
    scenarios_A = [
        ModelParams(**base_kw, eta=0.1),
        ModelParams(**base_kw, eta=1e-6),  # eta ≈ 0
    ]
    labels_A = ["eta=0.1", "eta≈0"]

    results_A = []
    for params, label in zip(scenarios_A, labels_A):
        r = _solve_scenario(params, label)
        if r is not None:
            results_A.append(r)

    _plot_figure2_style(results_A, "exp2_figure2_style.png")
    return results_A


# ──────────────────────────────────────────────────────────────
# SCENARIO SET B: Figure 3  (xi=0, eta=0.1)
# ──────────────────────────────────────────────────────────────

def run_scenario_B() -> dict:
    """
    Figure 3 style: xi=0, eta=0.1.

    mu=0.07, rho=0.08, sigma=0.25, p=0.5, xi=0, eta=0.1, lambda=0.5
    """
    print("\n=== Scenario B: Figure 3 (xi=0, eta=0.1) ===")

    params_B = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                           lambda_=0.5, xi=0.0, eta=0.1)
    result_B = _solve_scenario(params_B, "xi=0, eta=0.1")

    if result_B is not None:
        _plot_figure3_style(result_B, "exp2_figure3_style.png")

    # Verify xi=0 structural collapse
    if result_B is not None:
        from src.operators import compute_reset_operator
        J_tilde = compute_reset_operator(params_B, result_B['z_grid'], result_B['w'])
        expected_J = params_B.a**params_B.p * result_B['w'][0]
        max_diff = float(np.max(np.abs(J_tilde - expected_J)))
        print(f"  xi=0 collapse: max|J_tilde - a^p*w(0)| = {max_diff:.2e}  "
              f"(a^p*w(0)={expected_J:.6f})")
        result_B['xi0_collapse_max_diff'] = max_diff
        result_B['xi0_collapse_passed'] = max_diff < 0.1

    return result_B


# ──────────────────────────────────────────────────────────────
# SCENARIO SET C: Figure 4 (Sensitivity analysis)
# ──────────────────────────────────────────────────────────────

def run_scenario_C() -> dict:
    """
    Figure 4 style: threshold sensitivity to xi, eta, p, sigma.

    Baseline: mu=0.07, rho=0.08, sigma=0.25, p=0.5, xi=0.1, eta=0.1, lambda=0.5
    Vary one parameter at a time (ranges are implementation assumptions where
    exact paper ranges are unavailable).
    """
    print("\n=== Scenario C: Figure 4 (Sensitivity Analysis) ===")

    base = dict(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                lambda_=0.5, xi=0.1, eta=0.1)

    # ── Implementation assumption: sweep ranges (not stated in paper excerpt) ──
    sweeps = {
        "xi":    (np.linspace(0.0, 2.0, 9),  "xi"),
        "eta":   (np.linspace(0.0, 0.4, 9),  "eta"),
        "p":     (np.linspace(0.2, 0.8, 9),  "p"),
        "sigma": (np.linspace(0.10, 0.50, 9), "sigma"),
    }

    sensitivity_records = {key: [] for key in sweeps}

    for param_name, (values, _) in sweeps.items():
        print(f"\n  Varying {param_name}:")
        for val in values:
            kw = dict(**base)
            kw[param_name] = float(val)
            try:
                params = ModelParams(**kw)
            except ValueError as e:
                print(f"    {param_name}={val:.3f}: invalid ({e})")
                continue
            if not params.is_finite():
                print(f"    {param_name}={val:.3f}: tilde_rho={params.tilde_rho:.4f} (infinite)")
                continue

            r = _solve_scenario(params, f"{param_name}={val:.3f}", verbose=False)
            if r is None:
                continue

            reg = r['regions']
            sensitivity_records[param_name].append({
                param_name: round(float(val), 5),
                "tilde_rho": round(r['tilde_rho'], 6),
                "bar_z": round(r['bar_z'], 6),
                "z_star": reg.z_star,
                "z_upper": reg.z_upper,
                "z_upper2": reg.z_upper2,
                "z_0_reset": reg.z_0_reset,
                "z_lambda": reg.z_lambda,
                "has_second_continuation": reg.has_second_continuation,
            })

    # Produce threshold-sensitivity plots
    _plot_figure4_style(sensitivity_records, sweeps)

    # Save tables
    all_rows = []
    for param_name, records in sensitivity_records.items():
        for rec in records:
            row = dict(varied_param=param_name, **rec)
            all_rows.append(row)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp2_sensitivity.csv"), index=False)
    print(f"\n  Sensitivity table saved.")

    return {"sensitivity": sensitivity_records}


# ──────────────────────────────────────────────────────────────
# ROBUSTNESS CHECKS
# ──────────────────────────────────────────────────────────────

def run_robustness_checks() -> dict:
    """
    Confirm threshold stability under grid refinement and tighter tolerances.
    """
    print("\n=== Robustness Checks ===")

    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.1, eta=0.1)

    configs = [
        (300, 4.0, 1e-7, "N=300"),
        (600, 4.0, 1e-8, "N=600"),
        (1000, 4.0, 1e-8, "N=1000"),
        (600, 6.0, 1e-8, "zmax=6"),
    ]

    records = []
    for N, z_max, tol, label in configs:
        z_grid = make_grid(z_max, N)
        result = solve_qvi(params, z_grid, max_iter=5000, tol=tol, verbose=False)
        w = result['w']
        w_prime = result['w_prime']
        J_tilde = result['J_tilde']
        hjb_resid = result['hjb_resid']
        reg = classify_regions(params, z_grid, w, w_prime, J_tilde, hjb_resid)
        records.append({
            "config": label, "N": N, "z_max": z_max,
            "z_star": reg.z_star,
            "z_upper": reg.z_upper,
            "z_upper2": reg.z_upper2,
        })
        print(f"  {label}: z_*={reg.z_star}, z^*={reg.z_upper}, z^**={reg.z_upper2}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RESULTS_DIR, "exp2_robustness.csv"), index=False)
    return {"robustness": records}


# ──────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────

def _plot_figure2_style(results: list[dict], filename: str) -> None:
    """Plot w(z) and w'(z) for Figure 2 style comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r"Figure 2 Style: $w(z)$ and $w'(z)$ for varying $\eta$"
                 r" ($\xi=1$, baseline)", fontsize=12)

    colors = ['black', 'blue', 'red', 'green']
    for i, r in enumerate(results):
        z_g = r['z_grid']
        display_mask = z_g <= 2.0
        axes[0].plot(z_g[display_mask], r['w'][display_mask],
                     color=colors[i % len(colors)], lw=2, label=r['label'])
        axes[1].plot(z_g[display_mask], r['w_prime'][display_mask],
                     color=colors[i % len(colors)], lw=2, label=r['label'])

        reg = r['regions']
        if reg.z_star is not None:
            for ax in axes:
                ax.axvline(reg.z_star, color=colors[i % len(colors)],
                           ls='--', lw=1, alpha=0.5)

    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xlabel(r"$z = c/x$")
    axes[0].set_ylabel(r"$w(z)$")
    axes[0].set_title("Value function")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color='k', lw=0.8, ls='--')
    axes[1].set_xlabel(r"$z = c/x$")
    axes[1].set_ylabel(r"$w'(z)$")
    axes[1].set_title("Derivative of value function")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_figure3_style(result: dict, filename: str) -> None:
    """Plot w(z) and w'(z) for xi=0 scenario (Figure 3 style)."""
    from src.operators import compute_reset_operator
    r = result
    z_g = r['z_grid']
    display_mask = z_g <= 2.0

    J_tilde = compute_reset_operator(r['params'], z_g, r['w'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(r"Figure 3 Style: $\xi=0$, $\eta=0.1$ — Reset collapses to $a^p w(0)$",
                 fontsize=12)

    axes[0].plot(z_g[display_mask], r['w'][display_mask], 'k-', lw=2, label=r"$w(z)$")
    axes[0].plot(z_g[display_mask], J_tilde[display_mask], 'r--', lw=1.5,
                 label=r"$\tilde{J}[w](z)$")
    axes[0].axhline(r['params'].a**r['params'].p * r['w'][0], color='orange',
                    ls=':', lw=1.5, label=r"$a^p w(0)$")
    reg = r['regions']
    if reg.z_star is not None:
        axes[0].axvline(reg.z_star, color='blue', ls='--', lw=1.5, label=r"$z_*$")
    axes[0].set_xlabel(r"$z = c/x$")
    axes[0].set_ylabel(r"$w(z)$")
    axes[0].set_title("Value function and reset obstacle")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(z_g[display_mask], r['w_prime'][display_mask], 'b-', lw=1.5,
                 label=r"$w'(z)$")
    axes[1].axhline(0, color='k', ls='--', lw=0.8)
    if reg.z_star is not None:
        axes[1].axvline(reg.z_star, color='blue', ls='--', lw=1.5, label=r"$z_*$")
    axes[1].set_xlabel(r"$z = c/x$")
    axes[1].set_ylabel(r"$w'(z)$")
    axes[1].set_title("Derivative of value function")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_figure4_style(sensitivity_records: dict, sweeps: dict) -> None:
    """Produce threshold vs parameter plots (Figure 4 style)."""
    n_params = len(sweeps)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Figure 4 Style: Threshold Sensitivity Analysis", fontsize=13)
    axes_flat = axes.flatten()

    threshold_styles = {
        "z_star":   ("black", "o-",  r"$z_*$"),
        "z_upper":  ("blue",  "s--", r"$z^*$"),
        "z_upper2": ("red",   "^:",  r"$z^{**}$"),
        "z_0_reset": ("green", "D-.", r"$z_0$ (reset target)"),
    }

    for ax_idx, (param_name, (values, _)) in enumerate(sweeps.items()):
        ax = axes_flat[ax_idx]
        records = sensitivity_records[param_name]
        if not records:
            ax.set_title(f"No data for {param_name}")
            continue

        df = pd.DataFrame(records)

        for thresh_key, (color, style, lbl) in threshold_styles.items():
            col_data = df[thresh_key].values
            valid = [v is not None and not np.isnan(float(v if v is not None else np.nan))
                     for v in col_data]
            x_vals = df[param_name].values[valid]
            y_vals = np.array([float(v) for v, ok in zip(col_data, valid) if ok])
            if len(x_vals) > 0:
                ax.plot(x_vals, y_vals, style, color=color, lw=1.5,
                        markersize=5, label=lbl)

        # bar_z reference
        if 'bar_z' in df.columns:
            ax.plot(df[param_name].values, df['bar_z'].values,
                    'gray', ls=':', lw=1, label=r"$\bar{z}$")

        ax.set_xlabel(param_name)
        ax.set_ylabel("Threshold value")
        ax.set_title(f"Thresholds vs {param_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp2_figure4_sensitivity.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_policy_cone(results_list: list[dict], filename: str) -> None:
    """
    Figure 1 style: policy regions in (x, c) space as rays c = z*x.

    For each scenario, shade the increase, continuation, reset, and
    stopping cones using the yield thresholds.
    """
    fig, axes = plt.subplots(1, len(results_list),
                             figsize=(6 * len(results_list), 5))
    if len(results_list) == 1:
        axes = [axes]

    x_arr = np.linspace(0.01, 2.0, 200)

    for ax, r in zip(axes, results_list):
        reg = r['regions']
        params = r['params']

        # Shade cones
        z_pairs = [
            (0, reg.z_star or 0, 'lightblue', 'Increase'),
            (reg.z_star or 0, reg.z_upper or 0, 'lightgreen', 'Continuation'),
            (reg.z_upper or 0, reg.z_upper2 or reg.z_upper or 0, 'lightyellow', 'Reset'),
            (reg.z_upper2 or 0, reg.z_lambda or Z_MAX, 'lightsalmon', 'C2/Stop'),
        ]

        for z_lo, z_hi, color, lbl in z_pairs:
            if z_lo is None or z_hi is None or z_lo >= z_hi:
                continue
            c_lo = z_lo * x_arr
            c_hi = z_hi * x_arr
            ax.fill_between(x_arr, c_lo, c_hi, alpha=0.4, color=color, label=lbl)

        # Boundary rays
        for z_thresh, label, color in [
            (reg.z_star, r"$z_*$", 'blue'),
            (reg.z_upper, r"$z^*$", 'red'),
            (reg.z_upper2, r"$z^{**}$", 'darkred'),
        ]:
            if z_thresh is not None:
                ax.plot(x_arr, z_thresh * x_arr, color=color, lw=1.5,
                        ls='--', label=f"{label}={z_thresh:.3f}")

        ax.set_xlabel(r"$x$ (equity)")
        ax.set_ylabel(r"$c$ (dividend rate)")
        ax.set_title(f"Policy Cone: {r['label']}")
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("EXPERIMENT 2: Numerical Replication of Policy Regions")
    print("=" * 65)
    print(f"Grid settings: N={N_GRID}, z_max={Z_MAX}, tol={TOL}")
    print("(lambda=0.5 — implementation assumption; not in paper excerpt)")
    print("(Sensitivity sweep ranges are implementation assumptions)")

    # Scenario A: Figure 2
    results_A = run_scenario_A()

    # Scenario B: Figure 3
    result_B = run_scenario_B()

    # Scenario C: Figure 4 sensitivity
    result_C = run_scenario_C()

    # Robustness
    result_rob = run_robustness_checks()

    # Policy cone plots
    if results_A:
        _plot_policy_cone(results_A, "exp2_policy_cone_A.png")

    # ── Summary table ──
    rows = []
    for r in results_A:
        reg = r['regions']
        rows.append({
            "scenario": "A_Fig2",
            "label": r['label'],
            "eta": r['params'].eta,
            "xi": r['params'].xi,
            "tilde_rho": round(r['tilde_rho'], 6),
            "bar_z": round(r['bar_z'], 6),
            "z_star": reg.z_star,
            "z_upper": reg.z_upper,
            "z_upper2": reg.z_upper2,
            "z_0_reset": reg.z_0_reset,
            "z_lambda": reg.z_lambda,
            "has_C2": reg.has_second_continuation,
        })
    if result_B is not None:
        reg = result_B['regions']
        rows.append({
            "scenario": "B_Fig3",
            "label": result_B['label'],
            "eta": result_B['params'].eta,
            "xi": result_B['params'].xi,
            "tilde_rho": round(result_B['tilde_rho'], 6),
            "bar_z": round(result_B['bar_z'], 6),
            "z_star": reg.z_star,
            "z_upper": reg.z_upper,
            "z_upper2": reg.z_upper2,
            "z_0_reset": reg.z_0_reset,
            "z_lambda": reg.z_lambda,
            "has_C2": reg.has_second_continuation,
        })

    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp2_policy_regions.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nPolicy regions table saved to {csv_path}")

    # ── Qualitative hypotheses check ──
    print("\n=== Qualitative Hypotheses ===")
    if len(results_A) == 2:
        r_eta01 = next((r for r in results_A if "eta=0.1" in r['label']), None)
        r_eta0  = next((r for r in results_A if "eta≈0" in r['label']), None)
        if r_eta01 and r_eta0:
            zstar01 = r_eta01['regions'].z_star or 0
            zstar0  = r_eta0['regions'].z_star or 0
            zupper01 = r_eta01['regions'].z_upper or 0
            zupper0  = r_eta0['regions'].z_upper or 0
            print(f"  eta=0.1 z_*={zstar01:.4f}, z^*={zupper01:.4f}")
            print(f"  eta≈0   z_*={zstar0:.4f},  z^*={zupper0:.4f}")
            eta_wider = (zupper01 - zstar01) >= (zupper0 - zstar0)
            print(f"  eta=0.1 widens inaction band vs eta≈0: {eta_wider}")

    if result_B is not None:
        reg_B = result_B['regions']
        print(f"  xi=0 scenario:")
        print(f"    z_0_reset = {reg_B.z_0_reset} (should ≈ z_* = {reg_B.z_star})")
        print(f"    has_second_continuation = {reg_B.has_second_continuation} (should be False)")
        print(f"    xi=0 collapse passed: {result_B.get('xi0_collapse_passed', 'N/A')}")

    # ── JSON summary ──
    summary = {
        "experiment": "exp2",
        "grid": {"N": N_GRID, "z_max": Z_MAX, "tol": TOL},
        "lambda_assumption": 0.5,
        "sweep_range_assumption": "Not from paper; implementation-chosen",
        "baseline_tilde_rho": 0.0528125,
        "baseline_bar_z": 0.105625,
        "scenario_A": [
            {
                "label": r['label'],
                "z_star": r['regions'].z_star,
                "z_upper": r['regions'].z_upper,
                "z_upper2": r['regions'].z_upper2,
                "has_C2": r['regions'].has_second_continuation,
            }
            for r in results_A
        ],
        "scenario_B_xi0": {
            "z_star": result_B['regions'].z_star if result_B else None,
            "z_0_reset": result_B['regions'].z_0_reset if result_B else None,
            "has_C2": result_B['regions'].has_second_continuation if result_B else None,
            "xi0_collapse_passed": result_B.get('xi0_collapse_passed') if result_B else None,
        },
        "robustness": result_rob['robustness'],
    }
    with open(os.path.join(RESULTS_DIR, "exp2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {os.path.join(RESULTS_DIR, 'exp2_summary.json')}")
    print("\n" + "=" * 65)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()

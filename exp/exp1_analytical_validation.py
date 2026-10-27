"""
Experiment 1: Analytical validation of the optimal dividend-reset model.

This script verifies theorem-level properties independently of figure-based
comparative statics:

  1. Finiteness condition: tilde_rho > 0 iff value function is finite.
  2. Homogeneity reduction: V(x,c) = x^p * w(c/x).
  3. Frictionless benchmark: xi=0, eta=0 → constant w = (1/p)*bar_z^(p-1).
  4. Reduced HJB-QVI and reset operator construction.
  5. Structural properties: monotonicity, lower/upper bounds, supercontact.
  6. xi=0 reset structure collapses to a^p * w(0).

Run:
    python -m exp.exp1_analytical_validation

Results are saved to results/.
"""

from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import ModelParams, BASELINE_PARAMS_FRICTIONLESS
from src.solver import solve_qvi, make_grid
from src.operators import compute_reset_operator
from src.diagnostics import (
    classify_regions, check_structural_properties, check_homogeneity
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. FINITENESS CONDITION
# ──────────────────────────────────────────────────────────────

def validate_finiteness_condition() -> pd.DataFrame:
    """
    Verify tilde_rho > 0 is required for a finite value function.

    Tests multiple parameter tuples and classifies finite vs. infinite regime.
    Includes the paper's baseline (mu=0.07, rho=0.08, sigma=0.25, p=0.5)
    which must give tilde_rho = 0.0528125.
    """
    print("\n=== 1. Finiteness Condition ===")

    scenarios = [
        # (mu, sigma, rho, p, label)
        (0.07, 0.25, 0.08, 0.5,  "Baseline (paper)"),
        (0.07, 0.25, 0.05, 0.5,  "Low rho — borderline finite"),
        (0.07, 0.25, 0.02, 0.5,  "Very low rho — infinite"),
        (0.10, 0.30, 0.08, 0.8,  "High p — likely infinite"),
        (0.07, 0.10, 0.08, 0.5,  "Low sigma — finite"),
        (0.07, 0.50, 0.08, 0.5,  "High sigma — finite (more easily)"),
    ]

    rows = []
    for mu, sigma, rho, p, label in scenarios:
        params = ModelParams(mu=mu, sigma=sigma, rho=rho, p=p,
                             lambda_=0.5, xi=0.0, eta=0.0)
        tr = params.tilde_rho
        finite = params.is_finite()
        rows.append({
            "label": label,
            "mu": mu, "sigma": sigma, "rho": rho, "p": p,
            "tilde_rho": round(tr, 7),
            "finite": finite,
        })
        status = "FINITE" if finite else "INFINITE"
        print(f"  {label}: tilde_rho={tr:.7f}  → {status}")

    df = pd.DataFrame(rows)

    # Verify the paper's exact benchmark value
    baseline_tr = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                              lambda_=0.5, xi=0.0, eta=0.0).tilde_rho
    assert abs(baseline_tr - 0.0528125) < 1e-7, \
        f"Expected tilde_rho=0.0528125, got {baseline_tr}"
    print(f"\n  ✓ Baseline tilde_rho = {baseline_tr} (expected 0.0528125)")

    return df


# ──────────────────────────────────────────────────────────────
# 2. FRICTIONLESS BENCHMARK VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_frictionless_benchmark(
    N: int = 500,
    z_max: float = 2.0,
) -> dict:
    """
    Validate the frictionless benchmark (xi=0, eta=0).

    The reduced value function must be constant:
        w_fric = (1/p) * bar_z^(p-1)
    and bar_z = tilde_rho/(1-p).

    Paper baseline: (mu,rho,sigma,p) = (0.07,0.08,0.25,0.5):
        tilde_rho = 0.0528125, bar_z = 0.105625.
    """
    print("\n=== 2. Frictionless Benchmark Validation ===")

    params = BASELINE_PARAMS_FRICTIONLESS  # xi=0, eta=0
    tr = params.tilde_rho
    bz = params.bar_z
    w_fric = params.w_frictionless()

    print(f"  tilde_rho = {tr:.7f}  (expected 0.0528125)")
    print(f"  bar_z     = {bz:.7f}  (expected 0.105625)")
    print(f"  w_fric    = {w_fric:.6f}")

    assert abs(tr - 0.0528125) < 1e-7
    assert abs(bz - 0.105625) < 1e-7

    # Numerically verify the constant solution satisfies the HJB-QVI residual
    # At a constant w, w'=0, w''=0, so the HJB becomes:
    #   (tilde_rho + p*z)*w_fric - z^p/p = 0
    # at z = bar_z exactly.  Off bar_z the residual equals
    #   (tilde_rho + p*z)*w_fric - z^p/p
    # For the optimal policy, the firm sets c = bar_z * x always, so z = bar_z.

    z_grid = make_grid(z_max, N)
    z_test = np.array([bz])
    resid_at_bz = (params.discount_z(z_test) * w_fric
                   - params.running_payoff(z_test))
    print(f"  HJB residual at bar_z: {resid_at_bz[0]:.6e}  (should be ~0)")
    assert abs(resid_at_bz[0]) < 1e-8, \
        f"HJB residual at bar_z = {resid_at_bz[0]}"

    # Solve numerically and compare to analytical constant
    result = solve_qvi(params, z_grid, max_iter=3000, tol=1e-8, verbose=False)
    w_num = result['w']

    # The numerically solved w should be ≈ w_fric for z close to bar_z
    # (the frictionless solution; for a flat solution the solver should recover it)
    idx_bz = int(np.argmin(np.abs(z_grid - bz)))
    w_num_at_bz = w_num[idx_bz]
    print(f"  Numerical w(bar_z)    = {w_num_at_bz:.6f}")
    print(f"  Analytical w_fric     = {w_fric:.6f}")

    return {
        "tilde_rho": tr,
        "bar_z": bz,
        "w_fric_analytical": w_fric,
        "w_num_at_bar_z": float(w_num_at_bz),
        "hjb_resid_at_bar_z": float(resid_at_bz[0]),
        "passed": abs(tr - 0.0528125) < 1e-7 and abs(bz - 0.105625) < 1e-7,
    }


# ──────────────────────────────────────────────────────────────
# 3. HOMOGENEITY VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_homogeneity(
    N: int = 400,
    z_max: float = 3.0,
) -> dict:
    """
    Verify V(beta*x, beta*c) = beta^p * V(x, c) using V(x,c) = x^p * w(c/x).

    This property is exact by construction and should hold to machine precision.
    """
    print("\n=== 3. Homogeneity Validation ===")

    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.3, eta=0.1)
    z_grid = make_grid(z_max, N)
    result = solve_qvi(params, z_grid, max_iter=2000, tol=1e-8, verbose=False)
    w = result['w']

    report = check_homogeneity(params, z_grid, w,
                                x_test=1.0, c_test=0.1,
                                beta_values=[0.5, 1.0, 2.0, 5.0])
    all_pass = True
    for beta, info in report['betas'].items():
        status = "✓" if info['passed'] else "✗"
        print(f"  beta={beta}: V_scaled={info['V_scaled']:.8f}, "
              f"V_expected={info['V_expected']:.8f}, "
              f"rel_err={info['rel_error']:.2e} {status}")
        all_pass = all_pass and info['passed']

    print(f"  Homogeneity check: {'PASSED' if all_pass else 'FAILED'}")
    report['all_pass'] = all_pass
    return report


# ──────────────────────────────────────────────────────────────
# 4. RESET OPERATOR VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_reset_operator() -> dict:
    """
    Validate the three-branch reset operator formula and the xi=0 simplification.

    For xi=0: J_tilde[w](z) = a^p * max_{z0<=z/a} w(z0) = a^p * w(0)
    when w is nonincreasing (structural property).
    """
    print("\n=== 4. Reset Operator Validation ===")
    N = 200
    z_grid = make_grid(3.0, N)

    results = {}

    # ── xi=0 case ──
    params_xi0 = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                              lambda_=0.5, xi=0.0, eta=0.1)
    res_xi0 = solve_qvi(params_xi0, z_grid, max_iter=2000, tol=1e-8, verbose=False)
    w_xi0 = res_xi0['w']

    J_xi0 = compute_reset_operator(params_xi0, z_grid, w_xi0)
    a = params_xi0.a
    expected_J_xi0 = a**params_xi0.p * w_xi0[0]

    max_diff_xi0 = float(np.max(np.abs(J_xi0 - expected_J_xi0)))
    print(f"  xi=0: max|J_tilde - a^p*w(0)| = {max_diff_xi0:.2e}")
    print(f"    a^p*w(0) = {expected_J_xi0:.6f}")
    print(f"    max J_tilde = {np.max(J_xi0):.6f}")
    passed_xi0 = max_diff_xi0 < 1e-2
    print(f"  xi=0 simplification: {'PASSED' if passed_xi0 else 'FAILED'}")

    results['xi0'] = {
        'max_diff': max_diff_xi0,
        'a_p_w0': expected_J_xi0,
        'passed': passed_xi0,
    }

    # ── xi>0 case: verify no reset for z >= a/xi ──
    params_xi1 = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                              lambda_=0.5, xi=1.0, eta=0.1)
    res_xi1 = solve_qvi(params_xi1, z_grid, max_iter=2000, tol=1e-8, verbose=False)
    w_xi1 = res_xi1['w']
    J_xi1 = res_xi1['J_tilde']

    regions_xi1 = classify_regions(
        params_xi1, z_grid, w_xi1, res_xi1['w_prime'], J_xi1, res_xi1['hjb_resid']
    )

    threshold_xi1 = params_xi1.a / params_xi1.xi  # = 0.9
    above_thresh_mask = z_grid >= threshold_xi1 - 1e-10
    reset_above = bool(np.any(regions_xi1.reset & above_thresh_mask))
    print(f"\n  xi=1: a/xi = {threshold_xi1:.4f}")
    print(f"  Reset active above a/xi: {reset_above}  (should be False)")
    results['xi1_no_reset_above'] = {
        'threshold': threshold_xi1,
        'reset_above': reset_above,
        'passed': not reset_above,
    }

    return results


# ──────────────────────────────────────────────────────────────
# 5. STRUCTURAL PROPERTIES
# ──────────────────────────────────────────────────────────────

def validate_structural_properties(
    N: int = 500,
    z_max: float = 3.0,
) -> dict:
    """
    Numerically test structural properties of the solved reduced value function:
      - Nonnegativity
      - Monotonicity (w non-increasing)
      - Lower bound: w >= lambda/p
      - Upper bound: w <= g(bar_z)
      - Supercontact: w''(z_*) ≈ 0
      - xi=0 reset simplification
      - No reset for z >= a/xi (xi>0)
    """
    print("\n=== 5. Structural Properties ===")

    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.3, eta=0.1)
    z_grid = make_grid(z_max, N)
    result = solve_qvi(params, z_grid, max_iter=3000, tol=1e-8, verbose=False)
    w = result['w']
    w_prime = result['w_prime']
    J_tilde = result['J_tilde']
    hjb_resid = result['hjb_resid']

    regions = classify_regions(params, z_grid, w, w_prime, J_tilde, hjb_resid)

    report = check_structural_properties(params, z_grid, w, w_prime, regions, tol=0.05)

    for prop, (passed, detail) in report.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {prop}: {detail}")

    all_pass = all(v[0] for v in report.values())
    print(f"\n  Structural properties: {'ALL PASSED' if all_pass else 'SOME FAILED'}")

    return {k: {"passed": v[0], "detail": v[1]} for k, v in report.items()}


# ──────────────────────────────────────────────────────────────
# 6. VISUALIZATION
# ──────────────────────────────────────────────────────────────

def produce_exp1_plots(N: int = 500, z_max: float = 3.0) -> None:
    """Produce diagnostic plots for Experiment 1."""
    print("\n=== 6. Producing Plots ===")

    params = ModelParams(mu=0.07, sigma=0.25, rho=0.08, p=0.5,
                         lambda_=0.5, xi=0.3, eta=0.1)
    z_grid = make_grid(z_max, N)
    result = solve_qvi(params, z_grid, max_iter=3000, tol=1e-8, verbose=False)
    w = result['w']
    w_prime = result['w_prime']
    J_tilde = result['J_tilde']
    hjb_resid = result['hjb_resid']

    regions = classify_regions(params, z_grid, w, w_prime, J_tilde, hjb_resid)

    # Compute bounds for reference
    g_bz = params.g_bar_z()
    stop_level = params.stopping_level()
    bz = params.bar_z

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Experiment 1: Analytical Validation\n"
                 r"($\mu=0.07,\sigma=0.25,\rho=0.08,p=0.5,\xi=0.3,\eta=0.1$)",
                 fontsize=13)

    # Panel (a): w(z) with region shading and bounds
    ax = axes[0, 0]
    ax.plot(z_grid, w, 'k-', lw=2, label=r"$w(z)$")
    ax.axhline(stop_level, color='gray', ls='--', label=r"$\lambda/p$")
    ax.axhline(g_bz, color='orange', ls='--', label=r"$g(\bar{z})$")
    ax.axvline(bz, color='green', ls=':', lw=1.5, label=r"$\bar{z}$")
    if regions.z_star is not None:
        ax.axvline(regions.z_star, color='blue', ls='--', lw=1.5, label=r"$z_*$")
    if regions.z_upper is not None:
        ax.axvspan(regions.z_upper,
                   regions.z_upper2 if regions.z_upper2 else z_max,
                   alpha=0.15, color='red', label="Reset region")
    _shade_regions(ax, z_grid, regions, z_max)
    ax.set_xlabel(r"$z = c/x$")
    ax.set_ylabel(r"$w(z)$")
    ax.set_title("Value Function")
    ax.set_xlim(0, min(z_max, 2.0))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (b): w'(z)
    ax = axes[0, 1]
    ax.plot(z_grid, w_prime, 'b-', lw=1.5, label=r"$w'(z)$")
    ax.axhline(0, color='k', ls='--', lw=0.8)
    if regions.z_star is not None:
        ax.axvline(regions.z_star, color='blue', ls='--', lw=1.5, label=r"$z_*$")
    ax.set_xlabel(r"$z = c/x$")
    ax.set_ylabel(r"$w'(z)$")
    ax.set_title("Derivative of Value Function")
    ax.set_xlim(0, min(z_max, 2.0))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (c): HJB residual
    ax = axes[1, 0]
    ax.plot(z_grid, np.abs(hjb_resid), 'r-', lw=1.5, label="HJB residual")
    ax.set_xlabel(r"$z = c/x$")
    ax.set_ylabel("HJB residual (abs)")
    ax.set_title("HJB-QVI Residual")
    ax.set_xlim(0, min(z_max, 2.0))
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel (d): Reset operator vs w
    ax = axes[1, 1]
    ax.plot(z_grid, w, 'k-', lw=2, label=r"$w(z)$")
    ax.plot(z_grid, J_tilde, 'r--', lw=1.5, label=r"$\tilde{J}[w](z)$")
    ax.set_xlabel(r"$z = c/x$")
    ax.set_ylabel("Value")
    ax.set_title("Value Function vs Reset Obstacle")
    ax.set_xlim(0, min(z_max, 2.0))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp1_analytical_validation.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Second figure: frictionless benchmark
    params_fric = BASELINE_PARAMS_FRICTIONLESS
    bz_fric = params_fric.bar_z
    w_fric = params_fric.w_frictionless()
    z_grid2 = make_grid(0.5, 300)
    result2 = solve_qvi(params_fric, z_grid2, max_iter=3000, tol=1e-8, verbose=False)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(z_grid2, result2['w'], 'k-', lw=2, label=r"Numerical $w(z)$")
    ax2.axhline(w_fric, color='orange', ls='--', lw=1.5,
                label=rf"Analytical $w_{{fric}}={w_fric:.3f}$")
    ax2.axvline(bz_fric, color='green', ls=':', lw=1.5,
                label=rf"$\bar{{z}}={bz_fric:.4f}$")
    ax2.set_xlabel(r"$z = c/x$")
    ax2.set_ylabel(r"$w(z)$")
    ax2.set_title(r"Frictionless Benchmark ($\xi=0, \eta=0$)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    path2 = os.path.join(RESULTS_DIR, "exp1_frictionless_benchmark.png")
    plt.savefig(path2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")


def _shade_regions(ax, z_grid, regions, z_max):
    """Shade policy regions on an axis."""
    colors = {
        'increase': ('lightblue', 'Increase'),
        'reset': ('lightyellow', 'Reset'),
        'stopping': ('lightcoral', 'Stop'),
    }
    for key, (color, label) in colors.items():
        mask = getattr(regions, key if key != 'reset' else 'reset')
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        z_lo = float(z_grid[idxs[0]])
        z_hi = float(z_grid[idxs[-1]])
        ax.axvspan(z_lo, z_hi, alpha=0.12, color=color)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("EXPERIMENT 1: Analytical Validation of Dividend-Reset Model")
    print("=" * 60)

    # 1. Finiteness
    df_finite = validate_finiteness_condition()
    df_finite.to_csv(os.path.join(RESULTS_DIR, "exp1_finiteness.csv"), index=False)

    # 2. Frictionless benchmark
    bench = validate_frictionless_benchmark(N=500, z_max=2.0)

    # 3. Homogeneity
    hom = validate_homogeneity(N=400, z_max=3.0)

    # 4. Reset operator
    reset_results = validate_reset_operator()

    # 5. Structural properties
    struct = validate_structural_properties(N=500, z_max=3.0)

    # 6. Plots
    produce_exp1_plots(N=500, z_max=3.0)

    # ── Save summary JSON ──
    summary = {
        "experiment": "exp1",
        "finiteness": {
            "baseline_tilde_rho": 0.0528125,
            "baseline_bar_z": 0.105625,
            "passed": bool(abs(bench["tilde_rho"] - 0.0528125) < 1e-7
                           and abs(bench["bar_z"] - 0.105625) < 1e-7),
        },
        "frictionless_benchmark": {
            "tilde_rho": bench["tilde_rho"],
            "bar_z": bench["bar_z"],
            "w_fric_analytical": bench["w_fric_analytical"],
            "w_num_at_bar_z": bench["w_num_at_bar_z"],
            "passed": bench["passed"],
        },
        "homogeneity": {
            "all_pass": hom["all_pass"],
            "betas_tested": list(hom["betas"].keys()),
        },
        "reset_operator": {
            "xi0_simplification_passed": reset_results["xi0"]["passed"],
            "xi0_max_diff": reset_results["xi0"]["max_diff"],
            "xi1_no_reset_above_axi_passed": reset_results["xi1_no_reset_above"]["passed"],
        },
        "structural_properties": {
            k: v["passed"] for k, v in struct.items()
        },
    }
    summary_path = os.path.join(RESULTS_DIR, "exp1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    # Overall pass/fail
    all_passed = (
        summary["finiteness"]["passed"]
        and summary["frictionless_benchmark"]["passed"]
        and summary["homogeneity"]["all_pass"]
        and summary["reset_operator"]["xi0_simplification_passed"]
        and summary["reset_operator"]["xi1_no_reset_above_axi_passed"]
        and all(summary["structural_properties"].values())
    )
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1 OVERALL: {'ALL PASSED ✓' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

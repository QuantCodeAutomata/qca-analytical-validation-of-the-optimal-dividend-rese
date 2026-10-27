# AGENTS.md — Repository-Specific Knowledge

## Project Summary
Numerical implementation of the continuous-time optimal dividend-reset
control model. The value function V(x,c) = x^p * w(c/x) reduces via
homogeneity to a 1D HJB-QVI for w(z), solved numerically on a truncated
yield grid [0, z_max].

## Key Mathematical Objects

| Symbol        | Definition                                          |
|---------------|-----------------------------------------------------|
| `tilde_rho`   | `rho - mu*p + 0.5*sigma^2*p*(1-p)` (finiteness)   |
| `bar_z`       | `tilde_rho / (1-p)` (frictionless optimal yield)   |
| `w_fric`      | `(1/p) * bar_z^(p-1)` (frictionless constant)      |
| `g(z)`        | `z^p / (p*tilde_rho + p^2*z)` (upper bound)        |
| `J_tilde[w]`  | Nonlocal reset operator (3-branch for xi>0, 1 for xi=0) |

## Paper Baseline Benchmark Values
For `(mu, rho, sigma, p) = (0.07, 0.08, 0.25, 0.5)`:
- `tilde_rho = 0.0528125` (exact)
- `bar_z = 0.105625` (exact)
- `w_fric ≈ 6.15`

## Architecture

```
src/
  model.py       — ModelParams dataclass, all analytical quantities
  operators.py   — FD matrix (build_fd_matrix), nonlocal reset operator
  solver.py      — solve_qvi(), make_grid(), _enforce_nonincreasing()
  diagnostics.py — classify_regions(), check_structural_properties(), check_homogeneity()

exp/
  exp1_analytical_validation.py — Finiteness, homogeneity, frictionless, reset ops
  exp2_numerical_replication.py — Figure 2-4 style results

tests/
  test_model.py  — 25+ unit tests for ModelParams
  test_solver.py — 20+ unit tests for solver, operators, structural properties
```

## Solver Strategy
Policy iteration on truncated domain [0, z_max]:
1. Compute J_tilde[w^k] nonlocally (numpy interp, grid search)
2. Solve tridiagonal system A*w = rhs (scipy.sparse.linalg.spsolve)
3. Project: w = max(w_cont, J_tilde, lambda/p)
4. Enforce monotonicity via _enforce_nonincreasing (right-to-left max sweep)
5. Repeat until convergence

## Implementation Assumptions (not in paper excerpt)
- `lambda = 0.5` for Figure-style scenarios
- Sensitivity sweep ranges: xi∈[0,2], eta∈[0,0.4], p∈[0.2,0.8], sigma∈[0.1,0.5]
- `z_max = 4.0`, `N = 600` default grid
- xi=0 is handled as exact branch (not small-xi approximation)

## Test Commands
```bash
pytest tests/ -v                        # all tests
pytest tests/test_model.py -v           # model unit tests only
pytest tests/test_solver.py -v          # solver unit tests only

python -m exp.exp1_analytical_validation  # Experiment 1
python -m exp.exp2_numerical_replication  # Experiment 2
```

## Known Numerical Behavior
- With lambda=0.5, stopping region appears only at very high z
- xi=0 collapses reset target to lowest continuation level (z_0 = z_*)
- Higher eta widens inaction band; eta has larger impact than xi
- Solver convergence: typically 200-1000 iterations for standard params
- The reset operator loop (compute_reset_operator) is the bottleneck for large N

## Library Choices (Context7 confirmed)
- scipy.sparse, scipy.sparse.linalg: tridiagonal FD system
- numpy.interp: monotone 1D interpolation inside reset operator
- matplotlib: all visualization
- No quant-backtest library needed (pure PDE/control problem)

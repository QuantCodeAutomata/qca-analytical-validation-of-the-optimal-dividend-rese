# Optimal Dividend-Reset Model: Analytical Validation and Numerical Replication

A Python implementation of the continuous-time optimal dividend control model
with ratcheting, costly dividend resets, bankruptcy risk, and liquidation.

## Model Summary

The controlled equity process is:

```
dX_t = (mu*X_t - C_t)*dt + sigma*X_t*dW_t - dP_t
```

where `P_t` accumulates reset penalties `eta*X_{s-} + xi*Delta_C_s^-` at
each downward jump in the dividend rate `C_t`.

The value function `V(x, c)` is reduced to a 1D problem `w(z)` via the
homogeneity relation `V(x, c) = x^p * w(c/x)`, where `z = c/x` is the
*dividend yield*.

The reduced value function satisfies the HJB-QVI:

```
min{ (tilde_rho + p*z)*w - L^z w - z^p/p,
     -w',
     w - J_tilde[w],
     w - lambda/p } = 0
```

## Repository Structure

```
├── data/           # No market data required; see data/README.md
├── src/
│   ├── model.py         # ModelParams, derived constants, analytical quantities
│   ├── operators.py     # FD matrix, nonlocal reset operator
│   ├── solver.py        # HJB-QVI solver (policy iteration + obstacle projection)
│   └── diagnostics.py   # Region classification, structural property checks
├── exp/
│   ├── exp1_analytical_validation.py  # Experiment 1: Theorem-level validation
│   └── exp2_numerical_replication.py  # Experiment 2: Policy regions & comp. statics
├── tests/
│   ├── test_model.py    # Unit tests for model parameters
│   └── test_solver.py   # Unit tests for solver and operators
├── results/             # Generated plots, tables, and RESULTS.md
└── README.md
```

## Quick Start

```bash
# Install dependencies (all pre-installed in container)
pip install numpy scipy matplotlib pandas

# Run Experiment 1: Analytical Validation
python -m exp.exp1_analytical_validation

# Run Experiment 2: Numerical Replication
python -m exp.exp2_numerical_replication

# Run Tests
pytest tests/ -v
```

## Key Experiments

### Experiment 1 — Analytical Validation

Verifies:
- Finiteness condition: `tilde_rho = rho - mu*p + 0.5*sigma^2*p*(1-p) > 0`
- Paper benchmark: `tilde_rho = 0.0528125`, `bar_z = 0.105625`
- Homogeneity: `V(beta*x, beta*c) = beta^p * V(x, c)`
- Frictionless benchmark: constant `w_fric = (1/p)*bar_z^(p-1) ≈ 6.15`
- Structural properties: monotonicity, bounds, supercontact
- xi=0 reset simplification: `J_tilde = a^p * w(0)`

### Experiment 2 — Numerical Replication

Reproduces:
- Figure 2: `w(z)` and `w'(z)` for `eta=0.1` vs `eta≈0` (`xi=1` baseline)
- Figure 3: `xi=0`, `eta=0.1` regime
- Figure 4: Threshold sensitivity to `xi`, `eta`, `p`, `sigma`
- Qualitative findings: eta > xi impact, monotone continuation band, reset at distress

## Implementation Notes

- **Solver**: Monotone finite-difference policy iteration on truncated domain `[0, z_max]`
- **Reset operator**: Exact three-branch formula (xi>0) and simplified xi=0 branch
- **Grid**: Uniform grid; `N=600`, `z_max=4.0` by default
- **Lambda**: Set to `0.5` (implementation assumption; not in paper excerpt)
- **Sweep ranges**: Implementation assumptions; clearly labeled in output

## Parameter Constraints

| Parameter | Constraint    | Meaning                         |
|-----------|---------------|---------------------------------|
| `mu`      | `> 0`         | Equity drift                    |
| `sigma`   | `> 0`         | Equity volatility               |
| `rho`     | `> 0`         | Discount rate                   |
| `p`       | `(0, 1)`      | Power utility exponent          |
| `lambda_` | `>= 0`        | Liquidation utility weight      |
| `xi`      | `>= 0`        | Proportional reset friction     |
| `eta`     | `[0, 1)`      | Equity fraction lost at reset   |

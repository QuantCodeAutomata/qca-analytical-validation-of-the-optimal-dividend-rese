# Data Directory

This experiment requires **no empirical or market data**.

All inputs are model parameters and numerical state grids defined entirely in
source code. No data files are downloaded or generated here.

## Parameter Inputs

All experiments are driven by the parameter tuple:

```
(mu, sigma, rho, p, lambda, xi, eta)
```

Subject to constraints: `mu > 0`, `sigma > 0`, `rho > 0`, `0 < p < 1`,
`lambda >= 0`, `xi >= 0`, `0 <= eta < 1`.

## Benchmark Values

For `(mu, rho, sigma, p) = (0.07, 0.08, 0.25, 0.5)`:

| Quantity      | Value        |
|---------------|-------------|
| `tilde_rho`   | 0.0528125   |
| `bar_z`       | 0.105625    |
| `w_fric`      | ≈ 6.15      |

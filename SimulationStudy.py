import numpy as np
import pandas as pd

from RRAlgorithm import generate_data
from Statistics import compute_mse, compute_coverage
from SettingsImplementation import fit_np, fit_rr_kdr, fit_orr_kdr


def run_one_simulation(n, d, k, B_true, epsilon, cov_type="independent", seed=None):
    X, Y, _ = generate_data(n, d, B_true, cov_type=cov_type, seed=seed)

    # Setting 1: non-private
    B_np, cov_np, _ = fit_np(X, Y, k)
    mse_np = compute_mse(B_np, B_true)
    cp_np = compute_coverage(B_np, cov_np, B_true)

    # Setting 2: RR-k-D-R
    (B_rr, cov_rr, _), _, _ = fit_rr_kdr(X, Y, epsilon, k, seed=seed)
    mse_rr = compute_mse(B_rr, B_true)
    cp_rr = compute_coverage(B_rr, cov_rr, B_true)

    # Setting 3: ORR-k-D-R
    (B_orr, cov_orr, _), _, _ = fit_orr_kdr(X, Y, epsilon, k, gamma=0.8, seed=seed)
    mse_orr = compute_mse(B_orr, B_true)
    cp_orr = compute_coverage(B_orr, cov_orr, B_true)

    return {
        "epsilon": epsilon,
        "cov_type": cov_type,
        "mse_np": mse_np,
        "cp_np": cp_np,
        "mse_rrkdr": mse_rr,
        "cp_rrkdr": cp_rr,
        "mse_orrkdr": mse_orr,
        "cp_orrkdr": cp_orr
    }


def run_simulation_study(
    n=1000,
    d=4,
    k=3,
    B_true=None,
    eps_list=(0.1, 0.3, 0.5, 1.0),
    B=100,
    cov_types=("independent", "dependent")
):
    if B_true is None:
        B_true = np.array([
            [1.0, 0.5, -0.5, 0.8],
            [-0.8, 0.7, 0.3, -0.6]
        ])

    if B_true.shape != (k - 1, d):
        raise ValueError(f"B_true must have shape {(k - 1, d)}, got {B_true.shape}")

    results = []

    for cov_type in cov_types:
        for b in range(B):
            # Generate ONE dataset per replication
            X, Y, _ = generate_data(
                n=n,
                d=d,
                B=B_true,
                cov_type=cov_type,
                seed=b
            )

            # Non-private fit only once for this dataset
            B_np, cov_np, _ = fit_np(X, Y, k)
            mse_np = compute_mse(B_np, B_true)
            cp_np = compute_coverage(B_np, cov_np, B_true)

            # Apply private methods across all epsilons on the SAME dataset
            for eps in eps_list:
                # Setting 2: RR-k-D-R
                (B_rr, cov_rr, _), _, _ = fit_rr_kdr(X, Y, eps, k, seed=b)
                mse_rr = compute_mse(B_rr, B_true)
                cp_rr = compute_coverage(B_rr, cov_rr, B_true)

                # Setting 3: ORR-k-D-R
                (B_orr, cov_orr, _), _, _ = fit_orr_kdr(X, Y, eps, k, gamma=0.8, seed=b)
                mse_orr = compute_mse(B_orr, B_true)
                cp_orr = compute_coverage(B_orr, cov_orr, B_true)

                results.append({
                    "epsilon": eps,
                    "cov_type": cov_type,
                    "rep": b + 1,
                    "mse_np": mse_np,
                    "cp_np": cp_np,
                    "mse_rrkdr": mse_rr,
                    "cp_rrkdr": cp_rr,
                    "mse_orrkdr": mse_orr,
                    "cp_orrkdr": cp_orr
                })

    return pd.DataFrame(results)
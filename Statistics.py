import numpy as np
import matplotlib.pyplot as plt




def compute_mse(B_hat, B_true):
    """
    Mean squared error across all coefficients.
    """
    return np.mean((B_hat - B_true) ** 2)


def compute_coverage(B_hat, cov, B_true, alpha=0.05):
    """
    Marginal Wald-type coverage for all coefficients in vec(B).

    Returns the proportion of true coefficients covered by
    approximate 95% confidence intervals.
    """
    if cov is None:
        return np.nan

    beta_hat = B_hat.flatten()
    beta_true = B_true.flatten()

    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    z = 1.96

    lower = beta_hat - z * se
    upper = beta_hat + z * se

    covered = (beta_true >= lower) & (beta_true <= upper)
    return np.mean(covered)


def summarize_results(df):
    """
    Aggregate simulation results by covariance type and epsilon.
    """
    summary = df.groupby(["cov_type", "epsilon"]).agg(
        mse_np_mean=("mse_np", "mean"),
        cp_np_mean=("cp_np", "mean"),
        mse_rr_mean=("mse_rrkdr", "mean"),
        cp_rr_mean=("cp_rrkdr", "mean"),
        mse_orr_mean=("mse_orrkdr", "mean"),
        cp_orr_mean=("cp_orrkdr", "mean"),
    )
    return summary.reset_index()


def plot_results(df,k):
    summary = df.groupby(["cov_type", "epsilon"]).mean(numeric_only=True).reset_index()

    cov_types = summary["cov_type"].unique()

    for cov_type in cov_types:
        sub = summary[summary["cov_type"] == cov_type]

        # --- MSE plot ---
        plt.figure()
        plt.plot(sub["epsilon"], sub["mse_np"], marker='o', label="Non-private")
        plt.plot(sub["epsilon"], sub["mse_rrkdr"], marker='o', label=f'"RR-"{k}"D-R"')
        plt.plot(sub["epsilon"], sub["mse_orrkdr"], marker='o', label=f'"ORR-"{k}"D-R"')

        plt.xlabel("Epsilon")
        plt.ylabel("MSE")
        plt.title(f"MSE vs Epsilon ({cov_type})")
        plt.legend()
        plt.grid()
        plt.show()

        # --- Coverage plot ---
        plt.figure()
        plt.plot(sub["epsilon"], sub["cp_np"], marker='o', label="Non-private")
        plt.plot(sub["epsilon"], sub["cp_rrkdr"], marker='o', label=f'"RR-"{k}"D-R"')
        plt.plot(sub["epsilon"], sub["cp_orrkdr"], marker='o', label=f'"ORR-"{k}"D-R"')

        plt.xlabel("Epsilon")
        plt.ylabel("Coverage Probability")
        plt.title(f"Coverage vs Epsilon ({cov_type})")
        plt.legend()
        plt.grid()
        plt.show()

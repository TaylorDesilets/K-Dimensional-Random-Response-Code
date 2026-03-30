import numpy as np
from Statistics import summarize_results, plot_results
from SimulationStudy import run_simulation_study

def main():
    k = 3
    d = 4

    B_true = np.array([
        [1.0, 0.5, -0.5, 0.8],     # class 0 vs baseline
        [-0.8, 0.7, 0.3, -0.6]     # class 1 vs baseline
    ])  # shape (k-1, d)

    df = run_simulation_study(
        n=1000,
        d=d,
        k=k,
        B_true=B_true,
        eps_list=[0.1, 0.3, 0.5, 1.0],
        B=100,
        cov_types=("independent", "dependent")
    )

    print(df.head())
    print("\nSummary:")
    print(summarize_results(df))
    plot_results(df)


if __name__ == "__main__":
    main()
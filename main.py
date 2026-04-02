import numpy as np
from Statistics import summarize_results, plot_results
from SimulationStudy import run_simulation_study
from RealDatasetStudy import run_real_data_analysis



def run_simulation():
    k = 3
    d = 4

    B_true = np.array([
        [1.0, 0.5, -0.5, 0.8],
        [-0.8, 0.7, 0.3, -0.6]
    ])

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

def run_real():
    run_real_data_analysis(
        filepath="person.csv",
        sample_size=100000,
        epsilon=0.5,
        random_state=42
    )

    print("\nFinished real data analysis.")

def main(mode="simulation"):
    if mode == "simulation":
        run_simulation()

    elif mode == "real":
        run_real()

    else:
        raise ValueError("mode must be 'simulation' or 'real'")


if __name__ == "__main__":
    main(mode="simulation") 
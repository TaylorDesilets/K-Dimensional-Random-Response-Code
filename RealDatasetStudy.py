import numpy as np
import pandas as pd

from RRAlgorithm import make_rr_k_matrix, privatize_labels, fit_privatized_mlr


def collapse_severity(x):
    """
    Collapse 5 injury severity levels into 3 classes:
    0 = low / no injury
    1 = medium injury
    2 = fatal
    """
    if x in [0, 1]:
        return 0
    elif x in [2, 3]:
        return 1
    elif x == 4:
        return 2
    else:
        return np.nan


def load_person_data(filepath="person.csv"):
    """
    Load the person-level dataset.
    """
    return pd.read_csv(filepath, low_memory=False)


def prepare_real_data(filepath="person.csv", sample_size=100000, random_state=42):
    df = load_person_data(filepath)

    df["Y"] = df["INJ_SEV"].apply(collapse_severity)

    cols = ["AGE", "SEX", "PER_ALCH", "PER_DRUG"]

    df = df[cols + ["Y"]].dropna()
    df = df[(df["AGE"] >= 0) & (df["AGE"] <= 120)]

    df = df.sample(n=sample_size, random_state=random_state)

    # standardize age
    df["AGE"] = (df["AGE"] - df["AGE"].mean()) / df["AGE"].std()

    X = pd.get_dummies(df[cols], drop_first=True)
    X = X.astype(float).values

    Y = df["Y"].astype(int).values

    return X, Y, df


def print_class_counts(Y):
    """
    Print the number of observations in each response class.
    """
    print("Class counts:")
    print(pd.Series(Y).value_counts().sort_index())


def fit_private_model(X, Y, epsilon=0.5, seed=42):
    """
    Fit the privatized multinomial logistic regression model.

    Returns
    -------
    B_hat : ndarray
        Estimated coefficient matrix.
    cov : ndarray or None
        Estimated covariance matrix.
    result : OptimizeResult
        Optimization result.
    P : ndarray
        Randomized response matrix.
    Y_star : ndarray
        Privatized labels.
    """
    k = 3
    P = make_rr_k_matrix(k, epsilon)
    Y_star = privatize_labels(Y, P, seed=seed)
    B_hat, cov, result = fit_privatized_mlr(X, Y_star, P)

    return B_hat, cov, result, P, Y_star


def fit_nonprivate_model(X, Y):
    """
    Fit the non-private multinomial logistic regression model.
    """
    k = 3
    P_identity = np.eye(k)
    B_hat, cov, result = fit_privatized_mlr(X, Y, P_identity)

    return B_hat, cov, result


def run_real_data_analysis(filepath="person.csv", sample_size=10000, epsilon=0.5, random_state=42):
    """
    Run the full real-data analysis pipeline.

    Returns
    -------
    results : dict
        Dictionary containing data, fitted models, and outputs.
    """
    X, Y, df = prepare_real_data(
        filepath=filepath,
        sample_size=sample_size,
        random_state=random_state
    )

    print_class_counts(Y)

    B_hat, cov, result, P, Y_star = fit_private_model(
        X, Y, epsilon=epsilon, seed=random_state
    )

    B_np, cov_np, result_np = fit_nonprivate_model(X, Y)

    print("Private fit success:", result.success)
    print("Private message:", result.message)

    print("Non-private fit success:", result_np.success)
    print("Non-private message:", result_np.message)

    return {
        "X": X,
        "Y": Y,
        "df": df,
        "P": P,
        "Y_star": Y_star,
        "B_hat": B_hat,
        "cov": cov,
        "result": result,
        "B_np": B_np,
        "cov_np": cov_np,
        "result_np": result_np
    }
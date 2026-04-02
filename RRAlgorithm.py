
import numpy as np
from scipy.optimize import minimize


def multinomial_probs(X, B):
    """
    Multinomial logistic probabilities with the last class as baseline.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Covariate matrix.
    B : ndarray of shape (k-1, d)
        Regression coefficient matrix for the first k-1 classes.
        The last class is treated as the baseline.

    Returns
    -------
    probs : ndarray of shape (n, k)
        Class probabilities for each observation.
    """
    eta = X @ B.T  # (n, k-1)

    max_eta = np.max(eta, axis=1, keepdims=True)
    eta_stable = eta - max_eta # This is so our exp doesnt blow up for n -> infinity

    exp_eta = np.exp(eta_stable)

    denom = 1.0 + np.sum(exp_eta, axis=1, keepdims=True)
    probs_nonbaseline = exp_eta / denom
    probs_baseline = 1.0 / denom

    return np.hstack([probs_nonbaseline, probs_baseline])


def generate_data(n, d, B, cov_type="independent", seed=None):
    """
    Generate X and Y from a k-class multinomial logistic model.

    Parameters
    ----------
    n : int
        Sample size.
    d : int
        Number of covariates.
    B : ndarray of shape (k-1, d)
        True regression coefficient matrix.
    cov_type : str
        'independent' or 'dependent'
    seed : int or None
        Random seed.

    Returns
    -------
    X : ndarray of shape (n, d)
    Y : ndarray of shape (n,)
        Labels in {0, 1, ..., k-1}
    probs : ndarray of shape (n, k)
        True class probabilities.
    """
    rng = np.random.default_rng(seed)

    if cov_type == "independent":
        X = rng.normal(0, 1, size=(n, d))
    elif cov_type == "dependent":
        Sigma = 0.5 * np.ones((d, d)) + 0.5 * np.eye(d)
        X = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n)
    else:
        raise ValueError("cov_type must be 'independent' or 'dependent'")

    probs = multinomial_probs(X, B)
    k = probs.shape[1]
    Y = np.array([rng.choice(k, p=p) for p in probs])

    return X, Y, probs


def make_rr_k_matrix(k, epsilon):
    """
    Symmetric k-class randomized response matrix.

    P[a, b] = Pr(Y* = b | Y = a)

    Diagonal entries:
        exp(epsilon) / (exp(epsilon) + k - 1)

    Off-diagonal entries:
        1 / (exp(epsilon) + k - 1)
    """
    e_eps = np.exp(epsilon)
    p_diag = e_eps / (e_eps + k - 1)
    p_off = 1.0 / (e_eps + k - 1)

    P = np.full((k, k), p_off)
    np.fill_diagonal(P, p_diag)
    return P


def privatize_labels(Y, P, seed=None):
    """
    Privatize labels using transition matrix P.

    Parameters
    ----------
    Y : ndarray of shape (n,)
        True labels in {0, ..., k-1}
    P : ndarray of shape (k, k)
        Transition matrix.
    seed : int or None
        Random seed.

    Returns
    -------
    Y_star : ndarray of shape (n,)
        Privatized labels.
    """
    rng = np.random.default_rng(seed)
    Y_star = np.array([rng.choice(P.shape[1], p=P[y]) for y in Y])
    return Y_star


def observed_probs(X, B, P):
    """
    Observed privatized probabilities:
        q_ij = sum_k Pr(Y*=j | Y=k) Pr(Y=k | X_i)

    Parameters
    ----------
    X : ndarray of shape (n, d)
    B : ndarray of shape (k-1, d)
    P : ndarray of shape (k, k)

    Returns
    -------
    Q : ndarray of shape (n, k)
    """
    Pi = multinomial_probs(X, B)   # (n, k)
    Q = Pi @ P                     # (n, k)
    return Q


def neg_loglik(beta_vec, X, Y_star, P, k,lambda_reg=1e-1):
    """
    Negative log-likelihood for privatized multinomial logistic regression.

    Parameters
    ----------
    beta_vec : ndarray of shape ((k-1)*d,)
        Flattened coefficient matrix.
    X : ndarray of shape (n, d)
    Y_star : ndarray of shape (n,)
    P : ndarray of shape (k, k)
    k : int
        Number of classes.
    lambda_reg : float, default=1e-3
        L2 regularization strength used to stabilize estimation
        and discourage overly large coefficient values.


    Returns
    -------
    float
        Negative log-likelihood.
    """
    d = X.shape[1]
    B = beta_vec.reshape((k - 1, d))

    Q = observed_probs(X, B, P)
    Q = np.clip(Q, 1e-12, 1.0)

    ll = np.sum(np.log(Q[np.arange(len(Y_star)), Y_star]))

    # 🔑 L2 regularization
    penalty = lambda_reg * np.sum(B**2)

    return -ll + penalty


def fit_privatized_mlr(X, Y_star, P):
    """
    Estimate B by maximizing the privatized likelihood.

    Parameters
    ----------
    X : ndarray of shape (n, d)
    Y_star : ndarray of shape (n,)
    P : ndarray of shape (k, k)

    Returns
    -------
    B_hat : ndarray of shape (k-1, d)
    cov : ndarray or None
        Approximate covariance matrix for vec(B_hat).
    result : OptimizeResult
        scipy optimization result.
    """
    d = X.shape[1]
    k = P.shape[0]

    init = np.zeros((k - 1) * d)

    result = minimize(
        neg_loglik,
        init,
        args=(X, Y_star, P, k),
        method="BFGS"
    )

    B_hat = result.x.reshape((k - 1, d))

    cov = None
    if hasattr(result, "hess_inv"):
        try:
            cov = np.asarray(result.hess_inv)
        except Exception:
            cov = None

    return B_hat, cov, result



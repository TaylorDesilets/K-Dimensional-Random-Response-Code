
This repository contains the code used in the paper *“A Continuous Time Random Response Algorithm for Categorical Data”*, by Ce Zhang, Forough Fazeliasl, Linglong Kong, and Taylor Desilets.

See the paper for more info on the real dataset chosen
---

### Running the Code

To run the code, execute the following command in your terminal:

```bash
python main.py
```
I will go over Each File in this Repository and each of their purposes.
---
### 1. RRAlgorithm.py: Implementation of Privatisation Mechanism
This module simulates, privatizes, and estimates a multinomial logistic regression (MLR) model under a \(k\)-class randomized response mechanism.

### Main Functions

- **`make_rr_k_matrix(k, epsilon)`**  
  Constructs the $k \times k$ randomized response matrix $P\$.  
  - Diagonal: probability of reporting the true label  
  - Off-diagonal: probability of misreporting  
  - Smaller $\epsilon$ → stronger privacy

- **`multinomial_probs(X, B)`**  
  Computes class probabilities from a multinomial logistic regression model (last class as baseline).

- **`generate_data(n, d, B, cov_type, seed)`**  
  Simulates covariates $X$, true probabilities, and labels $Y$ from the MLR model.

- **`privatize_labels(Y, P)`**  
  Applies the randomized response mechanism to produce privatized labels $Y^*$.

- **`observed_probs(X, B, P)`**  
  Computes observed probabilities  
  $q_{ij} = \sum_k \Pr(Y^* = j \mid Y = k)\Pr(Y = k \mid X_i)$

- **`neg_loglik(beta_vec, X, Y_star, P, k, lambda_reg)`**  
  Negative log-likelihood for privatized data with L2 regularization.

- **`fit_privatized_mlr(X, Y_star, P)`**  
  Estimates model parameters using BFGS optimization. Returns $\hat{B} $, covariance (if available), and optimizer output.

### Procedure

1. Generate data with `generate_data`  
2. Create privacy mechanism with `make_rr_k_matrix`  
3. Privatize labels using `privatize_labels`  
4. Estimate parameters with `fit_privatized_mlr`

---

### 2. SettingsImplementation.py: Implementation of Benchmarking Method to test our algorithm. 

We compare three settings:
- Non-private estimation  (NP)
- Standard $k$-class randomized response (RR)  
- Our Optimal Method for $k$-class randomized response (ORR-k-D-R)  

### Main Functions

- **`fit_np(X, Y, k)`**  
  Setting 1: Non-private model.  
  Uses the identity matrix (no noise) and fits the model directly.

- **`fit_rr_kdr(X, Y, epsilon, k, seed=None)`**  
  Setting 2: Standard randomized response.  
  - Constructs $P$ using `make_rr_k_matrix`  
  - Privatizes labels $Y \rightarrow Y^*$ 
  - Fits the model using privatized data  

- **`fit_orr_kdr(X, Y, epsilon, k, gamma=0.5, seed=None)`**  
  Setting 3: Placeholder ORR-k-D-R method.  
  - Uses a perturbed RR matrix  
  - Projects rows onto the probability simplex  
  - Simulates a learned privatization mechanism  

### Helper Functions

- **`project_rows_to_simplex(A)`**  
  Ensures each row is positive and sums to 1.

- **`make_mock_orr_k_matrix(k, epsilon, gamma=0.5)`**  
  Creates a modified RR matrix by perturbing the standard RR mechanism and re-normalizing.

---

### 3. SimulationStudy.py: Runs Our 3 Settings on Simulated Data to Compare Performance

For each simulation run, the file:
- Generates multinomial data  
- Fits each method  
- Computes mean squared error (MSE)  
- Computes coverage probability (CP)  

The results are then stored in a pandas data frame for further analysis.

### Main Functions

- **`run_one_simulation(n, d, k, B_true, epsilon, cov_type="independent", seed=None)`**  
  Runs a single simulation replicate.  
  It:
  - generates data using `generate_data`  
  - fits the non-private, RR-k-D-R, and ORR-k-D-R models  
  - computes MSE and coverage probability for each method  
  - returns the results as a dictionary  

- **`run_simulation_study(n=1000, d=4, k=3, B_true=None, eps_list=(0.1, 0.3, 0.5, 1.0), B=100, cov_types=("independent", "dependent"))`**  
  Runs the full simulation study across:
  - multiple privacy levels $\epsilon$  
  - multiple covariance structures  
  - multiple simulation replicates  

  The output is returned as a pandas data frame, where each row corresponds to one simulation replicate.

---

### 4. RealDatasetStudy.py: Running our 3 Settings on a real Actuary Dataset to Compare Performance

We apply our privatized multinomial logistic regression to a real-world dataset to model predictors for Car Crashes. The goal is to model injury severity while comparing:

- Non-private estimation  
- Randomized response (RR) privatized estimation  

Injury severity is collapsed into three categories:
- 0: low / no injury  
- 1: medium injury  
- 2: fatal  

### Data Processing

The dataset is loaded from `person.csv` and processed as follows:
- Injury severity (`INJ_SEV`) is mapped into 3 classes ( 0-low injury, 1-medium injury, 2-fatal injury)
- A subset of predictors is selected:
  - age  
  - sex  
  - alcohol involvement  
  - drug involvement  
- Age is standardized  
- A random subsample is taken for computational efficiency  
- Predictors are one-hot encoded  

### Main Functions

- **`prepare_real_data(filepath, sample_size, random_state)`**  
  Loads and preprocesses the dataset, returning:
  - predictor matrix \(X\)  
  - response vector \(Y\)  
  - processed DataFrame  

- **`fit_private_model(X, Y, epsilon, seed)`**  
  Applies randomized response to the labels and fits the privatized multinomial logistic regression model.

- **`fit_nonprivate_model(X, Y)`**  
  Fits the model without any privacy mechanism (baseline comparison).

- **`run_real_data_analysis(filepath, sample_size, epsilon, random_state)`**  
  Runs the full pipeline:
  - prepares the data  
  - fits private and non-private models  
  - prints class counts and optimization results  
  - returns all outputs  

### 5. Statistics.py: Performance Measures and Summary Statistics Tools to Display Results

The main quantities of interest are:
- Mean squared error (MSE) of the estimated regression coefficients  
- Coverage probability (CP) of the confidence intervals  
- Summary tables and plots across privacy settings and covariance structures  

### Main Functions

- **`compute_mse(B_hat, B_true)`**  
  Computes the mean squared error between the estimated coefficient matrix and the true coefficient matrix.

- **`compute_coverage(B_hat, cov, B_true, alpha=0.05)`**  
  Computes the marginal Wald-type coverage probability for the regression coefficients.  
  It returns the proportion of true coefficients covered by approximate 95% confidence intervals.

- **`summarize_results(df)`**  
  Aggregates the simulation results by covariance structure and privacy level \( \epsilon \).  
  It returns the mean MSE and coverage probability for each method.

- **`plot_results(df)`**  
  Produces plots of:
  - MSE versus $\epsilon$  
  - Coverage probability versus $\epsilon$

  Separate plots are created for each covariance structure.

---

### 6. main.py: Main Entry Point for Running the Evaluation

NOTE: to run the entire loop, you only need to run this file, you need to input mode: "simulation" or "real" to run either the Simulated Data or the Real Dataset Implementation

### Main Functions

- **`run_simulation()`** 
  -Sets the number of classes $k$ and covariates $d$
  - Defines the true coefficient matrix $B_{\text{true}}$
  - Calls `run_simulation_study` to generate results across:
    - multiple privacy levels $\epsilon$
    - multiple covariance structures  
    - multiple simulation replicates  
  - Prints:
    - the first few rows of the results  
    - a summarized table of MSE and coverage  
  - Calls `plot_results` to visualize performance
 
- **`run_real()`**
  - Runs RealDatasetStudy functions and prints results
 
- **`main(mode=...)`**
  - Will either run the suimualtion or RealDataset Study based on wether you give it "real" or "simulation" as an argument for mode











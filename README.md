
This repository contains the code used in the paper *“A Continuous Time Random Response Algorithm for Categorical Data”*, by Ce Zhang, Forough Fazeliasl, Linglong Kong, and Taylor Desilets.
See the paper for more info on the real dataset chosen
---

### Running the Code

To run the code, Select if youd like to run the real dataset study or our simulated data study. 

execute the following command in your terminal to see our real dataset study:

```bash
python main.py real
```

execute the following command to view our simulated data study:
```bash
python main.py simulation
```

I will go over Each File in this Repository and each of their purposes.
---
### 1. RRAlgorithm.py: Implementation of Privatization Mechanism
This module simulates, privatizes, and estimates a multinomial logistic regression (MLR) model under a $k$-class randomized response mechanism. It also includes Fisher information inference for quantifying uncertainty.

### Main Functions

- **`make_rr_k_matrix(k, epsilon)`**  
  Constructs the $k \times k$ randomized response matrix $P$.  
  - Smaller $\epsilon$ → stronger privacy & worse utility 

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

- **`multinomial_prob_gradients_3class(X, B)`**  
  Computes gradients of multinomial probabilities with respect to model parameters for the 3-class case.

- **`fisher_information_privatized_3class(X, B, P)`**  
  Computes the empirical Fisher information matrix:
  $I_n(\beta, P) = \frac{1}{n} \sum_{i=1}^n \sum_{j=0}^{k-1}\frac{1}{q_{ij}} \, g_{ij} g_{ij}^\top$
  where $ g_{ij} = \nabla_\beta q_{ij} $.

- **`fisher_covariance_privatized_3class(X, B, P)`**  
  Computes the asymptotic covariance of $ \hat{B} $:
  $\text{Cov}(\hat{B}) \approx I_n^{-1} / n$

- **`fit_privatized_mlr(X, Y_star, P)`**  
  Estimates model parameters using BFGS optimization.  
  - Returns $ \hat{B} $, covariance, and optimizer output  
  - For the 3-class case, covariance is computed using the Fisher information  
  - Otherwise falls back to the optimizer’s inverse Hessian  

### Procedure

1. Generate data with `generate_data`  
2. Create privacy mechanism with `make_rr_k_matrix`  
3. Privatize labels using `privatize_labels`  
4. Estimate parameters with `fit_privatized_mlr` using Fisher information 
---

### 2. SettingsImplementation.py: Model Fitting Under Different Privacy Settings
This module implements three settings for multinomial logistic regression under varying levels of privacy, corresponding to the methods compared in the simulation and real data studies.

### Main Functions

- **`fit_np(X, Y, k)`**  
  Setting 1: Non-private estimation.  
  - Uses the identity transition matrix $P = I_k$  
  - Fits the standard multinomial logistic regression model without privacy  
  - Serves as a benchmark for comparison  

- **`fit_rr_kdr(X, Y, epsilon, k, seed)`**  
  Setting 2: Standard $k$-dimensional randomized response (RR-$k$-D-R).  
  - Constructs a symmetric randomized response matrix using $\epsilon$  
  - Privatizes labels $Y \rightarrow Y^*$  
  - Fits the privatized likelihood using $P_{\text{RR}}$  

- **`fit_orr_kdr(X, Y, epsilon, k, gamma, seed)`**  
  Setting 3: Optimized randomized response (ORR-$k$-D-R).  
  - Learns a transition matrix $P_{\text{ORR}}$ using a neural network  
  - The matrix is optimized to balance:
    - privacy (controlled by $\epsilon$)  
    - utility (controlled by $\gamma$)  
  - Privatizes labels using the learned mechanism  
  - Fits the privatized model using $P_{\text{ORR}}$  

- **`project_rows_to_simplex(A)`**  
  Projects each row of a matrix onto the probability simplex.  
  - Ensures positivity and row sums equal to 1  
  - Used to enforce valid transition matrices  

### Procedure

1. Choose a privacy setting (non-private, RR, or ORR)  
2. Construct or learn the transition matrix $P$  
3. Privatize labels using $Y^* \sim P(Y)$  
4. Estimate model parameters using `fit_privatized_mlr`  

---
### 3. NeuralNet.py: Learning the Optimal Privatization Mechanism
This module implements a neural network to learn an optimized transition matrix $P_{\text{ORR}}$ for privatized multinomial logistic regression. The learned mechanism balances privacy and utility through a weighted loss function.

### Main Components

- **`TransitionNet(k, hidden_dim)`**  
  Neural network model representing $f^\theta(Y, Y^*)$.  
  - Input: concatenation of one-hot encodings of true label $Y$ and privatized label $Y^*$  
  - Output: scalar score representing transition likelihood  
  - Architecture: fully connected network with ReLU activation and sigmoid output  

- **`build_transition_matrix(model, k)`**  
  Constructs the transition matrix $P$ by evaluating the neural network over all label pairs $(i, j)$.  
  - Applies softmax row-wise to ensure:
    - positivity  
    - rows sum to 1  
  - Output is a valid $k \times k$ transition matrix  

- **`privacy_loss_fn(beta, P)`**  
  Encourages privacy by penalizing large diagonal entries of $P$.  
  - Large diagonal values correspond to revealing the true label  
  - Also includes an $\ell_2$ penalty on $\beta$ for stability  

- **`utility_loss_fn(X, Y^*, B, P)`**  
  Measures utility using the negative log-likelihood of the privatized multinomial logistic regression model.  
  - Lower values indicate better predictive performance  
  - Based on:
    $Q = \Pi(X, B) P$

- **`learn_transition_matrix(X, Y^*, k, gamma, epochs, lr)`**  
  Learns an optimized transition matrix $P_{\text{ORR}}$ using gradient-based optimization.  
  - Jointly updates:
    - neural network parameters $\theta$  
    - regression coefficients $B$  
  - Minimizes the objective:
    $\mathcal{L}_{\text{total}} = (1 - \gamma)\,\mathcal{L}_{\text{privacy}} + \gamma\,\mathcal{L}_{\text{utility}}$

### Procedure

1. Initialize neural network $f^\theta$ and regression parameters $B$  
2. Construct transition matrix $P(\theta)$ using `build_transition_matrix`  
3. Compute:
   - privacy loss from $P$  
   - utility loss from the privatized likelihood  
4. Update $\theta$ and $B$ via gradient descent (Adam optimizer)  
5. Return the transition matrix $P_{\text{ORR}}$ that minimizes the total loss  
---
### 4. SimulationStudy.py: Runs Our 3 Settings on Simulated Data to Compare Performance

This module evaluates the non-private, RR-$k$-D-R, and ORR-$k$-D-R methods on simulated multinomial logistic regression data under different privacy levels and covariance structures.

For each simulation replicate, the file:
- Generates multinomial data  
- Fits each method  
- Computes mean squared error (MSE)  
- Computes coverage probability (CP)  

The results are then stored in a pandas data frame.

### Main Functions

- **`run_one_simulation(n, d, k, B_true, epsilon, cov_type="independent", seed=None)`**  
  Runs a single simulation replicate.  
  It:
  - generates data using `generate_data`  
  - fits the non-private, RR-$k$-D-R, and ORR-$k$-D-R models  
  - computes MSE and coverage probability for each method  
  - returns the results as a dictionary  

- **`run_simulation_study(n=1000, d=4, k=3, B_true=None, eps_list=(0.1, 0.3, 0.5, 1.0), B=100, cov_types=("independent", "dependent"))`**  
  Runs the full simulation study across:
  - multiple privacy levels $\epsilon$  
  - multiple covariance structures  
  - multiple simulation replicates  

  For each covariance structure and replicate:
  - one dataset is generated and held fixed across all values of $\epsilon$  
  - the non-private model is fit once on that dataset  
  - the RR-$k$-D-R and ORR-$k$-D-R methods are then fit across all privacy levels using the same data  

  This setup ensures that differences across $\epsilon$ reflect the effect of the privacy budget rather than variation in the simulated dataset.

  The output is returned as a pandas data frame.

---
### 5. RealDatasetStudy.py: Running Our 3 Settings on a Real Dataset to Compare Performance

This module applies the non-private, RR-$k$-D-R, and ORR-$k$-D-R methods to a real-world motor vehicle crash dataset. The goal is to model injury severity using privatized multinomial logistic regression and compare the three estimation settings.

Injury severity is collapsed into three categories:
- 0: low / no injury  
- 1: medium injury  
- 2: fatal  

### Data Processing

The dataset is loaded from `person.csv` and processed as follows:
- Injury severity (`INJ_SEV`) is mapped into 3 classes:
  - 0 = low / no injury  
  - 1 = medium injury  
  - 2 = fatal  
- A subset of predictors is selected:
  - age  
  - sex  
  - alcohol involvement  
  - drug involvement  
- Missing values are removed  
- Age is restricted to a valid range and standardized  
- A random subsample is taken for computational efficiency  
- Predictors are one-hot encoded  

### Main Functions

- **`collapse_severity(x)`**  
  Maps the original injury severity variable into the 3-category response used in the analysis.

- **`load_person_data(filepath="person.csv")`**  
  Loads the raw dataset from `person.csv`.

- **`prepare_real_data(filepath, sample_size, random_state)`**  
  Loads and preprocesses the dataset, returning:
  - predictor matrix $X$
  - response vector $Y$
  - processed DataFrame  

- **`print_class_counts(Y)`**  
  Prints the class distribution of the response variable to assess imbalance in the real dataset.

- **`fit_nonprivate_model(X, Y)`**  
  Fits the multinomial logistic regression model without any privacy mechanism.  
  - Uses the identity transition matrix  
  - Serves as the non-private baseline  

- **`fit_private_rr_model(X, Y, epsilon, seed)`**  
  Applies the standard randomized response mechanism to the labels and fits the privatized multinomial logistic regression model.

- **`fit_private_orr_model(X, Y, gamma, seed)`**  
  Learns a transition matrix using the neural network–based ORR mechanism, privatizes the labels using the learned matrix, and fits the privatized multinomial logistic regression model.

- **`run_real_data_analysis(filepath, sample_size, epsilon, gamma, random_state)`**  
  Runs the full real-data pipeline:
  - prepares the dataset  
  - prints class counts  
  - fits the non-private, RR, and ORR models  
  - prints optimizer success messages  
  - prints the RR and learned ORR transition matrices  
  - returns all outputs in a dictionary  
---
### 6. Statistics.py: Performance Measures and Summary Statistics Tools to Display Results

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
  Aggregates the simulation results by covariance structure and privacy level $\epsilon$.  
  It returns the mean MSE and coverage probability for each method.

- **`plot_results(df)`**  
  Produces plots of:
  - MSE versus $\epsilon$  
  - Coverage probability versus $\epsilon$

---

### 7. main.py: Running the Evaluation

This file serves as the main entry point for running both the simulation study and the real dataset analysis.

To run the full pipeline, execute this file and specify the mode:
- `"simulation"` → runs the simulation study  
- `"real"` → runs the real dataset analysis  

If no mode is provided, the simulation study runs by default.

### Main Functions

- **`run_simulation()`**  
  - Sets the number of classes $k$ and covariates $d$  
  - Defines the true coefficient matrix $B_{\text{true}}$  
  - Calls `run_simulation_study` to generate results across:
    - multiple privacy levels $\epsilon$  
    - multiple covariance structures  
    - multiple simulation replicates  
  - Prints:
    - the first few rows of the results  
    - a summarized table of MSE and coverage probability  
  - Calls `plot_results` to visualize performance across methods  

- **`run_real()`**  
  - Calls `run_real_data_analysis` to run the full real dataset pipeline  
  - Uses specified values of $\epsilon$ and $\gamma$  
  - Prints model fit results and learned transition matrices  

- **`main()`**  
  - Controls which is executed:
    - `"simulation"` → runs `run_simulation()`  
    - `"real"` → runs `run_real()`   










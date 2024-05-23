import numpy as np
from numpy.linalg import svd, norm, solve
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from SCCA_main_solutionV4 import SCCA
from scca_init0 import init0

# Assuming all SCCA related functions (SCCA, SCCA_solution, find_Omega, init0, init1) have been defined here

# Set the dimensions
p = 200
q = 200
n = 500

# Generate sigma.X
sigma_X = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        sigma_X[i, j] = 0.5 ** abs(i - j)

# sigma.Y is the same as sigma.X in this example
sigma_Y = sigma_X.copy()

# Set up theta and eta
theta = np.zeros((q, 1))
eta = np.zeros((p, 1))
id_indices = np.arange(8)  # Adjust for zero-based indexing
theta[id_indices, 0] = 1
theta[:, 0] /= np.sqrt(theta[:, 0].T @ sigma_Y @ theta[:, 0])

eta = theta.copy()

# Construct sigma.YX
sigma_YX = 0.9 * sigma_Y @ theta @ eta.T @ sigma_X


# Combine to form sigma
sigma = np.zeros((p + q, p + q))
sigma[:q, :q] = sigma_Y
sigma[q:(p + q), q:(p + q)] = sigma_X
sigma[:q, q:(p + q)] = sigma_YX
sigma[q:(p + q), :q] = sigma_YX.T

# Compute the square root of sigma
U, s, Vh = np.linalg.svd(sigma)
sigma_sqrt = U @ np.diag(np.sqrt(s)) @ Vh

# Set the random seed and generate the data
np.random.seed(314159)
Z = np.random.randn(2 * n * (p + q)).reshape(2 * n, p + q) @ sigma_sqrt


# Function to calculate row correlations and average them
def average_row_correlations(Z):
    n = Z.shape[0]
    avg_cor = np.zeros(n)
    
    for i in range(n):
        # Calculate correlation of i-th row with all other rows
        row_cor = np.array([np.corrcoef(Z[i, :], Z[j, :])[0, 1] for j in range(n) if i != j])
        avg_cor[i] = np.mean(row_cor)
    
    return avg_cor

# Calculate average row correlations
avg_cor = average_row_correlations(Z)

# Print the average row correlations
print(avg_cor)

Y = Z[:, :q]
X = Z[:, q:(p + q)]

# Split the data into training and tuning sets
id_train = np.random.choice(2 * n, n, replace=False)
X_train = X[id_train, :]
Y_train = Y[id_train, :]
X_tune = X[~id_train, :]
Y_tune = Y[~id_train, :]

# Estimate the covariance matrices
sigma_Y_hat = np.cov(Y_train, rowvar=False)
sigma_X_hat = np.cov(X_train, rowvar=False)
sigma_YX_hat = np.cov(Y_train.T, X_train.T)[:q, q:]

# Center the training data
X_train -= X_train.mean(axis=0)
Y_train -= Y_train.mean(axis=0)

# Define lambda values and initial options
lambdas = np.arange(1, 21) / 100
init_options = ["sparse", "svd"]

# Initialize rho.lambda matrix
rho_lambda = np.zeros((len(lambdas), len(init_options)))

for i_init, init_opt in enumerate(init_options):
    #print(f"Starting initialization method: {init_opt}")
    init_results = init0(sigma_YX_hat, sigma_X_hat, sigma_Y_hat, init_opt, 1, n)
    alpha_init = init_results["alpha_init"]
    beta_init = init_results["beta_init"]
    for i_lambda, lambda_val in enumerate(lambdas):
        lambda_alpha = lambda_val
        lambda_beta = lambda_val
        #print(f"Running SCCA with lambda {lambda_val:.2f} (Index {i_lambda+1}/{len(lambdas)}) using init method '{init_opt}'")
        result = SCCA(X_train, Y_train, niter=100, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta, npairs=1, alpha_init=alpha_init, beta_init=beta_init, init_method=init_opt, standardize=True)
        alpha_hat = result['alpha']
        beta_hat = result['beta']
        rho_lambda[i_lambda, i_init] = pearsonr(X_tune @ beta_hat[:, 0], Y_tune @ alpha_hat[:, 0])[0] ** 2
        if rho_lambda[i_lambda, i_init] < 1e-6:
            break

# Find the best lambda and init method
id_best = np.unravel_index(rho_lambda.argmax(), rho_lambda.shape)
best_lambda = lambdas[id_best[0]]
best_init = init_options[id_best[1]]

# Perform SCCA with the best lambda and init method
best_result = SCCA(X_train, Y_train, [best_lambda], [best_lambda], niter=100, npairs=1, 
                   init_method=best_init)

alpha_hat = best_result['alpha']
beta_hat = best_result['beta']

# Calculate norm differences
alpha_norm_diff = norm(alpha_hat @ solve(alpha_hat.T @ alpha_hat, alpha_hat.T) - theta @ solve(theta.T @ theta, theta.T))
beta_norm_diff = norm(beta_hat @ solve(beta_hat.T @ beta_hat, beta_hat.T) - eta @ solve(eta.T @ eta, eta.T))

alpha_norm_diff = norm(alpha_hat @ np.linalg.pinv(alpha_hat.T @ alpha_hat) @ alpha_hat.T - theta @ np.linalg.pinv(theta.T @ theta) @ theta.T, 'fro')
beta_norm_diff = norm(beta_hat @ np.linalg.pinv(beta_hat.T @ beta_hat) @ beta_hat.T - eta @ np.linalg.pinv(eta.T @ eta) @ eta.T, 'fro')

print(f"Alpha Norm Difference: {alpha_norm_diff}")
print(f"Beta Norm Difference: {beta_norm_diff}")

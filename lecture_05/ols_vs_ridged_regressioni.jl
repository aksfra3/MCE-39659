#---OLS vs. Riged Regression---#
using LinearAlgebra, Random, Statistics, Plots

# 1. Setup: 20 observations, 10 features (Small data = High Variance)
Random.seed!(123)
n, p = 20, 10
X = randn(n, p)

# Create Multicollinearity: Make column 2 almost identical to column 1
X[:, 2] = X[:, 1] + randn(n) * 0.001 

true_beta = [5.0, -4.0, 3.0, randn(7)...] # True weights
y = X * true_beta + randn(n) * 0.5        # Targets with noise

# 2. Define Solvers
# OLS: (X'X)⁻¹ X'y
ols_beta = (X' * X) \ (X' * y)

# Ridge: (X'X + λI)⁻¹ X'y (using λ = 2.0 as an example)
λ = 2.0
ridge_beta = (X' * X + λ * I) \ (X' * y)

# 3. Compare Results
println("True β[1] and β[2]: ", true_beta[1:2])
println("OLS  β[1] and β[2]: ", round.(ols_beta[1:2], digits=2))
println("Ridge β[1] and β[2]: ", round.(ridge_beta[1:2], digits=2))
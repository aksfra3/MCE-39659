#---Example for finding best lambda for riged regression---#

#Ridge regression: Example Julia
using LinearAlgebra, Random, Statistics

Random.seed!(42)
n, p = 100, 50
X = randn(n, p)
true_beta = randn(p)
y = X * true_beta + randn(n) * 0.5 # Add some noise

#Ridge Function ( β = (X'X + λI)⁻¹ X'y)
function ridge_reg(X, y, λ)
    p = size(X, 2)
    return (X' * X + λ * I) \ (X' * y)              #closed form solution
end



##Finding best lambda
train_idx = 1:80
val_idx = 81:100

X_train, y_train = X[train_idx, :], y[train_idx]
X_val, y_val = X[val_idx, :], y[val_idx]

lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]          #range of lambdas to test
errors = []

for λ in lambdas

    β_hat = ridge_reg(X_train, y_train, λ)

    predictions = X_val * β_hat

    mse = mean((y_val .- predictions).^2)
    push!(errors, mse)
    
    println("λ: $λ \t Validation MSE: $(round(mse, digits=4))")
end

best_λ = lambdas[argmin(errors)]
println("\n--- Best λ is $best_λ ---")




## CASE STUDY: LECTURE 05 ##

import Pkg; Pkg.add("CSV")
import Pkg; Pkg.add("DefaultApplication")

using CSV
using DataFrames

# Load the CSV file into a DataFrame
df = CSV.read("/Users/akselfraser/Documents/german_data.csv", DataFrame)

# View the first few rows
first(df, 5)


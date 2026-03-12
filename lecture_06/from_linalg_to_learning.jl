

## 7/30 --------------- From Linear Algebra to estimation and learning ------------------------------------------------------------------------------------------ ##

#- Key objective: Choose parameters (theta) to minimize the loss function (L(theta)) which measures how well the model fits the data.

#- Residuals: The difference between predicted and actual values
#- Loss: A measure of how well the model's predictions match the actual data
#- Gradient: The direction and rate of change of the loss function
#- Jacobian: A matrix of all first-order partial derivatives of a vector-valued function
#- Hessian: A matrix of all second-order partial derivatives of a scalar-valued function

predict(X, θ) = X * θ

residuals(X, y, θ) = y .- predict(X, θ)             # dot before operator (eg:  .-) tells language to perform the opertaion element wise

function loss_function(X, y, θ) #scalar loss function 
    r = residuals(X, y, θ)
    return 0.5 * dot(r, r) # Equivalent to 0.5 * r' * r
end

function compute_gradient(X, y, θ)
    r = residuals(X, y, θ)
    return -X' * r
end

function update_theta(θ, gradient, learning_rate=0.01)
    return θ .- learning_rate .* gradient
end

## 8/30 --------------------------------------------------------------------------------------------------------- ##

using Statistics, LinearAlgebra, Plots, DataFrames

# --- 1. CORE NUMERICAL OBJECTS (Roadmap: "What we estimate") ---

# Prediction: r_hat = X * θ
predict(X, θ) = X * θ

# Elastic Net Loss: MSE + L1 (Lasso) + L2 (Ridge)
# Formula: 0.5 * ||y - Xθ||² + λ * [α||θ||₁ + (1-α)0.5||θ||₂²]
function loss_function(X, y, θ, λ, α)
    r = y - predict(X, θ)
    mse = 0.5 * dot(r, r)
    l1 = λ * α * norm(θ, 1)
    l2 = λ * (1 - α) * 0.5 * dot(θ, θ)
    return mse + l1 + l2
end

# Gradient of MSE + Ridge part (Lasso handled by Soft-Thresholding)
compute_gradient(X, y, θ, λ, α) = -X' * (y - X*θ) + λ * (1 - α) * θ

# Soft-Thresholding Operator for the Lasso (L1) part
function soft_threshold(z, threshold)
    return sign(z) * max(0, abs(z) - threshold)
end

# Safe Standardization: Mean 0, Std 1 (Roadmap: "Scaling checks")
function safe_standardize(X)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)
    σ[σ .== 0] .= 1.0 # Avoid Division by Zero (NaN Fix)
    return (X .- μ) ./ σ
end

# --- 2. THE STRATEGY SIMULATION (Roadmap: "How we validate") ---

# Simulation Setup
N_stocks = 100
T_months = 60
N_features = 3 # e.g., Size, Momentum, Value

# Generate dummy features (X) and target returns (y)
X_raw = randn(N_stocks * T_months, N_features)
# True relationship: return = 0.1*size + 0.2*mom + noise
y = 0.1 * X_raw[:, 1] + 0.2 * X_raw[:, 2] + 0.5 * randn(N_stocks * T_months)

# Initialize Strategy Variables
λ = 0.01           # Penalty strength
α = 0.5           # Elastic Net mix (0.5 = 50% Lasso, 50% Ridge)
learning_rate = 0.0001
θ = zeros(N_features)
losses = Float64[]

# --- 3. ESTIMATION LOOP (Numerical Mindset: "Clear stopping rules") ---

X_std = safe_standardize(X_raw)

for iter in 1:1000
    # 1. Update Gradient (MSE + Ridge)
    grad = compute_gradient(X_std, y, θ, λ, α)
    θ .-= learning_rate .* grad
    
    # 2. Proximal Step (Lasso Soft-Thresholding)
    for j in 1:length(θ)
        θ[j] = soft_threshold(θ[j], learning_rate * λ * α)
    end
    
    # 3. Track Diagnostics
    push!(losses, loss_function(X_std, y, θ, λ, α))
end

# --- 4. TRADING & PERFORMANCE (Roadmap: "How we judge") ---

# Calculate predicted returns
r_hat = predict(X_std, θ)

# Simple Long-Short Logic: 
# Long top 10% (+1), Short bottom 10% (-1)
function get_ls_weights(preds, q=0.1)
    threshold_high = quantile(preds, 1-q)
    threshold_low = quantile(preds, q)
    weights = zeros(length(preds))
    weights[preds .>= threshold_high] .= 1.0
    weights[preds .<= threshold_low] .= -1.0
    return weights ./ sum(abs.(weights)) # Normalize
end

weights = get_ls_weights(r_hat)
strategy_return = dot(weights, y)

# --- 5. VISUALIZATION ---

p1 = plot(losses, title="Convergence Diagnostic", xlabel="Iteration", ylabel="Loss", lw=2, color=:blue)
p2 = bar(θ, title="Feature Importance (θ)", xticks=(1:N_features, ["Size", "Mom", "Value"]), color=:grey)

println("Final Parameters (θ): ", round.(θ, digits=4))
println("Example Strategy Return: ", round(strategy_return, digits=4))

plot(p1, p2, layout=(2,1), size=(600, 600))



# Improve the plotting aesthetics
p1 = plot(losses, 
    yscale=:log10, # Log scale reveals if the model is actually 'stuck'
    title="Loss Convergence (Log Scale)", 
    xlabel="Iteration", ylabel="Total Loss",
    lw=2, color=:crimson, grid=true)

p2 = bar(["Size", "Mom", "Value"], θ, 
    title="Model Coefficients (Selection)",
    ylabel="Weight (θ)",
    color=ifelse.(θ .== 0, :red, :green), # Red if Lasso killed the feature
    legend=false)

# Add a horizontal line at 0 for clarity
hline!(p2, [0], color=:black, lw=1)

plot(p1, p2, layout=(2,1), size=(700, 800))
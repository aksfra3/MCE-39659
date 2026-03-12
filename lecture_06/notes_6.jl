#PACKAGES
import Pkg; Pkg.add("YFinance")
import Pkg; Pkg.add("PrettyTables")
import Pkg; Pkg.add("Zygote")

using PrettyTables
using YFinance
using DataFrames
using Statistics
using Zygote
using Plots
using Dates
using LinearAlgebra


############################################################ ----- CROSS SECTIONAL STRATEGIES ----- ############################################################

#titlon


## 4/30 --------------- Cross sectional distribution of returns --------------------------------------------------------------------------- ##

#- This shows the returns of many different stocks (the U.S. universe) at a single point in time (one specific month).
#- Cross-sectional refers to comparing different entities at the same moment
#- The Goal: To find signals (Z_i,t) that tell you what to hold right now.

tickers = tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "MA", "GS", "JNJ", "UNH", "PG", "PFE", "WMT", "HD", "COST", "CAT", "XOM", "CVX", "AMD"]

data = [get_prices(t, startdt="2020-01-01", enddt="2023-12-31", interval="1mo") for t in tickers]

#Organize into DataFrame
df_list = []
for (i, t) in enumerate(tickers)
    temp_df = DataFrame(data[i])       #convert to dataframe
    temp_df[!, :ticker] .= t
    push!(df_list, temp_df)
end
df = vcat(df_list...)

#Calculate monthly returns (%)
sort!(df, [:ticker, :timestamp])
df[!, :ret] = combine(groupby(df, :ticker), :adjclose => (x -> [nothing; diff(x) ./ x[1:end-1] .* 100]) => :ret).ret
dropmissing!(df)

#SELECT A CROSS-SECTION (A specific month; March 2020)
target_month = Date(2022, 3, 1)
cross_section = filter(row -> Date(row.timestamp) == target_month, df)

#Plot the Distribution
histogram(cross_section.ret, 
    bins = 15, 
    title = "Cross-Sectional Return Distribution ($(target_month))",
    xlabel = "Monthly Returns (%)", 
    ylabel = "Count of Stocks",
    color = :darkgrey, 
    legend = false)




## 5/30 --------------- The time-series distribution of returns --------------------------------------------------------------------------- ##

#- This shows the returns of a single asset (the entire U.S. Equity Market) tracked over many points in time (multiple months).
#- Predict when to take risk (e.g., market timing or holding more cash vs. market).

ticker = "SPY"
market_data = get_prices(ticker, startdt="2010-01-01", enddt="2025-12-31", interval="1mo")

df_market = DataFrame(market_data)      #convert to dataframe

#Calculate monthly returns
# We sort to ensure time ordering, which is vital for time-series [cite: 227]
sort!(df_market, :timestamp)

# Calculate returns: (Price_t / Price_{t-1}) - 1
# We use [missing; ...] to keep the vector length equal to the DataFrame rows
returns_raw = [missing; diff(df_market.adjclose) ./ df_market.adjclose[1:end-1] .* 100]
df_market[!, :ret] = returns_raw

df_market_clean = dropmissing(df_market)            #drop first row

#plot
histogram(df_market_clean.ret, 
    bins = 40, 
    title = "Time-series Return Distribution (SPY)",
    xlabel = "Monthly Returns (%)", 
    ylabel = "Count (Months)",
    color = :black, 
    legend = false,
    alpha = 0.7)




## 6/30 --------------- Cross-section vs. Time-series ------------------------------------------------------------------------------------------ ##
#- Time series: Market timing (where will asset move within distribution)
#- Cross sectional: Asset allocation (which specific stocks will land in the "winning" right-hand tail of that distribution versus the "losing" left-hand tail)




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



## 8/30 --------------- How these objects connect ------------------------------------------------------------------------------------------ ##


# 1. Setup Numerical Example (3 stocks, 2 features)
# Target returns (y) and features (X)
X = [1.0 0.5; 
     1.0 1.5; 
     1.0 2.5]
y = [3.5, 6.5, 9.5]

# Initial guess for theta (the model parameters)
θ = [0.1, 0.1]

# 2. Define the core objects from Lecture Slide 8
function calculate_objects(X, y, θ)
    # Residuals: r(θ) = Xθ - y
    r = X * θ - y
    
    # Loss: F(θ) = 1/2 * r' * r
    loss = 0.5 * dot(r, r)
    
    # Jacobian: J(θ) = ∂r/∂θ' (For linear models, J = X)
    J = X
    
    # Gradient: ∇F(θ) = J' * r
    grad = J' * r
    
    # Hessian (Gauss-Newton): H(θ) ≈ J' * J
    H = J' * J
    
    return r, loss, grad, H
end

# 3. Execution and Connection
r, loss, grad, H = calculate_objects(X, y, θ)

# Newton Step: H * Δθ = -∇F(θ) ----> Iterative method to find the minimum of the loss function
Δθ = H \ -grad

println("Initial Loss: ", loss)
println("Gradient: ", grad)
println("Newton Update (Δθ): ", Δθ)
println("Updated θ: ", θ + Δθ)


## 9/30 --------------- Automtic differentiation (AD) ------------------------------------------------------------------------------------------ ##

#-Automatic Differentiation (AD) is not a numerical approximation like finite differences are. It is a way to calculate the exact derivative of a function defined by computer code

X = [1.0 0.2; 1.0 0.4; 1.0 0.6] # Features (z_i,t)
y = [0.05, 0.07, 0.09]          # Realized returns (r_i,t+1)
θ = [0.1, 0.1]                  # Initial parameters

# 2. Define the Loss Function F(θ) as code ( F(θ) = 1/2 * r(θ)' * r(θ))
function loss_function(θ)
    r = X * θ - y             # Residuals [cite: 70, 71]
    return 0.5 * dot(r, r)    # Scalar loss [cite: 74]
end

# 3. Use AD to get the Gradient ∇F(θ) 
# We don't need to manually derive J' * r; Zygote does it for us!
grad_ad = gradient(loss_function, θ)[1]

# 4. Compare with the Manual Gauss-Newton View (Slide 8)
r = X * θ - y
J = X                         # Jacobian [cite: 74]
grad_manual = J' * r          # Manual Gradient [cite: 75]

println("AD Gradient:     ", grad_ad)
println("Manual Gradient: ", grad_manual)



## 20/30 --------------- Elastic Net Computation: Coordinate Descent ------------------------------------------------------------------------------------------ ##


##----CASE STUDY: LECTURE 05----##
#Task: Build replicating portfolio for TTF

#Importing Data and Packages
    #import Pkg; Pkg.add("CSV")
    #import Pkg; Pkg.add("DefaultApplication")

using CSV
using DataFrames
using Plots

# Load the CSV file into a DataFrame
df = CSV.read("/Users/akselfraser/Documents/german_data.csv", DataFrame)
first(df, 10)


##TTF = Natural Gas
#fm1 = Power
#EUA = Emissions


#------Replicating using OLS------#
n = size(df, 1)
y = df.TTF              #actual y
X = [ones(n) df.fm1 df.EUA]

theta_ols = X \ y      #replicating coefficients (thetas/betas) ( \ = Assumes Least Squares solutino)
theta_ols_1 = (X' * X) \ (X' * y) #using closed form solution

println("Estimated Coefficients (theta_ols):")
println("Alpha (Intercept): ", theta_ols[1])
println("w_fm1 (Power):    ", theta_ols[2])
println("w_EUA (Emissions):", theta_ols[3])


#Plot of replicatin and TTF
y_hat = X * theta_ols

plot(df.Date[2000:3000], y[2000:3000], label="Actual TTF", color=:blue, linewidth=2)
plot!(df.Date[2000:3000], y_hat[2000:3000], label="Replicated TTF (Portfolio)", color=:red, linestyle=:dash)

xlabel!("Date")
ylabel!("Price")
title!("TTF vs. Replicating Portfolio")


#Example from this date: 2012-10-11
row_9_features = [1.0, 48.59, 8.13] 
prediction_9 = dot(row_9_features, theta_ols)
println("Manual Prediction for Row 9: ", prediction_9)
println("Actual Value for Row 9: 26.525")



#------Replicating using gradient descent------#
n = size(X, 1) #from earlier (X = [ones(n) df.fm1 df.EUA])
theta_gd = zeros(3)

step_size = 1 / opnorm((1/n) * X' * X)

max_iters = 500000 
tolerance = 1e-6

for i in 1:max_iters
    y_pred = X * theta_gd
    error = y_pred - y
    
    # Calculate the gradient: (1/n) * X' * error
    gradient = (1/n) * (X' * error)
    
    # Check stopping condition: stop when ||gradient|| is small
    if norm(gradient) < tolerance
        println("Gradient Descent converged at iteration $i")
        break
    end
    
    # Update step: theta = theta - alpha * gradient
    theta_gd -= step_size * gradient
end

println("GD Coefficients: ", theta_gd)
println("OLS Coefficients: ", theta_ols) # They should be very close!



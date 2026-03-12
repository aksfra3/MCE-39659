
# Packages # 
using Dates, DayCounts
using Plots
using Distributions




#---# TASK 1: #-------------------------------------------------------------------------------------------------------------------------------------------------#

####1a
valuation_date = Date(2026, 01, 23)
expiry_date = Date(2026, 12, 15)

actual_days = Dates.value(expiry_date - valuation_date)
T = actual_days / 365

####1b
r = 0.01933
discount_factor = exp(-r * T)



####1c
F0 = 88.44
strikes = [65, 70, 75, 80, 85]

function black_76_put(sigma, F0, K, T, r)

    d1 = (log(F0 / K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    std_normal = Normal(0,1)                #PDF for CDF

    put_price = exp(-r * T) * (K * cdf(std_normal, -d2) - F0 * cdf(std_normal, -d1))

    return put_price

end


σ = 0.10   
strikes = [65, 70, 75, 80, 85]       

for k_val in strikes


    price_1 = [black_76_put(σ, F0, k_val, T, r)]

    println("Strike: $k_val,   Put Price = $price_1")
end


#### 1d
F0 = 88.44
strikes = [65, 70, 75, 80, 85]
vol_range = 0.05:0.01:1.0

plot_2 = plot(title="Put Price vs Volatility (All Strikes)", 
         xlabel="Volatility (sigma)", 
         ylabel="Put Price P(sigma)",
         legend=:topleft)


for k_vals in strikes

    prices_2 = [black_76(s, F0, k_vals, T, r) for s in vol_range]

    plot!(plot_2, vol_range, prices_2, labels = "Strike (K) = $k_vals")
end

display(plot_2)




#---# TASK 3: IMPLIED VOLATILTIY #------------------------------------------------------------------------------------------------------------------------------------------------#

function black_76_put_2(sigma, K)                               #BS function for vola and strike

    r = 0.01933
    T = 326 / 365
    F0 = 88.44

    d1 = (log(F0 / K) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    std_normal = Normal(0,1)                #PDF for CDF

    put_price = exp(-r * T) * (K * cdf(std_normal, -d2) - F0 * cdf(std_normal, -d1))

    return put_price

end


function bisection_implied_vol(target_price, K)                     #Bisection algorithm for finding implied volatility             

    a = 1e-6
    b = 5.0
    
    tolerance = 1e-8
    max_iter = 100
    iterations = 0

    sigma_mid = 0.0

    while (b - a) > tolerance && iterations < max_iter
        iterations += 1
        

        sigma_mid = (a + b) / 2
        
        price_guess = black_76_put_2(sigma_mid, K)

        if price_guess > target_price
            b = sigma_mid
        else
            a = sigma_mid
        end
    end
    
    return sigma_mid, iterations
end


strikes = [65, 70, 75, 80, 85]
market_prices = [1.55, 2.19, 3.08, 4.43, 6.21]

println("Results")
println("K \t Implied Vol \t Iterations")


iv_results_2 = []                     #storing results

for i in 1:length(strikes)
    iv, iters = bisection_implied_vol(market_prices[i], strikes[i])
    push!(iv_results_2, iv)
    
    println("$(strikes[i]) \t $(round(iv*100, digits=2))% \t $iters")
end



#---# TASK 4: VOLAITLTIY SMILE #------------------------------------------------------------------------------------------------------------------------------------------------#

plot(strikes, iv_results_2, 
     marker = :circle, 
     linewidth = 2,
     title = "Task 4: Implied Volatility Surface (EUA Options)",
     xlabel = "Strike K (EUR)",
     ylabel = "Implied Volatility (sigma)",
     label = "Market IV",
     legend = :topright)





#---# TASK 5: CLIENT QUOTE FOR AN OFF-CHAIN STIRKE (BANK SETTING) #------------------------------------------------------------------------------------------------------------------------------------------------#

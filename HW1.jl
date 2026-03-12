#Packages
using Dates, DayCounts
using Plots
using Distributions


#Years to Maturity
valuation_date = Date(2026, 01, 23)
expiration_date = Date(2026, 12, 15)
dc = DayCounts.Actual365Fixed()

T = yearfrac(valuation_date, expiration_date, dc)

print("Years (T) to maturity:   ", T)


#Discount factor
discount_rate = 0.01933

discount_factor = exp(-discount_rate*T)
print("Discount factor is: ",discount_factor)


#Black Scholes- Sigma
settlement_price = 88.44
strike_levels = [65.0, 70.0, 75.0, 80.0, 85.0]      #strike levels
psm = [1.55, 2.19, 3.08, 4.43, 6.21]                

std_normal = Normal(0,1)                #PDF for CDF


function black_scholes(σ)               #Black-Scholes iteration function for vector of strike prices

    all_p = [] #storing put prices

    for i in 1:length(strike_levels)
        d1 = (log(settlement_price/strike_levels[i]) + 0.5*σ*T)/(σ*sqrt(T))
        d2 = d1 - σ*sqrt(T)

        P = discount_factor * (strike_levels[i] * cdf(std_normal,-d2) - settlement_price * cdf(std_normal, -d1))


        println("i: ", i, " "," strike: ", strike_levels[i], "  ","put: ", P, "       ", "sigma;  $(σ)")

        push!(all_p, P)

    end
    return all_p

end

black_scholes(0.05) #test



#Black Scholes- Sigma and Strike
function black_scholes2(σ,K)

        d1 = (log(settlement_price/K) + 0.5*σ*T)/(σ*sqrt(T))
        d2 = d1 - σ*sqrt(T)

        P = discount_factor * (K * cdf(std_normal,-d2) - settlement_price * cdf(std_normal, -d1))
        return P
end



# sigma, strike (K)
strike_levels = [65.0, 70.0, 75.0, 80.0, 85.0]   
sigma_levels = 0.01:0.01:1

plt = plot(title="Task 1: P(sigma) vs Sigma", xlabel="Sigma", ylabel="Price")

for b in strike_levels

    price_put=[]

    for g in sigma_levels
        pag = black_scholes2(g, b)
        push!(price_put, pag)
    end

    plot!(plt, sigma_levels, price_put)

end

display(plt)



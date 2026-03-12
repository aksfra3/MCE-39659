
# import Pkg; Pkg.add("ShiftedArrays")
# import Pkg; Pkg.add("ForwardDiff")


#Packages
using CSV
using DataFrames
using Dates
using ShiftedArrays


using Statistics
using LinearAlgebra
using Printf
using ForwardDiff

using Plots


#QUESTION 1

#Dataset
df = CSV.read(
    "/Users/akselfraser/MCE-39659/lecture_06/CRSP_Monthly.csv",
    DataFrame; #converts CSV into data frame object
    types = Dict(
        :prc        => Float64,
        :ret        => Float64,
        :retx       => Float64,
        :shrout     => Float64,
        :vol        => Float64,
        :exchcd     => Int,
        :shrcd      => Int
    ),

    strict = false,
    silencewarnings = true
)

#CRSP reports missing clse prices at the negative average of bid and ask, to fix thos we converts it to absolute value
minimum(skipmissing(df.prc))


df.prc = abs.(df.prc)

minimum(skipmissing(df.prc))


#QUESTION 2: Restrict sample to stocks on NYSE, AMEX, NASDAQ

#share code update
df = filter(:shrcd => x -> !ismissing(x) && (x == 10 || x == 11), df)
nrow(df)

#exchange code update
df = filter(:exchcd => x -> !ismissing(x) && (x == 1 || x == 2 || x == 3), df)


#calculate market cap for each stock
df.market_cap = df.prc .* df.shrout
#alternatively, we can use transform: 
transform!(df, [:prc, :shrout] => ((p, s) -> p .* s) => :market_cap)


#lag values by contact type (in this case permno): using shited array library
sort!(df, [:permno, :date])
transform!(groupby(df, :permno), :market_cap => (x -> ShiftedArrays.lag(x, 1)) => :lag1_market_cap)


#calculate weights
transform!(groupby(df, :date), :lag1_market_cap => (x -> coalesce(x, 0) ./ sum(skipmissing(x))) => :weight)



#QUESTION 2.3:

#NEW DATAFRAME: vw_market_returns

#value weighted market returns per moth         ->creates new dataframe vw_market_returns
vw_market_returns = combine(
    groupby(df, :date),
    [:ret, :weight] => ((r, w) -> sum(coalesce.(r .* w, 0))) => :vwret
)

#sort dates for vw_market_returns
vw_market_returns.date = Date.(vw_market_returns.date)
sort!(vw_market_returns, :date)


#cumulative return for each date
vw_market_returns.cumret = accumulate(*, 1 .+ vw_market_returns.vwret) .- 1



#Total cumulative return over sample period
total_cumret = prod(1 .+ vw_market_returns.vwret) - 1

plot(vw_market_returns.date, vw_market_returns.cumret,
    xlabel = "Date",
    ylabel = "Cumulative Return",
    label = "Value-Weighted Market",
    title = "Cumulative Returns")






#QUESTION 2.4: 

#Function for calculating annualized Sharpe Ratio
function calculate_annualized_sharpe(returns)

    clean_returns = filter(!ismissing, returns)     # Remove any missing values to prevent errors
    
    mu_hat = mean(clean_returns)
    sigma_hat = std(clean_returns)
    

    return sqrt(12) * (mu_hat / sigma_hat)
end



#weighted returns of each stock
df.weighted_ret = df.ret .* df.weight

#monthly market portfolio
market_portfolio = combine(groupby(df, :date), :weighted_ret => (x -> sum(skipmissing(x))) => :vw_ret) #creates new vector called vw_

sharpe_annualized = calculate_annualized_sharpe(market_portfolio.vw_ret)
println("Sharpe Annualized: $(round(sharpe_annualized, digits = 4))")


######## QUESTION 3: ########

#short term reversal
transform!(groupby(df, :permno), :ret => identity => :ret_1_0)

select!(df, Not([:ret_1_0]))
sort!(df, [:permno, :date])

g = groupby(df, :permno)



#helper function
function cumret_AB(r::AbstractVector{<:Union{Missing,Real}}, lag_from::Int, lag_to::Int)
    @assert lag_from >= lag_to >= 0

    m   = length(r)
    out = Vector{Union{Missing,Float64}}(missing, m)

    # prefix sums of log(1+r) and prefix counts of bad entries
    pref = zeros(Float64, m + 1)
    bad  = zeros(Int,     m + 1)

    @inbounds for i in 1:m
        x = r[i]
        if ismissing(x)
            pref[i+1] = pref[i]
            bad[i+1]  = bad[i] + 1
        else
            xi = Float64(x)
            if !isfinite(xi) || xi < -1.0
                pref[i+1] = pref[i]
                bad[i+1]  = bad[i] + 1
            else
                pref[i+1] = pref[i] + log1p(xi)
                bad[i+1]  = bad[i]
            end
        end
    end

    # window indices are [t-lag_from, ..., t-lag_to]
    @inbounds for t in 1:m
        start = t - lag_from
        stop  = t - lag_to
        if start < 1 || stop < 1
            continue
        end
        if (bad[stop+1] - bad[start]) == 0
            out[t] = exp(pref[stop+1] - pref[start]) - 1.0
        end
    end

    return out
end



transform!(g,
    :ret => (r -> cumret_AB(r, 1, 0)) => :ret_1_0,
    :ret => (r -> cumret_AB(r, 3, 1)) => :ret_3_1,
    :ret => (r -> cumret_AB(r, 6, 1)) => :ret_6_1,
    :ret => (r -> cumret_AB(r, 9, 1)) => :ret_9_1,
    :ret => (r -> cumret_AB(r, 12, 1)) => :ret_12_1,
    :ret => (r -> cumret_AB(r, 12, 7)) => :ret_12_7
)



#lag charactersitics by 1 month
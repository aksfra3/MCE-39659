

#- - - - - - - - -Root Finding- - - - - - - - -#

##### Bisection Method #####


## Bisection method
function bisection(f::Function, a::Float64, b::Float64, n::Int=50)
    for i in 1:n
        m = (a + b)/2
        if f(a)*f(m) <= 0
            b = m
        else
            a = m
        end
    end
    (a + b)/2
end

## Example function in class 28.01 
f(x) = x^2 - 2
bisection(f, 1.0, 4.0)


##### Newtons Method #####

##### Damped Newton #####

##### Bracketed Newton #####



#- - - - - - - - -Optimization- - - - - - - - -#



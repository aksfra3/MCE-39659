


#-------------------------------------------------------------#

#Newton Minimizing (Simple)
g(x) = x^2 - 4x - 5
g1(x) = 2x - 4
g2(x) = 2

f(x) = x^2- 2
f1(x) = 2x
f2(x) = 2

iter = 10
tolerance = 1e-6
x = 1

for i in 1:iter

    x_new = x - (g1(x)/g2(x))
    println("i: $i, x: $x, f(x) $(f(x))")

    if abs(x_new - x) < tolerance
        println("Converged at i: $i, x: $x, f(x) $(f(x))")
        break
    end


    x = x_new

end

#-------------------------------------------------------------#

#Bisection Root finding (simple)
function bisection(f, a, b, tol, N)

    for i in 1:N

        x = (a+b) / 2
        error1 = (b-a) / 2

        println("iter: $i, x: $x, ϵ: $error1")

        if f(a) * f(b) > 0
            println("Root not in interval")
            return nothing
        end

        if abs(x - a) <= tol
            ggs = round(x, digits = 4)
            println("The root is: $ggs, found after $i iterations")
            return 
        end

        if f(a) * f(b) < 0
            b = x
        else 
            a = x
        end

    end

    return x

end

N = 100
a0 = 2
b0 = 8
tol = 1e-6

f(x) = x^2 - 4x - 5

bisection(f, a0, b0, tol, N)


#-------------------------------------------------------------#


#Newton Root Finding (Simple)

function newton(g, g1, x0, tol, N)

    x = x0

    for i in 1:N

        x_new = x - (g(x)/g1(x))
        println("Iteration $i, x: $x, f(x): $(f(x))")

        if abs(x_new - x) < tol
            println("Converged at iteration $i, x = $x, f(x) = $(f(x))")
            break
        end

        x = x_new

    end

end


g(x) = x^2 - 4x - 5
g1(x) = 2x - 4
g2(x) = 2
x0 = 10
tol = 1e-6
N = 100


newton(g, g1, x0, tol, N)
###### PACKAGES #####

###### ARRAY BASICS ######

#--- Vectors

# A column vector (standard in math/Julia)
x = [10, 20, 30] 

# A row vector (notice the spaces, not commas)
y = [10, 20, 30]

x + y           #works

x * y           #doesnt work

#--- Random Numbers
z=randn(10000)

using Plots
plot(z)
histogram(z, bins = 100)


typeof(z)


# A range (useful for time steps t=1,2,3...)
t = 1:10                    # Creates a range from 1 to 10
t_array = collect(1:10)     # Converts the range to an actual array


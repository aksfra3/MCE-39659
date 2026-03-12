


#Know details to taylor approximation

#for exam: how to write code to use automatic differentiation
#   what to do: all operations need to be basic math operations- make it smooth
#   use ForwardDiff.jl or Zygote.jl packages
#                       fowarddiff.gradient()
#  eg. dont use max(a,b) or abs(a) because they are not differentiable, instead use smooth approximations like log(exp(a)+exp(b)) for max and sqrt(a^2 + eps()) for abs
#   ^will come on midterm

#multiple codes = zip folder

#how to present code on slides
        # 


#fine with null result eg. cannot trade on this strat- explain why it doesnt work (eg condition number)


#make sure data is clean- start with right data!!! fix data
#check correct alignment. eg price apple 

#bad scaling: eg euros and interest rates or dollars and number ofr barrels of oil, check conditioning number

#fewer libraries = better





#last ML part ( linalg instead of dataframes)
#dont regularize the intercept

# use epsilon on midterm, need to make lasso penalties smooth


#set constant step 
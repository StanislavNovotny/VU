using Flux
using Plots

dataX = collect(-20.0:0.1:20.0)
W = 3.0
k = Int64.(dataX .<= 0)

p(x) = exp.(W*log.(abs.(x))).*cos.(pi.*W*k)

plot(p(dataX))

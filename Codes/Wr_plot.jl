using Flux
using Plots

dataX = collect(-7.0:0.1:7.0)
W = -2.0
k = Int64.(dataX .<= 0)

p(x) = exp.(W*log.(abs.(x))).*cos.(pi.*W*k)

plot(p(dataX))

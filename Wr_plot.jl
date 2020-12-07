using Flux
using Plots

dataX = collect(-7.0:0.1:7.0)
W = -2.0
k = zeros(length(dataX))
for i in 1:length(dataX)
    if dataX[i] > 0
        k[i] = 1
    end
end

p(x) = exp.(W.*log.(abs.(x))).*cos.(W.*pi.*k)

plot(p(dataX))

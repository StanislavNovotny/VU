using Flux
using Plots
using Flux: throttle

#generovani dat

dataX = rand(100)
dataY = 50 .+ 100 * dataX + 2 * randn(100);
scatter(dataX, dataY, legend=:bottomright, label="data")

#presny fit

X = hcat(ones(length(dataX)),dataX)
Y = dataY
posun,sklon = inv(X'*X)*(X'*Y)

plot!((x) -> posun + sklon * x, 0, 1, label="Presny fit")

#Flux model

data = zip(dataX, dataY)
model = Dense(1, 1, identity)
loss(x, y) = Flux.mse(model([x]), y)
opt = ADAM(0.1)

for i = 1:100
  Flux.train!(loss, params(model), data, opt)
end

(θ, bias) = Flux.params(model)

plot!((x) ->  bias[1] + θ[1] * x, 0, 1, label="Flux fit")

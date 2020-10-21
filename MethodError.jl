using Flux
using Plots
using NeuralArithmetic

#generovani dat

dataX = -5.0:0.4:5.0
dataYr = 50 .+ 100 * dataX.^3;
dataY = dataYr + 1000 * randn(length(dataX))

scale = 1.3*maximum(dataY)
dataYr /= scale
dataY /= scale

#Flux model

n = 29 # hidden neurons
iter = 500 # iterace
model = Chain(NPU(1,n), NAU(n,1))
loss(x,y) = Flux.mse(model(x),y)
opt = ADAM()

LL = zeros(1,iter);

ps = params(model);

for i=1:iter
  l = loss(dataX',dataY')
  gs = gradient(()->loss(dataX',dataY'),ps)
  Flux.Optimise.update!(opt, ps, gs)
  LL[i]= l
end

p = plot(dataX,dataYr,linestyle=:dash,xlabel="x",ylabel="y",label="Prava data")

plot!(dataX,dataY,seriestype=:scatter,label="data")

y=(model(dataX'))
plot!(dataX,Flux.data(y)[:],label="Predikce")

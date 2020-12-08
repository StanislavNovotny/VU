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

X = convert(Array{Float32}, dataX)
Yr = convert(Array{Float32}, dataYr)
Y = convert(Array{Float32}, dataY)

#Flux model

n = 512 # hidden neurons
iter = 5000 # iterace
model = Chain(NPU(1,n), NAU(n,1))

psNPU = params(model[1])
psNAU = params(model[2])


loss(x,y) = Flux.mse(model(x),y) +
              0.0001*sum(abs.(vcat([params(model)[i][:] for i
             in 1:length(params(model))]...)))


opt = ADAM()

LL = zeros(1,iter);

ps = params(model);

for i=1:iter
  l = loss(X',Y')
  gs = gradient(()->loss(X',Y'),ps)
  Flux.Optimise.update!(opt, ps, gs)
  LL[i]= l
end

p = plot(X,Yr,linestyle=:dash,xlabel="x",ylabel="y",label="Prava data")

plot!(X,Y,seriestype=:scatter,label="data")

y=(model(X'))
plot!(X,Flux.data(y)[:],label="Predikce")

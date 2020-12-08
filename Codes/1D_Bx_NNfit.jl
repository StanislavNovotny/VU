using Flux
using Plots
using NeuralArithmetic
using MAT

#generovani dat

M = matread("function_data-1D.mat")
dataX = collect(1.0:220.0) .- 103
dataY = vec(M["Bx_rez"])

scale = 1.3*maximum(dataY)
dataY /= scale

X = Float32.(dataX)
Y = Float32.(dataY)

#Flux model


iter = 4000 # iterace
model = Chain(NAU(1,5), Dense(5,5,identity), NPU(5,4), NAU(4,1), NPU(1,1), NAU(1,1))

#psNPU = params(model[1])
#psNAU = params(model[2])


sqnorm(x) = sum(abs, x)

loss(x,y) = Flux.mse(model(x),y) +
              0*sum(sqnorm, Flux.params(model))


opt = ADAM(0.01)

LL = zeros(1,iter);

ps = params(model);

for i=1:iter
  l = loss(X',Y')
  gs = gradient(()->loss(X',Y'),ps)
  Flux.Optimise.update!(opt, ps, gs)
  LL[i]= l
end


p = plot(X,Y,seriestype=:scatter,markersize = 1,label="data")

y=(model(X'))
plot!(X,Flux.data(y)[1,:],label="Predikce")

#savefig("NN_4000iter")

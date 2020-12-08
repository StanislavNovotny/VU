using Flux
using Plots
using NeuralArithmetic

#p(x) = (25.2.*((x.-0.004)./0.116) .+ 0.688) ./(((x.-0.014)./0.43).^4 +
# 1.22.*((x.+0.27)./1.25).^3 .+ 0.25.*((x.-0.6)./0.72).^2 .+1.17.*
# ((x.+0.34)./1.38).+0.001)

p(x) = 1/(((x-1)/2)^2+((x+3)/4)^4)

plot(p)

dataX = collect(-7.0:0.01:7.0)


Y = p.(dataX)
plot(Y)

X = Float32.(dataX)
Y = Float32.(Y)

iter = 6000 # iterace
model = Chain(NAU(1,2),Dense(2,2,identity),NPU(2,2),NAU(2,1),NPU(1,1))

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

o = plot(X,Y,seriestype=:scatter,markersize = 1,label="data")

y=(model(X'))
plot!(X,Flux.data(y)[1,:],label="Predikce")

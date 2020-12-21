using Flux
using Plots
using NeuralArithmetic
using LinearAlgebra

#init
p(x) = -x^3 + 4*x^2 - 16
Dom = collect(1.0:0.1:6.0)
Y = p.(Dom)

X = Float32.(Dom)
Y = Float32.(Y)

o = plot(X,Y,seriestype=:scatter,markersize = 1,label="data")

model = Chain(NaiveNPU(1,2),Dense(2,1,identity))

Wr = [5,3]
Wi = [0,0]
A = [-5,4]
b = [-16]

FinalParams = [3, 2, 0, 0, -1, 4, -16]

MaxIter = 8000
sqnorm(x) = sum(abs, x)
loss(x,y) = Flux.mse(model(x),y) +
              0*sum(sqnorm, Flux.params(model))

opt = ADAM(0.01)
ps = params(model)
LL = zeros(1,MaxIter);

function InitParams()
  params(model[1])[1][:] .= Wr
  params(model[1])[2][:] .= Wi
  params(model[2])[1][:] .= A
  params(model[2])[2]    .= b
  return ps
end

function FreezeParams(Freeze)
  tmp = 0
  if Freeze < 2
    tmp = 1
  elseif Freeze < 4
    tmp = 2
  else
    tmp = 3
  end

  ps = params(model[tmp:end])

  if Freeze%2 == 1
    if Freeze == 1
      delete!(ps,params(model[1])[1])
    else
      delete!(ps,params(model[2])[1])
    end
  end
  return ps
end

#_______________________________________________________________________________

InitParams()
FreezeParams(2)
tmp1 = vcat([params(model)[i][:] for i in 1:length(params(model))]...)
for i=1:MaxIter
  NoI = i
  l = loss(X',Y')
  gs = gradient(()->loss(X',Y'),ps)
  Flux.Optimise.update!(opt, ps, gs)
  LL[i]= l
  if sum(abs.(tmp1 .- FinalParams) .< 0.01) == length(FinalParams)
        println("Pocet iteraci: ",NoI)
    break
  end
end

gs = gradient(()->loss(X',Y'),ps)

println("Parametry modelu: ",params(model))
println("Hodnota ztratove fce: ",LL[end])
println("Grad Wr: ",norm(gs[ps[1]]))
println("Grad Wi: ",norm(gs[ps[2]]))
println("Grad A: ",norm(gs[ps[3]]))
println("Grad b: ",norm(gs[ps[4]]))
#_______________________________________________________________________________

y=(model(X'))
plot!(X,y[:],label="Predikce")

plot(LL', label="Lost function")

plot(log.(LL)')

#savefig("NN3k_psFREE")

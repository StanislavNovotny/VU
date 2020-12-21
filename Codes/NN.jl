using Flux
using Plots
using NeuralArithmetic
using LinearAlgebra

λ = 0
T = Float32

#init
p(x) = -x^3 + 4*x^2 - 16

X = Matrix(T.(1.0:0.1:6.0)')
Y = p.(X)

model = Chain(NaiveNPU(1,2),Dense(2,1,identity))

Wr = T.([5; 3])[:,:]
Wi = T.([0; 0])[:,:]
A = T.([-5 4])
b = T.([-16])

final_params = T.([3, 2, 0, 0, -1, 4, -16])

max_iter = 8000
sqnorm(x) = sum(abs, x)
loss(x,y,λ) = Flux.mse(model(x),y) + λ*sum(sqnorm, Flux.params(model))

opt = ADAM(0.01)
ps = params(model)
LL = zeros(max_iter);

function InitParams!(model, Wr, Wi, A, b)
  params(model[1])[1] .= Wr
  params(model[1])[2] .= Wi
  params(model[2])[1] .= A
  params(model[2])[2] .= b
  return nothing
end

function FreezeParams!(model, freeze)
  tmp = 0
  if freeze < 2
    tmp = 1
  elseif freeze < 4
    tmp = 2
  else
    tmp = 3
  end

  ps = params(model[tmp:end])

  if freeze%2 == 1
    if freeze == 1
      delete!(ps,params(model[1])[1])
    else
      delete!(ps,params(model[2])[1])
    end
  end
  return ps
end

#_______________________________________________________________________________

InitParams!(model, Wr, Wi, A, b)
FreezeParams!(model, 2)
for i=1:max_iter
  NoI = i
  LL[i] = loss(X,Y,λ)
  gs = gradient(()->loss(X,Y,λ),ps)
  Flux.Optimise.update!(opt, ps, gs)
end

gs = gradient(()->loss(X,Y,λ),ps)

println("Parametry modelu: ",params(model))
println("Hodnota ztratove fce: ",LL[end])
println("Grad Wr: ",norm(gs[ps[1]]))
println("Grad Wi: ",norm(gs[ps[2]]))
println("Grad A: ",norm(gs[ps[3]]))
println("Grad b: ",norm(gs[ps[4]]))
#_______________________________________________________________________________

y=(model(X))

scatter(X[:],Y[:],markersize = 1,label="data")
plot!(X[:],y[:],label="Predikce")

plot(LL, label="Lost function")

plot(log.(LL))

#savefig("NN3k_psFREE")

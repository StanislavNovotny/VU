using Flux
using Plots
using NeuralArithmetic
using LinearAlgebra
using LatexTables

λ = 0
T = Float32

#init
p(x) = -x^3 + 4*x^2 - 16

X = Matrix(T.(1.0:0.1:6.0)')
Y = p.(X)

model = Chain(NaiveNPU(1,2),Dense(2,1,identity))

Wr = T.([0; 0])[:,:]
Wi = T.([0; 0])[:,:]
A = T.([0 0])
b = T.([0])

max_iter = 9000
sqnorm(x) = sum(abs, x)
loss(x,y,λ) = Flux.mse(model(x),y) + λ*sum(sqnorm, Flux.params(model))

opt = ADAM(0.01)
ps = params(model)
LL = zeros(max_iter);

function init_params!(model, data::Tuple)
    i = data[1]
    j = data[2]
    x = data[3]

    (params(model[i]))[j] .= x

    return nothing
end

function freeze_params!(model, ps, data::Tuple)
    i = data[1]
    j = data[2]

    delete!(ps,params(model[i])[j])

    return nothing
end

#_______________________________________________________________________________

init = [(1,1,Wr),(1,2,Wi), (2,1,A), (2,2,b)]
frz = []

[init_params!(model, data) for data in init]
[freeze_params!(model, ps, data) for data in frz]

for i=1:max_iter
  LL[i] = loss(X,Y,λ)
  gs = gradient(()->loss(X,Y,λ),ps)
  Flux.Optimise.update!(opt, ps, gs)
end

ps = params(model)
gs = gradient(()->loss(X,Y,λ),ps)

println("Parametry modelu: ",ps)
println("Hodnota ztratove fce: ",LL[end])
println("Grad Wr: ",norm(gs[ps[1]]))
println("Grad Wi: ",norm(gs[ps[2]]))
println("Grad A: ",norm(gs[ps[3]]))
println("Grad b: ",norm(gs[ps[4]]))
#_______________________________________________________________________________

y = model(X)

scatter(X[:],Y[:],markersize = 1,label="data")
plot!(X[:],y[:],label="Predikce 1")

#savefig("Frz0")
plot(LL, label="Lost function")
plot(log.(LL), label="Log of Lost function")

#savefig("")
##

body1 = rand(4,7)
body1[4,:] .= vcat([params(model)[i][:] for i in 1:length(params(model))]...)

s1 = table_to_tex(body1;
    caption="???",
    header=["\$W^r[1]\$", "\$W^r[2]\$", "\$W^i[1]\$", "\$W^i[2]\$", "Dense[1]", "Dense[2]", "Bias"],
    other_rules = [Cells((1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2)) => (CellColor(:gray!50),)]
)

print(s1)
##
body2 = rand(4,5)
body2[1,:] .= [LL[end], norm(gs[ps[1]]), norm(gs[ps[2]]), norm(gs[ps[3]]), norm(gs[ps[4]])]

s2 = table_to_tex(body2;
    caption="???",
    header=["\$Loss\$", "\$NoG Wr\$", "\$NoG Wi\$", "\$NoG A\$", "NoG b"]
)

print(s2)
##

print(s2)

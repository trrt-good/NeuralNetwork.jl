"""
this is to figure out the backpropagation calculus on a small scale 
before scaling it to the flexible algorithm in `NeuralNet.jl`

--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

L1 : 3 nodes (input layer)
L2 : 4 nodes (hidden layer)
L4 : 2 nodes (output layer)

a1

a₂
a₃

w₂
w₃

b₂
b₃

r(x) = max(x, 0)
r'(x) = (x < 0)? 0 : x
C(a₁) = (r(w₃*r(w₂*a₁+b₂)+b₃)-y)²
tₙ(aₙ₋₁) = wₙ*aₙ₋₁+bₙ
C(a₁) = (r(t₃(r(t₂(a₁))))-y)²
C'(t₂) = 2*(r(t₃(r(t₂(a₁))))-y)*r'(t₃(r(t₂(a₁))))

C'(a₂) = 


"""

traingNeuralNet(a1::Array, x::Array)
    a2 = Array{Float32, 1}(1, 4)
    a3 = Array{Float32, 1}(1, 4)

    w2 = Array{Float32, 2}(1, 4, 3)
    w3 = Array{Float32, 2}(1, 4, 4)

    b2 = Array{Float32, 1}(0, 4)
    b3 = Array{Float32, 1}(0, 4)
end
"""
this is to figure out the backpropagation calculus on a small scale 
before scaling it to the flexible algorithm in `NeuralNet.jl`

--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

L0 : 3 nodes (input layer)
L1 : 4 nodes (hidden layer)
L2 : 4 nodes (hidden layer)
L3 : 2 nodes (output layer)

a0: input data

a1: value of nodes in layer 1
a2: value of nodes in layer 2
a3: value of nodes in layer 3

w1: weights for layer 1, which connect a0 and a1
w2: weights for layer 2, which connect a1 and a2
w3: weights for layer 3, which connect a2 and a3

b1: biases for layer 1
b2: biases for layer 2
b3: biases for layer 3

"""

const LEARN_RATE = 0.5

trainNeuralNet(a0::Array)
    a1 = Array{Float32, 1}(1, 4)
    a2 = Array{Float32, 1}(1, 4)
    a3 = Array{Float32, 1}(1, 2)
     
    w1 = Array{Float32, 2}(1, 3, 4)
    w2 = Array{Float32, 2}(1, 4, 4)
    w3 = Array{Float32, 2}(1, 4, 2)

    b1 = Array{Float32, 1}(0, 4)
    b2 = Array{Float32, 1}(0, 4)
    b3 = Array{Float32, 1}(0, 2)

    for i in 1:1000
        # run data through the network 
        a1 = w1*a0+b1
        a2 = w2*a1+b2
        a3 = w3*a2+b3

        # change weights
        w1_temp = w1 - LEARN_RATE*2*transpose(w3*w2)*transpose(a0*transpose(a3))
        w2_temp = w2 - LEARN_RATE*2*transpose(w3)*transpose(a1*transpose(a3))
        w3_temp = w3 - LEARN_RATE*2*transpose(a2*transpose(a3))

        # change biases
        b1_temp = b1 - LEARN_RATE*2*transpose(w3*w2)*a3
        b2_temp = b2 - LEARN_RATE*2*transpose(w3)*a3
        b3_temp = b3 - LEARN_RATE*2*a3

        # update actual weights and biases to temp values
        w1 = w1_temp
        w2 = w2_temp
        w3 = w3_temp

        b1 = b1_temp
        b2 = b2_temp
        b3 = b3_temp
    end

    
end
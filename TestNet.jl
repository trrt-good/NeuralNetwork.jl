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

const LEARN_RATE = 0.005
const INPUT_ARRAY = Array{Float32, 1}([1, -10, 3])
const OUTPUT_ARRAY = Array{Float32, 1}([30, 50])

function relu(A::Array)
    for i in 1:length(A)
        A[i] = max(0, A[i])
    end
    return A
end

function trainNeuralNet(a0::Array, y::Array)
    a1 = fill(1.0, 4)
    a2 = fill(1.0, 4)
    a3 = fill(1.0, 2)
     
    w1 = fill(1.0, (4, 3))
    w2 = fill(1.0, (4, 4))
    w3 = fill(1.0, (2, 4))

    b1 = fill(0.0, 4)
    b2 = fill(0.0, 4)
    b3 = fill(0.0, 2)

    for i in 1:1000
        # run data through the network 
        a1 = relu(w1*a0+b1)
        a2 = relu(w2*a1+b2)
        a3 = relu(w3*a2+b3)

        # change weights
        w1_temp = w1 - LEARN_RATE*2*transpose(relu(w3*w2))*transpose(a0*transpose(a3-y))
        w2_temp = w2 - LEARN_RATE*2*transpose(relu(w3))*transpose(a1*transpose(a3-y))
        w3_temp = w3 - LEARN_RATE*2*transpose(a2*transpose(a3-y))

        # change biases
        b1_temp = b1 - LEARN_RATE*2*transpose(relu(w3*w2))*(a3-y)
        b2_temp = b2 - LEARN_RATE*2*transpose(relu(w3))*(a3-y)
        b3_temp = b3 - LEARN_RATE*2*(a3-y)

        # update actual weights and biases to temp values
        w1 = w1_temp
        w2 = w2_temp
        w3 = w3_temp

        b1 = b1_temp
        b2 = b2_temp
        b3 = b3_temp

        println(a3)
    end

    a1 = relu(w1*a0+b1)
    a2 = relu(w2*a1+b2)
    a3 = relu(w3*a2+b3)

    print(a3)
end

trainNeuralNet(INPUT_ARRAY, OUTPUT_ARRAY)
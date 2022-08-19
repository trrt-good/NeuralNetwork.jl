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

function relu(A)
    for i in eachindex(A)
        A[i] = max(0, A[i])
    end
    return A
end

function fixedNeuralNet(a0::Array, y::Array)
    LEARN_RATE = 0.01

    a1 = fill(1.0, 4)
    a2 = fill(1.0, 4)
    a3 = fill(1.0, 2)
     
    w1 = fill(1, (4, 3))
    w2 = fill(1, (4, 4))
    w3 = fill(1, (2, 4))

    b1 = fill(0.0, 4)
    b2 = fill(0.0, 4)
    b3 = fill(0.0, 2)

    for i in 1:1
        # run data through the network 
        a1 = relu(w1*a0+b1)
        a2 = relu(w2*a1+b2)
        a3 = relu(w3*a2+b3)

        # change weights
        w1_temp = LEARN_RATE*transpose(w3*w2)*transpose(a0*transpose(a3-y))
        w2_temp = LEARN_RATE*transpose(w3)*transpose(a1*transpose(a3-y))
        w3_temp = LEARN_RATE*transpose(a2*transpose(a3-y))

        # change biases
        b1_temp = LEARN_RATE*transpose(w3*w2)*(a3-y)
        b2_temp = LEARN_RATE*transpose(w3)*(a3-y)
        b3_temp = LEARN_RATE*(a3-y)

        # update actual weights and biases to temp values
        w1 -= w1_temp
        w2 -= w2_temp
        w3 -= w3_temp

        b1 -= b1_temp
        b2 -= b2_temp
        b3 -= b3_temp

        println(w3_temp)
        println(b3_temp)
        println(w2_temp)
        println(b2_temp)

    end

    a1 = relu(w1*a0+b1)
    a2 = relu(w2*a1+b2)
    a3 = relu(w3*a2+b3)

    println("layer 1:")
    println(w1)
    println(b1)
    println("layer 2:")
    println(w2)
    println(b2)
    println("layer 3:")
    println(w3)
    println(b3)
    println("activation:")
    println(a3)

end

fixedNeuralNet(Array{Float32, 1}([1, 0, 0]), Array{Float32, 1}([10, 10]))
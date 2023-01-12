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

const INPUT_ARRAY = [
    [1, 2, 3], 
    [0, 1, 0], 
    [0, 0, 1], 
    [1, 1, 0]
]
const OUTPUT_ARRAY =[
    [10, 1], 
    [1, 0], 
    [0, 0], 
    [7, 3]
]

function relu(A)
    for i in eachindex(A)
        A[i] = max(0, A[i])
    end
    return A
end

function randomMatrix(rows, columns)
    # return rand(Float32, rows, columns
    return fill(1.0, (rows, columns))
end

function cost(activation, y)
    return sum((y-activation).^2)
end

function fixedNeuralNet(a0::Array, y::Array)
    println("\n\n\n\n")
    LEARN_RATE = 0.0005
    AVGMULT = 1.0/length(a0)

    a1 = randomMatrix(4, 1)
    a2 = randomMatrix(4, 1)
    a3 = randomMatrix(2, 1)
     
    w1 = randomMatrix(4, 3)
    w2 = randomMatrix(4, 4)
    w3 = randomMatrix(2, 4)

    w1_temp = zeros(4, 3)
    w2_temp = zeros(4, 4)
    w3_temp = zeros(2, 4)

    b1 = randomMatrix(4, 1)
    b2 = randomMatrix(4, 1)
    b3 = randomMatrix(2, 1)

    b1_temp = zeros(4, 1)
    b2_temp = zeros(4, 1)
    b3_temp = zeros(2, 1)

    for i in 1:1000000
        cst = 0;
        for nthExample in 1:length(a0)
            # run data through the network 
            a1 = relu(w1*a0[nthExample]+b1)
            a2 = relu(w2*a1+b2)
            a3 = relu(w3*a2+b3)

            # println(a1)
            # println(a2)
            # println(a3)
            cst += cost(a3, y[nthExample])

            # println(cost(a3, y[nthExample]))

            # change weights
            w1_temp += LEARN_RATE*transpose(w3*w2)*transpose(a0[nthExample]*transpose(a3-y[nthExample]))
            w2_temp += LEARN_RATE*transpose(w3)*transpose(a1*transpose(a3-y[nthExample]))
            w3_temp += LEARN_RATE*transpose(a2*transpose(a3-y[nthExample]))

            # change biases
            b1_temp += LEARN_RATE*transpose(w3*w2)*(a3-y[nthExample])
            b2_temp += LEARN_RATE*transpose(w3)*(a3-y[nthExample])
            b3_temp += LEARN_RATE*(a3-y[nthExample])

            # println(b2_temp)
            # println(a3-y)
            # println(w3)

            # update actual weights and biases to temp values
            
            # println(w1_temp)
            # println(b1_temp)
            # println(w2_temp)
            # println(b2_temp)
            # println(w3_temp)
            # println(b3_temp)
            # println("\n\n\n\n")
        end
        w1 -= w1_temp.*AVGMULT
        w2 -= w2_temp.*AVGMULT
        w3 -= w3_temp.*AVGMULT

        b1 -= b1_temp.*AVGMULT
        b2 -= b2_temp.*AVGMULT
        b3 -= b3_temp.*AVGMULT

        w1_temp = zeros(4, 3)
        w2_temp = zeros(4, 4)
        w3_temp = zeros(2, 4)

        b1_temp = zeros(4, 1)
        b2_temp = zeros(4, 1)
        b3_temp = zeros(2, 1)

        # print("cost: ")
        # println(cst*AVGMULT)
    end

    # a1 = relu(w1*a0[1]+b1)
    # a2 = relu(w2*a1+b2)
    # a3 = relu(w3*a2+b3)

    # println(y[1])
    # println(a3)

    # cst = cost(a3, y[1]);

    # a1 = relu(w1*a0[2]+b1)
    # a2 = relu(w2*a1+b2)
    # a3 = relu(w3*a2+b3)

    # cst = (cst + cost(a3, y[2]))/2;

    # println(y[2])
    # println(a3)

    # println("cost:")
    # println(cst)

    # println("layer 1:")
    # println(w1)
    # println(b1)
    # println("layer 2:")
    # println(w2)
    # println(b2)
    # println("layer 3:")
    # println(w3)
    # println(b3)
    # println("activation:")
    # println(a3)

end

fixedNeuralNet(INPUT_ARRAY, OUTPUT_ARRAY)
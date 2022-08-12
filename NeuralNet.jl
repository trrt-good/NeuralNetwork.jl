const initialWeightValue = 1.0
const initialActivationValue = 1.0
const initialBiasValue = 0.0

function validateArgs(nodesInEachLayer::Array, inputData::Array, outputData::Array)
    # do some checks like matching lengths of input and output 
    # and consistant number of inputs per training example
    return nothing
end

function relu(A)
    for i in eachindex(A)
        A[i] = max(0, A[i])
    end
    return A
end

function trainNeuralNet(nodesInEachLayer::Array, inputData::Array, outputData::Array)
    # validate data:
    validity = validateArgs(nodesInEachLayer, inputData, outputData)
    if (!isnothing(validity))
        print(validity)
        return validity
    end

    # initialize neural network parts (weights, activators, biases) based on `nodesInEachLayer`:

    # weights (using an array of 2d matrices):
    weights = Array{Matrix, 1}(undef, length(nodesInEachLayer)-1)
    weights_temp = Array{Matrix, 1}(undef, length(nodesInEachLayer)-1)
    for index in eachindex(weights)
        # first index of `nodesInEachLayer` is the input layer 
        # which doesnt have an associated weight matrix
        weights[index] = fill(Float32(initialWeightValue), (nodesInEachLayer[index + 1], nodesInEachLayer[index]))
        weights_temp[index] = fill(Float32(initialWeightValue), (nodesInEachLayer[index + 1], nodesInEachLayer[index]))
    end

    # activators (does not include input layer):
    activators = Array{Vector, 1}(undef, length(nodesInEachLayer)-1)
    activators_temp = Array{Vector, 1}(undef, length(nodesInEachLayer)-1)
    for index in eachindex(activators)
        activators[index] = fill(Float32(initialWeightValue), nodesInEachLayer[index])
        activators_temp[index] = fill(Float32(initialWeightValue), nodesInEachLayer[index])
    end

    # biases
    biases = Array{Vector, 1}(undef, length(nodesInEachLayer)-1)
    biases_temp = Array{Vector, 1}(undef, length(nodesInEachLayer)-1)
    for index in eachindex(biases)
        biases = fill(Float32(initialWeightValue), nodesInEachLayer[index + 1])
        biases_temp = fill(Float32(initialWeightValue), nodesInEachLayer[index + 1])
    end

    for i in 1:1000
        # run data through the network
        activators[1] = relu(weights[1]*inputData+biases[1])
        for a in 2:length(activators)
            activators[a] = relu(weights[a]*activators[a-1]+biases[a])
        end

        # change weights
        for (index, weight) in enumerate(weights_temp)
            
        end
    end
end
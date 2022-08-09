const initialWeightValue = 1
const initialActivationValue = 1
const initialBiasValue = 0

function validateData(inputData::Array, outputData::Array)
    # do some checks like matching lengths of input and output 
    # and consistant number of inputs per training example
    return nothing
end

function trainNeuralNet(nodesInEachLayer::Array, inputData::Array, outputData::Array)
    # validate data:
    validity = validateData(inputData, outputData)
    if (!isnothing(validity))
        print(validity)
        return validity
    end

    #initialize neural network parts (weights, activators, biases) based on `nodesInEachLayer`
    #weights (using an array of 2d matrices):
    weights = Array{Array, 1}(undef, length(nodesInEachLayer)-1)
    for (index, entry) in enumerate(weights)
        # first index of `nodesInEachLayer` is the input layer 
        # which doesnt have an associated weight matrix
        entry = Array{Float32, 2}(initialWeightValue, nodesInEachLayer[index + 1], nodesInEachLayer[index])
    end

    #activators (does not include input layer):
    activators = Array{Array, 1}(undef, length(nodesInEachLayer)-1)
    for (index, entry) in enumerate(activators)
        entry = Array{Float32, 1}(initialActivationValue, nodesInEachLayer[index])
    end

    #biases
    biases = Array{Array, 1}(undef, length(nodesInEachLayer)-1)
    for (index, entry) in enumerate(activators)
        entry = Array{Float32, 1}(initialBiasValue, nodesInEachLayer[index + 1])
    end
end
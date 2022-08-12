const LEARN_RATE = 0.01

const INPUT_ARRAY = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
const OUTPUT_ARRAY =[[0, 1], [1, 0], [0, 0], [10, 10]]
const NODES_EACH_LAYER = [3, 4, 4, 2]

function validateArgs(nodesInEachLayer::Array, inputData::Array, outputData::Array)
    # do some checks like matching lengths of input and output 
    # and consistant number of inputs per training example
    return nothing
end

function relu(A)
    for i in eachindex(A)
        A[i] = Float32(max(0.0, A[i]))
    end
    return A
end

function resetToZero(A)
    for i in eachindex(A)
        for j in eachindex(A[i])
            A[i][j] = Float32(0)
        end
    end
end

function trainNeuralNet(nodesInEachLayer::Array, inputData::Array, outputData::Array)
    # validate data:
    validity = validateArgs(nodesInEachLayer, inputData, outputData)
    if (!isnothing(validity))
        print(validity)
        return validity
    end

    nLayers = length(nodesInEachLayer)-1
    nTrainingExamples = length(inputData)
    avgMult = 1/nTrainingExamples

    # initialize neural network parts (weights, activators, biases) based on `nodesInEachLayer`:

    # weights (using an array of 2d matrices):
    weights = Array{Matrix, 1}(undef, length(nodesInEachLayer)-1)
    weights_change = Array{Matrix, 1}(undef, length(nodesInEachLayer)-1)
    for index in eachindex(weights)
        # first index of `nodesInEachLayer` is the input layer 
        # which doesnt have an associated weight matrix
        weights[index] = rand(Float32, (nodesInEachLayer[index + 1], nodesInEachLayer[index]))
        weights_change[index] = fill(Float32(0), (nodesInEachLayer[index + 1], nodesInEachLayer[index]))
    end

    # activators (does not include input layer):
    activators = Array{Vector, 1}(undef, length(nodesInEachLayer)-1)
    for index in eachindex(activators)
        activators[index] = rand(Float32, nodesInEachLayer[index])
    end

    # biases
    biases = Array{Vector, 1}(undef, length(nodesInEachLayer)-1)
    biases_change = Array{Vector, 1}(undef, length(nodesInEachLayer)-1)
    for index in eachindex(biases)
        biases[index] = rand(Float32, nodesInEachLayer[index + 1])
        biases_change[index] = fill(Float32(0), nodesInEachLayer[index + 1])
    end

    for i in 1:10000
        for nthExample in 1:nTrainingExamples
            # run data through the network
            activators[1] = relu(weights[1]*inputData[nthExample]+biases[1])
            for a in 2:length(activators)
                activators[a] = relu(weights[a]*activators[a-1]+biases[a])
            end

            # change change weights
            weights_change[nLayers] -= avgMult*LEARN_RATE*transpose(activators[nLayers-1]*transpose(activators[nLayers]-outputData[nthExample])) #update the last weight on it's own because it's special
            for l in 1:nLayers-1
                weightProduct = weights[l+1]
                for lp in l+2:nLayers
                    weightProduct = weights[lp]*weightProduct
                end
                weights_change[l] -= avgMult*LEARN_RATE*(transpose(weightProduct)*transpose((l == 1 ? inputData[nthExample] : activators[l-1])*transpose(activators[nLayers]-outputData[nthExample])))
            end

            #change change biases 
            biases_change[nLayers] -= avgMult*LEARN_RATE*(activators[nLayers]-outputData[nthExample]) #update the last bias on it's own because it's special too
            for l in 1:nLayers-1
                weightProduct = weights[l+1]
                for lp in l+2:nLayers
                    weightProduct = weights[lp]*weightProduct
                end
                biases_change[l] -= avgMult*LEARN_RATE*(transpose(weightProduct)*(activators[nLayers]-outputData[nthExample]))
            end
        end

        #update biases and weights to change values
        for index in 1:nLayers
            weights[index] = weights[index] + weights_change[index]
            biases[index] = biases[index] + biases_change[index]
        end

        resetToZero(weights_change)
        resetToZero(biases_change)
    end

    println("\n\n\n")

    println("biases: $biases")
    println("weights: $weights")
    
    for i in 1:nTrainingExamples
        activators[1] = relu(weights[1]*inputData[i]+biases[1])
        for a in 2:length(activators)
            activators[a] = relu(weights[a]*activators[a-1]+biases[a])
        end

        println("activators: $activators")
    end

    
end

trainNeuralNet(NODES_EACH_LAYER, INPUT_ARRAY, OUTPUT_ARRAY)
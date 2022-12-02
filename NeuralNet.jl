 LEARN_RATE = 0.0001

 INPUT_ARRAY = [
    [1, 2, 3], 
    [0, 1, 0], 
    [0, 0, 1], 
    [1, 1, 0]
]
 OUTPUT_ARRAY =[
    [10, 1], 
    [1, 0], 
    [0, 0], 
    [10, 10]
]
 NODES_EACH_LAYER = [3, 4, 4, 2]

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

function cost(activation, y)
    return sum((y-activation).^2)
end

function trainNeuralNet(nodesInEachLayer::Array, inputData::Array, outputData::Array)
    # validate data:
    validity = validateArgs(nodesInEachLayer, inputData, outputData)
    if (!isnothing(validity))
        print(validity)
        return validity
    end

    nLayers = length(nodesInEachLayer)-1
    println("Layers: $nLayers")
    nTrainingExamples = length(inputData)
    println("TrainingExamples: $nTrainingExamples")
    avgMult = 1/nTrainingExamples

    # initialize neural network parts (weights, activators, biases) based on `nodesInEachLayer`:

    # weights (using an array of 2d matrices):
    weights = Array{Matrix, 1}(undef, nLayers)
    weights_change = Array{Matrix, 1}(undef, nLayers)
    for index in eachindex(weights)
        # first index of `nodesInEachLayer` is the input layer 
        # which doesnt have an associated weight matrix
        
        # weights[index] = rand(Float32, (nodesInEachLayer[index + 1], nodesInEachLayer[index]))
        weights[index] = fill(Float32(1), (nodesInEachLayer[index + 1], nodesInEachLayer[index]))
        weights_change[index] = fill(Float32(0), (nodesInEachLayer[index + 1], nodesInEachLayer[index]))
    end

    # weights[2] = 
    # [
    #     1000 1 1 1;
    #     1 1 1 1;
    #     1 1 1 1;
    #     1 1 1 1;
    # ]

    # activators (does not include input layer):
    activators = Array{Vector, 1}(undef, nLayers)
    for index in eachindex(activators)
        activators[index] = rand(Float32, nodesInEachLayer[index+1])
    end

    # biases
    biases = Array{Vector, 1}(undef, nLayers)
    biases_change = Array{Vector, 1}(undef, nLayers)
    for index in eachindex(biases)
        # biases[index] = rand(Float32, nodesInEachLayer[index + 1])
        biases[index] = fill(Float32(1), nodesInEachLayer[index + 1])
        biases_change[index] = fill(Float32(0), nodesInEachLayer[index + 1])
    end

    activators[1] = relu(weights[1]*inputData[1]+biases[1])
    for a in 2:nLayers
        activators[a] = relu(weights[a]*activators[a-1]+biases[a])
    end
    
    println("\n\ninitial activation:")
    println(activators[3])

    println("desired output:")
    println(outputData[1])

    println("cost:")
    println(cost(activators[3], outputData[1]))
    println("\n")

    for i in 1:100
        for nthExample in 1:nTrainingExamples
            # run data through the network
            activators[1] = relu(weights[1]*inputData[nthExample]+biases[1])
            for a in 2:nLayers
                activators[a] = relu(weights[a]*activators[a-1]+biases[a])
            end

            println(cost(activators[3], outputData[1]))

            # change bias and weight gradients
            weights_change[nLayers] = avgMult*LEARN_RATE*transpose(activators[nLayers-1]*transpose(activators[nLayers]-outputData[nthExample])) #update the last weight on it's own because it's special
            biases_change[nLayers] = avgMult*LEARN_RATE*(activators[nLayers]-outputData[nthExample]) #update the last bias on it's own because it's special too

            # for l in 1:nLayers-1
            #     weightProduct = weights[l+1]
            #     for lp in l+2:nLayers
            #         weightProduct = weights[lp]*weightProduct
            #     end
            #     weights_change[l] -= avgMult*LEARN_RATE*(transpose(weightProduct)*transpose((l == 1 ? inputData[nthExample] : activators[l-1])*transpose(activators[nLayers]-outputData[nthExample])))
            #     biases_change[l] -= avgMult*LEARN_RATE*(transpose(weightProduct)*(activators[nLayers]-outputData[nthExample]))
            # end

            weightProduct = weights[nLayers]

            weights_change[nLayers-1] = avgMult*LEARN_RATE*transpose(weights[nLayers])*transpose(activators[nLayers-2]*transpose(activators[nLayers]-outputData[nthExample]))
            biases_change[nLayers-1] = avgMult*LEARN_RATE*(transpose(weights[nLayers])*(activators[nLayers]-outputData[nthExample]))

            for l in nLayers-2:1
                weightProduct = weightProduct*weights[l+1]
                weights_change[l] = avgMult*LEARN_RATE*(transpose(weightProduct)*transpose((l == 1 ? inputData[nthExample] : activators[l-1])*transpose(activators[nLayers]-outputData[nthExample])))
                biases_change[l] = avgMult*LEARN_RATE*(transpose(weightProduct)*(activators[nLayers]-outputData[nthExample]))
            end
        end

        # println(weights_change[1])
        # println(biases_change[1])
        # println(weights_change[2])
        # println(biases_change[2])
        # println(weights_change[3]) 
        # println(biases_change[3])

        #update biases and weights to change values
        for index in 1:nLayers
            weights[index] = weights[index] - weights_change[index]
            biases[index] = biases[index] - biases_change[index]
        end

        resetToZero(weights_change)
        resetToZero(biases_change)
    end

    println("\n\n")
    
    activators[1] = relu(weights[1]*inputData[1]+biases[1])
    for a in 2:length(activators)
        activators[a] = relu(weights[a]*activators[a-1]+biases[a])
    end
    println("after activation:")
    println(activators[3])

    println("desired output:")
    println(outputData[1])

    println("cost:")
    println(cost(activators[3], outputData[1]))
    println("\n")
    # lastLayer = activators[nLayers]
    # println("activators: $lastLayer")
    for i in 1:nTrainingExamples
        
    end
end

trainNeuralNet(NODES_EACH_LAYER, INPUT_ARRAY, OUTPUT_ARRAY)
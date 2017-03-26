import random
import math
import time

# import layer and neuron classes that I previouly wrote
from layer import Layer
from neuron import Neuron

# Neu = neurons 
# Num = numbers
# Lay = layer
# Dataset = data set
# Der = partial derivative / derivative
# Sqrt = square/squared
# Funct = function
# WRT = with respect to

# the weights and bias numbers are assigned randomly
# this is the class the represent the neural network, the instance can be found in the bottom
# I learned most of my work from:
# https://en.wikipedia.org/wiki/Backpropagation
# https://www.youtube.com/watch?v=LOc_y67AzCA
# https://www.youtube.com/watch?v=h3l4qz76JhQ
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU

inputOutputDataset = [ # inputOutputDataset[0] = input and inputOutputDataset[1] = target output, these are the numbers that are being train
	[0, 0, 0], [0, 0, 0]
]

trainingDataset = [
    [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]],
    [[0.2, 0.3, 0.4], [0.44, 0.55, 0.66]],
    [[0.3, 0.4, 0.5], [0.555, 0.666, 0.777]],
    [[0.4, 0.5, 0.6], [0.6666, 0.7777, 0.8888]],
    [[0.5, 0.6, 0.7], [0.77777, 0.88888, 0.99999]]
]

numTraining = 15000
learningRate = 0.75          # A balance must be found for the argotitom to execute smoothly and when the error goes back to the code, not to mess it up

class NetWork(object):

        ##### Code that runs #####
        def __init__(self, numHiddenNeu):

                self.numInputNeu = len(inputOutputDataset[0])
                self.numHiddenNeu = numHiddenNeu
                self.numOutputNeu = len(inputOutputDataset[1])



                # set layers; there is not one for input because input doesn't have a bias or previous layer to have connection with
                self.hiddenLay = Layer(self.numHiddenNeu)
                self.outputLay = Layer(self.numOutputNeu)

                # set weight to each connection
                self.setWeights()

        #### Methods ######
        def setWeights(self):
                for i in range(len(self.hiddenLay.neurons)):
                        for x in range(self.numInputNeu):
                                self.hiddenLay.neurons[i].weights.append(random.random())

                for i in range(len(self.outputLay.neurons)):
                        for h in range(len(self.hiddenLay.neurons)):
                                self.outputLay.neurons[i].weights.append(random.random())


        def getOutput(self, inputNum):
                    hiddenToOutputs = self.hiddenLay.forward(inputNum)
                    return self.outputLay.forward(hiddenToOutputs)

        # def predictionFunct(self, inputNum):


        def train(self):
                    self.getOutput(inputOutputDataset[0])

                    # output delta
                    derErrorWRTTotalInputOfOutput = [0] * len(self.outputLay.neurons) # or [0] * self.numOutputNeu
                    for i in range(len(self.outputLay.neurons)):
                        derErrorWRTTotalInputOfOutput[i] = self.outputLay.neurons[i].calculateDerErrorTotalInput(inputOutputDataset[1][i])

                    # hidden delta
                    derErrorWRTTotalInputOfHidden = [0] * len(self.hiddenLay.neurons)
                    for i in range(len(self.hiddenLay.neurons)):

                            derErrorWRTHiddenToOutput = 0
                            for h in range(len(self.outputLay.neurons)):
                                    derErrorWRTHiddenToOutput += derErrorWRTTotalInputOfOutput[h] * self.outputLay.neurons[h].weights[i]

                            derErrorWRTTotalInputOfHidden[i] = derErrorWRTHiddenToOutput * self.hiddenLay.neurons[i].derSigmoidFunct()

                    ####### change weights #######
                    # to output
                    for i in range(len(self.outputLay.neurons)):
                            for h in range(len(self.outputLay.neurons[i].weights)):

                                    derErrorWRTWeight = derErrorWRTTotalInputOfOutput[i] * self.outputLay.neurons[i].calculateDerInputWeight(h)

                                    self.outputLay.neurons[i].weights[h] -= learningRate * derErrorWRTWeight

                    # to hidden
                    for i in range(len(self.hiddenLay.neurons)):
                            for h in range(len(self.hiddenLay.neurons[i].weights)):

                                    derErrorWRTWeight = derErrorWRTTotalInputOfHidden[i] * self.hiddenLay.neurons[i].calculateDerInputWeight(h)

                                    self.hiddenLay.neurons[i].weights[h] -= learningRate * derErrorWRTWeight

        def getTotalError(self):
                totalError = 0
                self.getOutput(inputOutputDataset[0])
                for i in range(len(inputOutputDataset[1])):
                        totalError += self.outputLay.neurons[i].calculateError(inputOutputDataset[1][i])
                return totalError


NeuNetwork = NetWork(10) # the network has 10 neurons in the hidden layer

# training section of the algorithm
time1 = time.time()
for x in range(numTraining):
    for index in range(len(trainingDataset)):
        inputOutputDataset = [trainingDataset[index][0], trainingDataset[index][1]]
        NeuNetwork.train()

    print("{0:5d} -> Error: {1:.10f}".format(x + 1, NeuNetwork.getTotalError()))
time2 = time.time()
predictionSet = [ # only inputs to the array; the program will give back the output
    [0.6, 0.3, 0.4]
]
print("Time: {0:.5f}".format(time2 - time1))
print("Prediction 1: {0:.5f}".format(NeuNetwork.getOutput(predictionSet[0])[0]))
print("Prediction 2: {0:.5f}".format(NeuNetwork.getOutput(predictionSet[0])[1]))
print("Prediction 3: {0:.5f}".format(NeuNetwork.getOutput(predictionSet[0])[2]))
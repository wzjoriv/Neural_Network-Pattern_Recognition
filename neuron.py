import random
import math
# this is the class that represent the neurons

class Neuron(object):
        ##### Code that runs #####
        def __init__(self, bias):
            self.bias = bias
            self.weights = []

        def sigmoidFunct(self, numInput): # non-linear and differential activation function
            return 1 / (1 + math.exp(-numInput))

        def derSigmoidFunct(self): # derivative of the activation function
            return self.output * (1 - self.output)

        def calculateDevErrorOutput(self, target): #derivative of the error after output neuron
            return -(target - self.output)

        def calculateOutput(self, inputs): # total output from a neuron
            self.inputs = inputs
            self.output = self.sigmoidFunct(self.calculateTotalWRTInput())
            return self.output

        def calculateTotalWRTInput(self): # sum of all connection with their weights and bias number added
            total = 0
            for i in range(len(self.inputs)):
                total += self.inputs[i] * self.weights[i]
            return total + self.bias

        def calculateError(self, target): # mean value method
            return 0.5 * ((target - self.output) ** 2)

        def calculateDerErrorTotalInput(self, target): # derivative of total error after input
            return self.calculateDevErrorOutput(target) * self.derSigmoidFunct()

        def calculateDerInputWeight(self, index):
            return self.inputs[index]
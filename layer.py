import random
import math
from neuron import Neuron
# this is the class for the layers

class Layer(object):
		##### Code that runs #####
		def __init__(self, numNeu):
				self.bias = random.random()

				self.neurons = []
				for i in range(numNeu): # add bias number to every neuron, must be added here because every neuron in each layer has the same bias number
						self.neurons.append(Neuron(self.bias))

		def forward(self, inputNum):
				outputs = []
				for neuron in self.neurons:
						outputs.append(neuron.calculateOutput(inputNum))
				return outputs

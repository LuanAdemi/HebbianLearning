import numpy as np
from typing import Callable
from scipy.stats import logistic

# TODO: DEBUG and see if it works, future Luan.
# Thanks past Luan. This code is crap. Backprop not working... D:

class HebbianNeuron():
    """
    A simple HebbianNeuron class.
    
    Inputs:
        weights -> nd-Array
        bias -> float
        activation -> Callable
        
    Attributes:
        weights -> nd-Array
        bias -> float
        activation -> Callable
        
    Functions:
        prop(inputs -> nd-Array) -> propagates through the neuron
    
    """
    def __init__(self, 
                 weights:np.ndarray = np.zeros(10), 
                 bias:float = 0, 
                 activation:Callable = None):
        
        # our neuron storage
        self.weights = weights
        self.bias = bias
        
        # in case we want to use activation
        self.activation = activation 
    
    def __repr__(self):
        if self.activation is not None:
            return f"HebbianNeuron(weights={self.weights}, bias={self.bias}, activation={self.activation.__name__})"
        else:
            return f"HebbianNeuron(weights={self.weights}, bias={self.bias}, activation=None)"
        
    def prop(self, inputs:np.ndarray) -> np.ndarray:
        assert self.weights.shape[0] == inputs.shape[0], f"The input {inputs.shape} could not be broadcasted with the weights {self.weights.shape}."
        
        # Y = sum(weight_i * x_i) + b
        out = (self.weights * inputs).sum() + self.bias
        
        # activation
        if self.activation is not None:
            out = self.activation(out)
            
        return logistic.cdf(out)
    
    def update(self, ruleParameter:tuple, output:float ,prevOutputs:np.ndarray, alpha:float):
        
        # Update the weights using Hebbian Plasticity
        # PreNeuron (--weight--> PostNeuron(This Neuron))
        #                ^ We want to adjust this based on the output 
        #                  of the pre and post neurons.
        
        A, B, C, D = ruleParameter # the rule parameters
        
        """
                   o1                 o1        
                   o2                 o2
        dW = A  *  o3  *  O  +  B  *  o3  +  C  *  O  +  D
                   o4                 o4
                   o5                 o5
        """
        #output = self.prop(prevOutputs)
        self.weights += alpha * (A*prevOutputs*output + B*prevOutputs + C*output + D)
    
class HebbianLayer():
    """
    A simple Hebbian Layer class.
    
    Inputs:
        prevSize -> int
        size -> int
        weights -> nd-Array
        bias -> nd-Array
        activations -> list
        
    Attributes:
        prevSize -> int
        size -> int
        weights -> nd-Array
        bias -> float
        activations -> list
        neurons -> list
        
    Functions:
        prop(inputs -> nd-Array) -> propagates through the Layer
    
    """
    def __init__(self,
                prevSize:int = 1,
                size:int = 1,
                weights:np.ndarray = None,
                bias:np.ndarray = None,
                activations:list = None
                ):
        
        self.prevSize = prevSize
        self.size = size
        self.weights = weights
        self.bias = bias
        self.activations = activations
        
        # if weights not defined: randomly initialize them
        if weights is None:
            self.weights = np.random.rand(self.size, self.prevSize)
            
        # if bias not defined: set them to zero
        if bias is None:
            self.bias = np.zeros(self.size)
            
        # if activations not defined: set them to None
        if activations is None:
            self.activations = [None for _ in range(self.size)]
        
        # create the neurons
        self.neurons = []
        
        for i in range(self.size):
            self.neurons.append(HebbianNeuron(self.weights[i], self.bias[i], self.activations[i]))
    
    def __repr__(self):
        return f"HebbianLayer(shape=({self.prevSize}, {self.size}))"
    
    def prop(self, inputs:np.ndarray) -> np.ndarray:
        out = []

        for neuron in self.neurons:
            out.append(neuron.prop(inputs))
        return np.array(out)

class HebbianNetwork():
    """
    A simple Hebbian Network.
    
    Inputs:
        shape -> tuple
        weights -> nd-Array
        bias -> nd-Array
        activations -> list
        
    Attributes:
        shape -> tuple
        weights -> nd-Array
        bias -> nd-Array
        activations -> list
        layers -> list
        
    Functions:
        prop(inputs -> nd-Array) -> propagates through the Layers
    
    """
    def __init__(self,
                shape:tuple = (1, 1),
                weights:np.ndarray = None,
                bias:np.ndarray = None,
                activations:list = None):
        
        self.shape = shape
        self.weights = weights
        self.bias = bias
        self.activations = activations
        
        # if weights not defined: randomly initialize them
        if weights is None:
            self.weights = np.zeros(len(self.shape),dtype=object)
            
        # if bias not defined: set them to zero
        if bias is None:
            self.bias = np.zeros(len(self.shape),dtype=object)
            
        
        self.layers = []
        
        # create the first layer
        l0 = HebbianLayer(self.shape[0], self.shape[0])
        self.layers.append(l0)
        self.weights[0] = l0.weights
        
        for i in range(1, len(self.shape)):
            layer = HebbianLayer(self.shape[i-1], self.shape[i])
            self.layers.append(layer)
            self.weights[i] = layer.weights
            
            
    def __repr__(self):
        rep = f"HebbianNetwork: X({self.layers[0].prevSize})"
        for i in range(len(self.shape)):
            rep += f" -> HebbianLayer({self.layers[i].size})"
        rep += f" -> Y({self.layers[-1].size})"
        return rep
    
    def prop(self, inputs:np.ndarray):
        assert inputs.shape[0] == self.layers[0].prevSize, "The input layer size doesn't match the passed vector."
        # propagate through the layers
        x = inputs
        for layer in self.layers:
            x = layer.prop(x)
            
        return x  
    
    def backwards(self, inputs:np.ndarray, rules:tuple, alpha:float):
        outs = []
        
        # store the actual inputs
        outs.append(inputs)
        
        # collect the outputs of each layer
        x = inputs
        for layer in self.layers:
            x = layer.prop(x)
            outs.append(x)
        
        # 
        for l, layer in enumerate(self.layers):
            # Layers
            for n, neuron in enumerate(layer.neurons):
                # Neurons
                # first element of outs is the input vector
                prevOutputs = outs[l]
                output = outs[l+1][n]
                neuron.update(rules, output, prevOutputs, alpha)
import numpy as np;

class perceptron(object):
    """description of class"""

    def __init__(self, **kwargs):
        self.synaptic_weigths = None;
        self.outputs = None;
        self.error = None;
        return super().__init__(**kwargs);
    
    def setSeed(self, seed):
        np.random.seed(seed);

    def sigmoid_derivative(self, x):
        return x * (1 - x);

    def setAmountVariables(self, num):
        self.synaptic_weigths = 2 * np.random.random((num,1)) - 1;

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x));

    def trainPerceptron(self, inputs, outputs, times):
        for i in range(times):
            input_layer = inputs;
            self.outputs = self.sigmoid(np.dot(input_layer, self.synaptic_weigths));
            self.error = outputs - self.outputs;
            adjustments = self.error * self.sigmoid_derivative(self.outputs);
            self.synaptic_weigths += np.dot(input_layer.T, adjustments);

    def process(self, inputs):
        inputs = inputs.astype(float);
        self.outputs = self.sigmoid(np.dot(inputs, self.synaptic_weigths));
        return self.outputs;
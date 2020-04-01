import numpy as np;
from perceptron import perceptron;

def trainingNeuron(trainingTimes):
    neuron = perceptron();
    training_inputs = np.array([[0,0,1],
                               [1,1,1],
                               [1,0,1],
                               [0,1,1]]);

    training_outputs = np.array([[0,1,1,0]]).T;

    neuron.setSeed(1);
    neuron.setAmountVariables(3);

    print("Synaptic weigths:", neuron.synaptic_weigths);

    neuron.trainPerceptron(training_inputs,training_outputs,trainingTimes);

    print("Synaptic weigths after training:", neuron.synaptic_weigths);
    print("Trained outputs:", neuron.outputs);
    return neuron;


def main():
    trainingTimes = int(input("number of training times: "));
    neuron = trainingNeuron(trainingTimes);
    ToForecast_inputs = np.array([[1,0,0]]);
    Forecast_outputs = neuron.process(ToForecast_inputs);
    print("Forecast of : [1,0,0]", "Forecast outputs:", neuron.outputs);
    values = np.array(input("Type 3 numbers 1 or 0 separated by a coma: ").split(","));
    Forecast_outputs = neuron.process(values);
    print("Forecast of : " + str(values), "Forecast outputs:", neuron.outputs);
    

if __name__ == "__main__":
    main()
//Set this page to only be included once and import the vector library
#pragma once
#include<vector>

//Define the struct that bundles one neuron's data together
struct neuron{
    std::vector<double>weights;
    double bias;
    std::vector<double>inputs;
    double output;
};

//Declare the forward pass function to calculate the output of a neuron
double neuronForward(
    neuron&n,
    std::vector<double>newInputs
);

//Declare the updateNeuron function to adjust weights and bias based on error and learning rate
void updateNeuron(
    neuron&n,
    double error,
    double learningRate
);
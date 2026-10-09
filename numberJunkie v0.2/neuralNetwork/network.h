//Set this page to only be included once and import the vector library
#pragma once
#include<vector>
#include"layer.h"

//Declare the network struct that bundles all layers together
struct network{
    std::vector<layer>layers;
};

//Declare the constructor methods to form the network
neuron neuronBuilder(
    bool hasWeights,
    bool hasBias,
    std::vector<double>inputs,
    std::vector<double>weights,
    double bias,
    int numInputs,
    std::mt19937&rng,
    std::uniform_real_distribution<double>&dist
);
layer layerBuilder(
    std::vector<bool>hasWeights,
    std::vector<bool>hasBias,
    std::vector<std::vector<double>>weights,
    std::vector<double>bias,
    int numInputs,
    int numNeurons,
    std::mt19937&rng,
    std::uniform_real_distribution<double>&dist
);
network networkBuilder(
    std::string AIDNA,
    int numInputs,
    int numNeurons,
    int numLayers,
    int numOutputs
);

//Declare the forwardPass function to calculate the outputs of the network
std::vector<double>forwardPass(
    network&n,
    std::vector<double>inputs
);

//Declare the backProp function to update every layer's neurons' weights and bias in the network
void backProp(
    network&n,
    std::string AIDNA,
    double oldCapital,
    double newCapital,
    double learningRate
);

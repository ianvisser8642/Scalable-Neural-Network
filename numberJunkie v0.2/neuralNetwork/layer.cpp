//Importing the necessary libraries and header files
#include<vector>
#include"neuron.h"
#include"layer.h"

//Define the layerForward function to calculate the outputs of every neuron in a layer
std::vector<double>layerForward(
    layer&l,
    std::vector<double>inputs
){
    std::vector<double>outputs;

    //Loop through each neuron in the layer and calculate its output
    for(int i=0;i<l.neurons.size();i++){
        double neuronOutput=neuronForward(l.neurons[i],inputs);
        outputs.push_back(neuronOutput);
    }
    return outputs;
}

//Define the layerUpdate function to update every neuron's weights and bias in a layer
void updateLayer(
    layer&l,
    std::vector<double>errors,
    double learningRate
){
    //Loop through each neuron in the layer and update its weights and bias
    for(int i=0;i<l.neurons.size();i++){
        updateNeuron(l.neurons[i],errors[i],learningRate);
    }
}
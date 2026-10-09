//Set this page to only be included once and import the vector library
#pragma once
#include<vector>
#include"neuron.h"

//Define the struct that bundles one layer's neurons together
struct layer{
    std::vector<neuron>neurons;
};

//Declare the layerForward function to calculate the outputs of every neuron in a layer
std::vector<double>layerForward(
    layer&l,
    std::vector<double>inputs
);

//Declare the layerUpdate function to update every neuron's weights and bias in a layer
void updateLayer(
    layer&l,
    std::vector<double>errors,
    double learningRate
);
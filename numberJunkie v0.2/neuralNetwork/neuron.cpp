//Importing the necessary libraries and header files
#include<vector>
#include"methods.h"
#include"neuron.h"

//Define the forward pass function to calculate the output of a neuron
double neuronForward(
    neuron&n,
    std::vector<double>newInputs
){
    n.inputs=newInputs;
    double result=0.0;
    for(int i=0;i<n.inputs.size();i++){
        result+=n.inputs[i]*n.weights[i];
    }
    result=result+n.bias;
    n.output=findSigmoid(result);
    return n.output;
}

//Define the updateNeuron function to adjust weights and bias based on error and learning rate
void updateNeuron(
    neuron&n,
    double error,
    double learningRate
){
    double sigmoidDerivative=findSigmoidDerivative(n.output);

    //Calculate the policy gradient for each weight and update it accordingly
    for(int i=0;i<n.weights.size();i++){
        double policyGradient=error*sigmoidDerivative*n.inputs[i];
        n.weights[i]+=learningRate*policyGradient;
    }
    n.bias+=learningRate*error*sigmoidDerivative;
}
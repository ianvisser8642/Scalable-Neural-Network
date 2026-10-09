//Importing the necessary libraries
#include<vector>
#include<cmath>
#include"methods.h"

//Define the sigmoid activation function to get a value between 0 and 1
double findSigmoid(double result){
    double confidence=1.0/(1.0+exp(-result));
    return confidence;
}

//Caclulate D(result)/D(sum) to get the derivative of the sigmoid function
double findSigmoidDerivative(double sigmoid){
    double sigmoidDerivative=sigmoid*(1-sigmoid);
    return sigmoidDerivative;
}

//Calculate the error between the predicted loss/profit and the actual investment
double findError(
    double newCapital,
    double oldCapital
){
    double error=(newCapital-(oldCapital*1.01))/oldCapital;
    return error;
}
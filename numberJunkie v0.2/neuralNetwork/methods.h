//Set this page to only be included once and import the vector library
#pragma once
#include<vector>

//Declare methods to be called from the methods.cpp file
double findSigmoid(double result);
double findSigmoidDerivative(double sigmoid);
double findError(
    double newCapital,
    double oldCapital
);
//Importing the necessary libraries and header files
#include<vector>
#include<random>
#include<fstream>
#include"layer.h"
#include"network.h"
#include"methods.h"

//Build the constructor methods to form the network
neuron neuronBuilder(
    std::vector<bool> hasWeights,
    bool hasBias,
    std::vector<double>weights,
    double bias,
    int numInputs,
    std::mt19937&rng,
    std::uniform_real_distribution<double>&dist
){
    //Generate random values for the missing weights and biases
    weights.resize(numInputs);
    for(int i=0;i<numInputs;i++){
        if(!hasWeights[i]){
            weights[i]=dist(rng);
        }
        else{}
    }
    if(!hasBias){
        bias=dist(rng);
    }
    else{}
    
    //Define the neuron
    neuron n;
    n.weights=weights;
    n.bias=bias;
    return n;
}
layer layerBuilder(
    std::vector<std::vector<bool>>hasWeights,
    std::vector<bool>hasBias,
    std::vector<std::vector<double>>weights,
    std::vector<double>bias,
    int numInputs,
    int numNeurons,
    std::mt19937&rng,
    std::uniform_real_distribution<double>&dist
){
    //Iterate through the number of neurons designated to 1 layer and pass the build information to neuronBuilder for each neuron
    layer l;
    for(int i=0;i<numNeurons;i++){
        l.neurons.push_back(neuronBuilder(hasWeights[i],hasBias[i],weights[i],bias[i],numInputs,rng,dist));
    }
    return l;
}
network networkBuilder(
    std::string AIDNA,
    int numInputs,
    int numNeurons,
    int numLayers,
    int numOutputs
){
    //Instantiate the network structure and specify the random number range
    network n;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double>dist(-1.0,1.0);

    //Calculate how many numbers the file should contain if it is complete
    int expectedNumbers=(numNeurons*(numInputs+1))
        +((numLayers-1)*numNeurons*(numNeurons+1))
        +(numOutputs*(numNeurons+1));

    //Read every number in the file into one flat list
    std::vector<double>allNumbers;
    std::ifstream inFile("storage/"+AIDNA+".txt");
    if(inFile){
        double num;
        while(inFile>>num){
            allNumbers.push_back(num);
        }
    }
    inFile.close();

    //Check if the file had exactly the right amount of data
    bool fileIsValid=(allNumbers.size()==expectedNumbers);

    //Keep track of where we are in the flat list as we carve it up
    int position=0;

    //Loop through each layer that needs to be built, hidden layers first then the output layer
    for(int layerIndex=0;layerIndex<numLayers+1;layerIndex++){

        //Work out how many neurons and how many inputs per neuron this layer needs
        int neuronsThisLayer=(layerIndex<numLayers)?numNeurons:numOutputs;
        int inputsThisLayer=(layerIndex==0)?numInputs:numNeurons;

        //Prepare the per neuron containers that layerBuilder expects
        std::vector<std::vector<bool>>hasWeights;
        std::vector<bool>hasBias;
        std::vector<std::vector<double>>weights;
        std::vector<double>bias;

        //Fill those containers, either from the file or with empty placeholders
        for(int neuronIndex=0;neuronIndex<neuronsThisLayer;neuronIndex++){
            if(fileIsValid){
                std::vector<double>neuronWeights;
                for(int i=0;i<inputsThisLayer;i++){
                    neuronWeights.push_back(allNumbers[position]);
                    position++;
                }
                double neuronBias=allNumbers[position];
                position++;

                weights.push_back(neuronWeights);
                bias.push_back(neuronBias);
                hasWeights.push_back(std::vector<bool>(inputsThisLayer, true));
                hasBias.push_back(true);
            }
            else{
                weights.push_back(std::vector<double>());
                bias.push_back(0.0);
                hasWeights.push_back(std::vector<bool>(inputsThisLayer, true));
                hasBias.push_back(false);
            }
        }
        //Build this layer using the containers just filled and add it to the network
        layer l=layerBuilder(hasWeights,hasBias,weights,bias,inputsThisLayer,neuronsThisLayer,rng,dist);
        n.layers.push_back(l);
    }
    return n;
}
//Define the forwardPass function to calculate the confidence the AI has for each choice
std::vector<double>forwardPass(
    network&n,
    std::vector<double>inputs
){
    std::vector<double>outputs=layerForward(n.layers[0],inputs);

    //Loop through each layer in the network and calculate its output passing it as input for the next layer
    for(int i=1;i<n.layers.size();i++){
        outputs=layerForward(n.layers[i],outputs);
    }
    return outputs;
}
//Define the backProp function to update all the weights and biases
void backProp(
    network&n,
    std::string AIDNA,
    double error,
    double learningRate
){
    //Declare the errors vector, one entry per layer
    std::vector<std::vector<double>>errors(n.layers.size());

    //Identify the output layer's index
    int o=n.layers.size()-1;

    //Size and fill the output layer's error vector
    errors[o].resize(
        n.layers[o].neurons.size()
    );
    for(int i=0;i<n.layers[o].neurons.size();i++){
        errors[o][i]=error;
    }
    //Find the confidence of the largest output
    int highestOutput=0;
    for(int i=1;i<n.layers[o].neurons.size();i++){
        if(n.layers[o].neurons[i].output>n.layers[o].neurons[highestOutput].output){
            highestOutput=i;
        }
    }
    //Calculate confidence adjustment
    double adjustment=std::abs(error)*n.layers[o].neurons[highestOutput].output;

    //Calculate the combined confidence of the other outputs
    double otherTotal=0.0;
    for(int i=0;i<n.layers[o].neurons.size();i++){
        if(i!=highestOutput){
            otherTotal+=n.layers[o].neurons[i].output;
        }
    }
    //Calculate the error for every output neuron based on confidence adjustment
    for(int i=0;i<n.layers[o].neurons.size();i++){
        double confidence=n.layers[o].neurons[i].output;
        if(i==highestOutput){
            if(error>0){
                errors[o][i]=adjustment;
            }
            else if(error<0){
                errors[o][i]=-adjustment;
            }
            else{
                errors[o][i]=0.0;
            }
        }
        else{
            double proportionalAdjustment=adjustment*(confidence/otherTotal);
            if(error>0){
                errors[o][i]=-proportionalAdjustment;
            }
            else if(error<0){
                errors[o][i]=proportionalAdjustment;
            }
            else{
                errors[o][i]=0.0;
            }
        }
    }
    //Calculate the error for every hidden layer working backwards
    for(int i=o-1;i>=0;i--){

        //Size the error vector for this layer
        errors[i].resize(
            n.layers[i].neurons.size()
        );
        //Calculate the error for every neuron in this layer
        for(int j=0;j<n.layers[i].neurons.size();j++){
            double neuronError=0.0;

            //Check the neurons in the next layer
            for(int l=0;l<n.layers[i+1].neurons.size();l++){

                //Add this neuron's contribution to the error
                neuronError+=errors[i+1][l]*n.layers[i+1].neurons[l].weights[j];
            }
            //Store this neuron's error
            errors[i][j]=neuronError;
        }
    }
    //Update every layer using its calculated error
    for(int i=0;i<n.layers.size();i++){
        updateLayer(n.layers[i],errors[i],learningRate);
    }
    //Open or generate the .txt file where the weights and biases will live
    std::ofstream outFile("storage/"+AIDNA+".txt");
    
    //Iterate through all the layers
    for(int i=0;i<n.layers.size();i++){

        //Iterate through all the neurons per layer
        for(int j=0;j<n.layers[i].neurons.size();j++){

            //Save every weight belonging to this neuron
            for(int l=0;l<n.layers[i].neurons[j].weights.size();l++){
                outFile<<n.layers[i].neurons[j].weights[l]<<" ";
            }
            //Save the bias after all the neurons weights
            outFile<<n.layers[i].neurons[j].bias<<"\n";
        }
    }
    //Close file
    outFile.close();
}

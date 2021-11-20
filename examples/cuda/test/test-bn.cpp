#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <iomanip>

#include "agent/doomagent.h"


void sleep(unsigned int time){
    std::this_thread::sleep_for(std::chrono::milliseconds(time));
}

using namespace vizdoom;

AgentDebug doomDebug(2);

std::vector<vizdoom::BufferPtr> initInput()
{
    std::vector<vizdoom::BufferPtr> screenBufs;
    screenBufs.push_back(vizdoom::BufferPtr(new std::vector<uint8_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 ,30, 31, 32, 33, 34, 35,	36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
                    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,	72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
                    90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107})));
    screenBufs.push_back(vizdoom::BufferPtr(new std::vector<uint8_t>({0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11,
		12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23,
		24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35})));
    screenBufs.push_back(vizdoom::BufferPtr(new std::vector<uint8_t>({0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0, 0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0, 
                                                   0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0, 0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0, 
                                                   0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0, 0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 2, 0, 2, 0, 1, 2, 1, 0})));
    screenBufs.push_back(vizdoom::BufferPtr(new std::vector<uint8_t>({0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
                                                   1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 
                                                   1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1})));
    screenBufs.push_back(vizdoom::BufferPtr(new std::vector<uint8_t>({0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 2, 0, 2, 0, 0, 2, 1, 0, 2, 2, 
                                                                      1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 2, 1, 1, 2, 2,
                                                                      2, 0, 0, 2, 0, 1, 2, 0, 2, 2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2,
                                                                      0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0, 1, 2, 0, 2, 0, 0, 2, 1, 0, 2, 2})));
    //screenBufs.push_back(new std::vector<uint8_t>({6, 6, 3, -4, -8, 3, -2, -8, 1, -4, 1, -3, -4, -5, 7, 5, -6, 4, -1, -1, 2, 1, -2, 8, 8, -5, -1, -8, 1, 7, -1, -1, 7, 3, 9, -2, 
    //                                               -4, 4, -2, -4, -4, -7, 1, 7, 6, -4, 7, -4, 5, 9, 1, -8, 1, -4, -3, 1, 9, -3, 2, -2, 0, 5, -5, 6, 5, -9, -6, -8, 3, 2, -1, 0, 
    //                                               8, 7, 7, -5, 3, 6, 5, 3, 4, 7, 9, -7, -9, 1, -9, 5, -5, -3, 2, 4, -2, 4, 8, 5, -8, 9, -1, 1, 0, 3, -2, 9, 0, -9, -3, 8}));
    
    return screenBufs;
}

void initWeights(ActionValueNetwork *network)
{
    std::vector<float> conv1Weights = {	1, 4, 1, 2, 3, 4, 3, 2, 3, 4, 1, 2, 1,
										1, 4, 1, 2, 3, 4, 3, 2, 3, 4, 1, 2, 1,
										1, 4, 1, 2, 3, 4, 3, 2, 3, 4, 1, 2, 1};
	std::vector<float> conv2Weights = {	1, 4, 1, 2, 3, 4, 3, 2, 3, 4, 1, 2, 1,
										1, 4, 1, 2, 3, 4, 3, 2, 3, 4, 1, 2, 1,
										1, 4, 1, 2, 3, 4, 3, 2, 3, 4, 1, 2, 1,
										1, 4, 1, 2, 3, 4, 3, 2, 3, 4, 1, 2, 1};
	std::vector<float> dense1Weights = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1};
	std::vector<float> dense2Weights = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1,
										1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1};
	std::vector<float> dense3Weights = {1, 2, 3, 4, 1, 1,
										1, 2, 3, 4, 1, 1,
										1, 2, 3, 4, 1, 1,
										1, 2, 3, 4, 1, 1,
										1, 2, 3, 4, 1, 1,
										1, 2, 3, 4, 1, 1,
										1, 2, 3, 4, 1, 1,
										1, 2, 3, 4, 1, 1};

    std::vector<float> conv1Gamma = {1, 1, 1};
    std::vector<float> conv2Gamma = {1, 1, 1, 1};
    std::vector<float> dense1Gamma = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> dense2Gamma = {1, 1, 1, 1, 1};
    std::vector<float> conv1Beta = {0, 0, 0};
    std::vector<float> conv2Beta = {0, 0, 0, 0};
    std::vector<float> dense1Beta = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> dense2Beta = {0, 0, 0, 0, 0};
 
    auto layers = network->getLayers();
	auto iter = layers.begin();
	iter++;
	((Conv3dLayer*)(*iter))->setWeights(conv1Weights, conv1Gamma, conv1Beta);
	iter++;
	iter++;
	((Conv3dLayer*)(*iter))->setWeights(conv2Weights, conv2Gamma, conv2Beta);
	iter++;
	iter++;
	((DenseLayer*)(*iter))->setWeights(dense1Weights, dense1Gamma, dense1Beta);
	iter++;
	((DenseLayer*)(*iter))->setWeights(dense2Weights, dense2Gamma, dense2Beta);
	iter++;
	((OutputLayer*)(*iter))->setWeights(dense3Weights);		
}

void printLayers(ActionValueNetwork *network)
{
    auto layers = network->getLayers();    
    for(auto it=layers.begin(); it!=layers.end(); ++it)
	{
		if((*it)->layerType() == NetworkLayer::INPUT)
		{
			std::cout << "InputLayer" << std::endl;			            
            std::vector<unsigned> layerSize = (*it)->layerSize();

            std::vector<float> activations = (*it)->activations();
		    for(int i=0; i<activations.size(); ++i)
            {
			    std::cout << activations[i] << " ";
                if((i+1) % (layerSize[1]*layerSize[2]*layerSize[3]) == 0)
                    std::cout << std::endl;
            }
		    std::cout << std::endl;

		}
		else if((*it)->layerType() == NetworkLayer::CONV)
		{
			std::cout << "Conv3dLayer" << std::endl;	
            std::vector<unsigned> layerSize = (*it)->layerSize();

			std::vector<float> weights = ((Conv3dLayer*)(*it))->weights();
			for(int i=0; i<weights.size(); ++i)
				std::cout << (int)weights[i] << " ";
			std::cout << std::endl;
            std::vector<float> gammas = ((Conv3dLayer*)(*it))->gammas();
			for(int i=0; i<gammas.size(); ++i)
				std::cout << (int)gammas[i] << " ";
			std::cout << std::endl;
			std::vector<float> betas = ((Conv3dLayer*)(*it))->betas();
			for(int i=0; i<betas.size(); ++i)
				std::cout << (int)betas[i] << " ";
			std::cout << std::endl << std::endl;

			std::vector<float> dotProducts = ((Conv3dLayer*)(*it))->dotProducts();
            //std::cout << layerSize[1] << " " << layerSize[2] << " " << layerSize[3] << std::endl;
			for(int i=0; i<dotProducts.size(); ++i)
            {
				std::cout << dotProducts[i] << " ";
                if((i+1) % (layerSize[1]*layerSize[2]*layerSize[3]) == 0)
                    std::cout << std::endl;
            }
			std::cout << std::endl;

			std::vector<float> means = ((Conv3dLayer*)(*it))->means();
			for(int i=0; i<means.size(); ++i)
				std::cout << means[i] << " ";
			std::cout << std::endl;
            std::cout << "There is some error here because of float type low precission with big numbers and many decimal places:" << std::endl;
			std::vector<float> vars = ((Conv3dLayer*)(*it))->vars();
			for(int i=0; i<vars.size(); ++i)
				std::cout << std::fixed << vars[i] << " ";
			std::cout << std::endl << std::endl;
			//for(int i=0; i<vars.size(); ++i)
			//	std::cout << std::fixed << vars[i]/100 << " ";
			//std::cout << std::endl << std::endl;
			
            std::vector<float> zHat = ((Conv3dLayer*)(*it))->zHat();
			for(int i=0; i<zHat.size(); ++i)
            {
			//	std::cout << std::fixed << std::setprecision(16) << zHat[i] << " ";
				std::cout << zHat[i] << " ";
                if((i+1) % (layerSize[1]*layerSize[2]*layerSize[3]) == 0)
                    std::cout << std::endl;
            }
			std::cout << std::endl;

			std::vector<float> normed = ((Conv3dLayer*)(*it))->normed();
			for(int i=0; i<normed.size(); ++i)
            {
				std::cout << normed[i] << " ";
                if((i+1) % (layerSize[1]*layerSize[2]*layerSize[3]) == 0)
                    std::cout << std::endl;
            }
			std::cout << std::endl;

            std::vector<float> activations = (*it)->activations();
		    for(int i=0; i<activations.size(); ++i)
            {
			    std::cout << activations[i] << " ";
                if((i+1) % (layerSize[1]*layerSize[2]*layerSize[3]) == 0)
                    std::cout << std::endl;
            }
		    std::cout << std::endl;
		}
		else if((*it)->layerType() == NetworkLayer::MAX_POOL)
		{
			std::cout << "Pool3dLayer" << std::endl;
            std::vector<unsigned> layerSize = (*it)->layerSize();
            
            std::vector<float> activations = (*it)->activations();
		    for(int i=0; i<activations.size(); ++i)
            {
			    std::cout << activations[i] << " ";
                if((i+1) % (layerSize[1]*layerSize[2]*layerSize[3]) == 0)
                    std::cout << std::endl;
            }
		    std::cout << std::endl;
		}
		else if((*it)->layerType() == NetworkLayer::FC)
		{
			std::cout << "DenseLayer" << std::endl;
			std::vector<unsigned> layerSize = (*it)->layerSize();

			std::vector<float> weights = ((DenseLayer*)(*it))->weights();
			for(int i=0; i<weights.size(); ++i)
				std::cout << (int)weights[i] << " ";
			std::cout << std::endl;
            std::vector<float> cachedWeights = ((DenseLayer*)(*it))->cachedWeights();
			for(int i=0; i<cachedWeights.size(); ++i)
				std::cout << (int)cachedWeights[i] << " ";
			std::cout << std::endl;
            std::vector<float> gammas = ((DenseLayer*)(*it))->gammas();
			for(int i=0; i<gammas.size(); ++i)
				std::cout << gammas[i] << " ";
			std::cout << std::endl;
            std::vector<float> betas = ((DenseLayer*)(*it))->betas();
			for(int i=0; i<betas.size(); ++i)
				std::cout << betas[i] << " ";
			std::cout << std::endl << std::endl;

            std::vector<float> dotProducts = ((DenseLayer*)(*it))->dotProducts();
			for(int i=0; i<dotProducts.size(); ++i)
            {
				std::cout << dotProducts[i] << " ";
                if((i+1) % layerSize[1] == 0)
                    std::cout << std::endl;
            }
			std::cout << std::endl;

            std::vector<float> means = ((DenseLayer*)(*it))->means();
			for(int i=0; i<means.size(); ++i)
				std::cout << means[i] << " ";
			std::cout << std::endl;
            std::vector<float> vars = ((DenseLayer*)(*it))->vars();
			for(int i=0; i<vars.size(); ++i)
				std::cout << vars[i] << " ";
			std::cout << std::endl << std::endl;

            std::vector<float> zHat = ((DenseLayer*)(*it))->zHat();
			for(int i=0; i<zHat.size(); ++i)
            {
				std::cout << zHat[i] << " ";
                if((i+1) % layerSize[1] == 0)
                    std::cout << std::endl;
            }
			std::cout << std::endl;
            std::vector<float> normed = ((DenseLayer*)(*it))->normed();
			for(int i=0; i<dotProducts.size(); ++i)
            {
				std::cout << normed[i] << " ";
                if((i+1) % layerSize[1] == 0)
                    std::cout << std::endl;
            }
			std::cout << std::endl;          
            std::vector<float> activations = (*it)->activations();
		    for(int i=0; i<activations.size(); ++i)
            {
			    std::cout << activations[i] << " ";
                if((i+1) % layerSize[1] == 0)
                    std::cout << std::endl;
            }
		    std::cout << std::endl;
		}	
		else if((*it)->layerType() == NetworkLayer::OUTPUT)
		{
			std::cout << "OutputLayer (may give incorect but close results for great values because of float type low precision)" << std::endl;
			std::vector<unsigned> layerSize = (*it)->layerSize();
			
            std::vector<float> weights = ((OutputLayer*)(*it))->weights();
			for(int i=0; i<weights.size(); ++i)
				std::cout << (int)weights[i] << " ";
			std::cout << std::endl << std::endl;

            std::vector<float> cachedWeights = ((OutputLayer*)(*it))->cachedWeights();
			for(int i=0; i<cachedWeights.size(); ++i)
				std::cout << (int)cachedWeights[i] << " ";
			std::cout << std::endl << std::endl;

			std::vector<float> dotProducts = ((OutputLayer*)(*it))->dotProducts();
			for(int i=0; i<dotProducts.size(); ++i)
            {
				std::cout << dotProducts[i] << " ";
                if((i+1) % layerSize[1] == 0)
                    std::cout << std::endl;
            }
			std::cout << std::endl;
		    
            std::vector<float> activations = (*it)->activations();
		    for(int i=0; i<activations.size(); ++i)
            {
			    std::cout << activations[i] << " ";
                if((i+1) % layerSize[1] == 0)
                    std::cout << std::endl;
            }
		    std::cout << std::endl;
		}
		//std::vector<float> activations = (*it)->activations();
		//for(int i=0; i<activations.size(); ++i)
		//	std::cout << (int)activations[i] << " ";
		//std::cout << std::endl;
        std::cout << "------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    }
        
    //double f = 63692.449219;
    //float f = 636924400.0f;
	//for(int i=0; i<1000; ++i)
	//	std::cout << std::fixed << std::setprecision(6) << (f+i)/10000 << std::endl;
	//std::cout << std::endl;
    //double d = 6369244.75;
    //float f1 = d;
    //float f2 = f1/100;
    //std::cout << d << " " << f1 << " " << f2 << std::endl;
}

int main() 
{
        // Lunar lander agentConf = {x, 50000, 4, 8, x, x, x}.
        //AgentConfig agentConf = {8, 1000, 1, 1, 7, 1, 1};

        // My old configuration.
        //NetworkConfig netConf = {{160, 120, 3}, {8, 4}, {4, 2}, {1, 1}, {RELU, RELU}, {4, 2}, {1, 1}, {50, 20}, 8};                             // 2GB RAM, 8sec per st>

	//numActions, replayBufferMaxSize, numReplay, numMiniBatch, seed, discount, tau
    unsigned numReplay = 1;
    unsigned numMiniBatch = 4;
    unsigned bufferSize = 5;
	AgentConfig agentConf = {3, 100, numReplay, numMiniBatch, 1, 1};
	//inputSize, numFilters, filterDim, filterStride, activations, poolDim, poolStride, numHiddenUnits, numActions
	NetworkConfig netConf = {{6, 6, 3}, {3, 4}, {2, 2}, {1, 1}, {TANH, TANH}, {2, 2}, {1, 1}, {10, 5}, 3};
	OptimizerConfig optConf = {0.1, 0.2, 0.3, 0.5};
    DoomAgent agent(agentConf, netConf, optConf);				
    
	ActionValueNetwork *network = agent.getNetwork();
    initWeights(network);	
    std::vector<vizdoom::BufferPtr> screenBufs = initInput();

	agent.agentStart(screenBufs[0]);		
    for(unsigned i=1; i<bufferSize; ++i)
    	std::vector<double> currentAction = agent.agentStep(0.0, screenBufs[i]);
	
    // Prints layers after getActionValuePreds is executed, meaning that the network includes samples 0, 1, 2 and 3 in batch (sample 4 is excluded).
    // To print forward prop of sample 4, print layers after getActionValueTargets and before getActionValuePreds.  
    printLayers(network);     
}       

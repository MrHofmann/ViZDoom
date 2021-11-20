#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

#include "agent/doomagent.h"


void sleep(unsigned int time){
    std::this_thread::sleep_for(std::chrono::milliseconds(time));
}

using namespace vizdoom;

AgentDebug doomDebug(2);


int main() 
{
        // Lunar lander agentConf = {x, 50000, 4, 8, x, x, x}.
        //AgentConfig agentConf = {8, 1000, 1, 1, 7, 1, 1};

        // My old configuration.
        //NetworkConfig netConf = {{160, 120, 3}, {8, 4}, {4, 2}, {1, 1}, {RELU, RELU}, {4, 2}, {1, 1}, {50, 20}, 8};                             // 2GB RAM, 8sec per st>

	//numActions, replayBufferSize, numReplay, numMiniBatch, seed, discount, tau
	AgentConfig agentConf = {3, 100, 10, 1, 1, 1};
	//inputSize, numFilters, filterDim, filterStride, activations, poolDim, poolStride, numHiddenUnits, numActions
	NetworkConfig netConf = {{6, 6, 3}, {3, 4}, {2, 2}, {1, 1}, {TANH, TANH}, {2, 2}, {1, 1}, {10, 5}, 3};
	OptimizerConfig optConf = {0.1, 0.2, 0.3, 0.5};
		
	vizdoom::BufferPtr screenBuf(new std::vector<uint8_t>(
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 ,30, 31, 32, 33, 34, 35,
		36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
		72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107}));

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


	DoomAgent agent(agentConf, netConf, optConf);				
	agent.agentStart(screenBuf);		
 		
	ActionValueNetwork *network = agent.getNetwork();
	auto layers = network->getLayers();
	auto iter = layers.begin();
	iter++;
	((Conv3dLayer*)(*iter))->setWeights(conv1Weights);
	iter++;
	iter++;
	((Conv3dLayer*)(*iter))->setWeights(conv2Weights);
	iter++;
	iter++;
	((DenseLayer*)(*iter))->setWeights(dense1Weights);
	iter++;
	((DenseLayer*)(*iter))->setWeights(dense2Weights);
	iter++;
	((OutputLayer*)(*iter))->setWeights(dense3Weights);		
	
	std::vector<double> currentAction = agent.agentStep(0.0, screenBuf);
		
	for(auto it=layers.begin(); it!=layers.end(); ++it)
	{
		if((*it)->layerType() == NetworkLayer::INPUT)
		{
			std::cout << "InputLayer" << std::endl;
			
		}
		else if((*it)->layerType() == NetworkLayer::CONV)
		{
			std::cout << "Conv3dLayer" << std::endl;
	
			std::vector<float> weights = ((Conv3dLayer*)(*it))->weights();
			for(int i=0; i<weights.size(); ++i)
				std::cout << weights[i] << " ";
			std::cout << std::endl;

			std::vector<float> dotProducts = ((Conv3dLayer*)(*it))->dotProducts();
			for(int i=0; i<dotProducts.size(); ++i)
				std::cout << dotProducts[i] << " ";
			std::cout << std::endl;

			std::vector<float> zHat = ((Conv3dLayer*)(*it))->zHat();
			for(int i=0; i<zHat.size(); ++i)
				std::cout << zHat[i] << " ";
			std::cout << std::endl;

			std::vector<float> normed = ((Conv3dLayer*)(*it))->normed();
			for(int i=0; i<normed.size(); ++i)
				std::cout << normed[i] << " ";
			std::cout << std::endl;
		}
		else if((*it)->layerType() == NetworkLayer::MAX_POOL)
		{
			std::cout << "Pool3dLayer" << std::endl;
		}
		else if((*it)->layerType() == NetworkLayer::FC)
		{
			std::cout << "DenseLayer" << std::endl;
			
			std::vector<float> weights = ((DenseLayer*)(*it))->weights();
			for(int i=0; i<weights.size(); ++i)
				std::cout << weights[i] << " ";
			std::cout << std::endl;

			std::vector<float> dotProducts = ((DenseLayer*)(*it))->dotProducts();
			for(int i=0; i<dotProducts.size(); ++i)
				std::cout << (int)dotProducts[i] << " ";
			std::cout << std::endl;

		}	
		else if((*it)->layerType() == NetworkLayer::OUTPUT)
		{
			std::cout << "OutputLayer (may give incorect but close results for great values because of float type low precision)" << std::endl;
			
			std::vector<float> weights = ((OutputLayer*)(*it))->weights();
			for(int i=0; i<weights.size(); ++i)
				std::cout << weights[i] << " ";
			std::cout << std::endl;

			std::vector<float> dotProducts = ((OutputLayer*)(*it))->dotProducts();
			for(int i=0; i<dotProducts.size(); ++i)
				std::cout << (int)dotProducts[i] << " ";
			std::cout << std::endl;

		}
	
		std::vector<float> activations = (*it)->activations();
		for(int i=0; i<activations.size(); ++i)
			std::cout << (int)activations[i] << " ";
		std::cout << std::endl;
	}
    
    //float f=618801000.0f;
    //for(unsigned i=0; i<2000; ++i)
    //{
    //    std::cout << f << " ";
    //    f = 618801000.0f + i*1.0f;
    //}
    //std::cout << std::endl;
        
}       

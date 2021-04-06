#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

#include "agent/doomagent.h"


int main() 
{	
	//std::map<std::string, matrix4d> weights_conv;
	//std::map<std::string, matrix1d> bias_conv;
	//std::map<std::string, matrix2d> weights_fc;
	//std::map<std::string, double> bias_fc;
	
	//matrix4d l1_filters;
	//matrix1d l1_biases;
	//for(unsigned i=0; i<3; ++i)
	//{
	//	//l1_filters.push_back({{{1, 2}, {3, 4}}, {{4, 3}, {2, 1}}, {{1, 4}, {3, 2}}});
	//	l1_filters.push_back({{{1, 4, 1}, {2, 3, 4}}, {{3, 2, 3}, {4, 1, 2}}});
	//	l1_biases.push_back(1);
	//}
	//weights_conv["conv_0"] = l1_filters;
	//bias_conv["conv_0"] = l1_biases;

	//matrix4d l2_filters;
	//matrix1d l2_biases;
	//for(unsigned i=0; i<4; ++i)
	//{
	//	//l2_filters.push_back({{{1, 2}, {3, 4}}, {{4, 3}, {2, 1}}, {{1, 4}, {3, 2}}});
	//	l2_filters.push_back({{{1, 4, 1}, {2, 3, 4}}, {{3, 2, 3}, {4, 1, 2}}});
	//	l2_biases.push_back(1);
	//}
	//weights_conv["conv_1"] = l2_filters;
	//bias_conv["conv_1"] = l2_biases;

	//matrix2d w_fc1;
	//double b_fc1 = 1;
	//for(unsigned i=0; i<16; ++i)
	//	w_fc1.push_back({1, 2, 3, 4, 1, 2, 3, 4, 1, 2});
	//weights_fc["fc_0"] = w_fc1;
	//bias_fc["fc_1"] = b_fc1;

	//matrix2d w_fc2;
	//double b_fc2 = 1;
	//for(unsigned i=0; i<10; ++i)
	//	w_fc2.push_back({1, 2, 3, 4, 1});
	//weights_fc["fc_1"] = w_fc2;
	//bias_fc["fc_1"] = b_fc2;

//	std::map<std::string, matrix2d> weights = network->get_weights_fc();
//	for(auto it=weights.begin(); it!=weights.end(); ++it)
//	{
//		matrix2d w = it->second;
//		for(unsigned i=0; i<w.size(); ++i)
//			for(unsigned j=0; j<w[i].size(); ++j)
//				std::cout << w[i][j] << " ";
//
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;

	
	AgentConfig agent_conf = {3, 100, 10, 7, 1, 1};
	NetworkConfig net_conf = {{6, 6, 3}, {3, 4}, {2, 2}, {1, 1}, {TANH, TANH}, {2, 2}, {1, 1}, {10, 5}, 3};
	OptimizerConfig opt_conf = {0.1, 0.2, 0.3, 0.5};
		
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


	DoomAgent agent(agent_conf, net_conf, opt_conf);				
	agent.agent_start(screenBuf);		
 		
	ActionValueNetwork *network = agent.get_network();
	auto layers = network->get_layers();
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
	((DenseLayer*)(*iter))->setWeights(dense3Weights);		
	
	std::vector<double> currentAction = agent.agent_step(0.0, screenBuf);
		
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
				std::cout << dotProducts[i] << " ";
			std::cout << std::endl;

		}

		std::vector<float> activations = (*it)->activations();
		for(int i=0; i<activations.size(); ++i)
			std::cout << activations[i] << " ";
		std::cout << std::endl;
	}
	
}       

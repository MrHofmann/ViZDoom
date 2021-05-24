#include "actionvaluenetwork.h"

extern AgentDebug doomDebug;

ActionValueNetwork::ActionValueNetwork(const AgentConfig &agentConf, const NetworkConfig &conf)
	:_stateDim(conf.stateDim), _batchSize(agentConf.numMinibatch), _numActions(conf.numActions)
{
	auto start = doomDebug.start("ActionValueNetwork::ActionValueNetwork", 2);

	NetworkLayer *prevLayer = nullptr;
	std::vector<unsigned> prevLayerSize = {_batchSize, _stateDim[0], _stateDim[1], _stateDim[2]};
	NetworkLayer *layer = new InputLayer("input", RELU, prevLayerSize, prevLayer);
	_layers.push_back(layer);
	_layerSizes["input"] = prevLayerSize;

	prevLayer = layer;
	for(unsigned i=0; i<conf.numFilters.size(); ++i)
	{
		std::string layerName = "conv_" + std::to_string(i);
		unsigned numFilter = conf.numFilters[i];
		unsigned filterDim = conf.filterDim[i];
		unsigned filterStride = conf.filterStrides[i];
		ActivationType activation = conf.activations[i];

		// Add padding later.
		unsigned layerHeight = (prevLayerSize[1] - conf.filterDim[i])/conf.filterStrides[i] + 1;
		unsigned layerWidth = (prevLayerSize[2] - conf.filterDim[i])/conf.filterStrides[i] + 1;
		unsigned layerDepth = conf.numFilters[i];
		std::vector<unsigned> layerSize = {_batchSize, layerHeight, layerWidth, layerDepth};
		layer = new Conv3dLayer(layerName, activation, layerSize, prevLayer, filterDim, prevLayerSize[3], filterStride);
		_layers.push_back(layer);
		_layerSizes[layerName] = layerSize;

		prevLayer->setNextLayer(layer);
		prevLayer = layer;
		prevLayerSize = layerSize;
		
		if(conf.poolDim[i] > 0)
		{
		   	layerName = "max_pool_" + std::to_string(i);
			unsigned poolDim = conf.poolDim[i];
			unsigned poolStride = conf.poolStrides[i];

			layerHeight = (prevLayerSize[1] - poolDim)/poolStride + 1;
			layerWidth = (prevLayerSize[2] - poolDim)/poolStride + 1;
			layerDepth = prevLayerSize[3];
			layerSize = {_batchSize, layerHeight, layerWidth, layerDepth};
			layer = new Pool3dLayer(layerName, activation, layerSize, prevLayer, poolDim, poolStride);
			_layers.push_back(layer);
			_layerSizes[layerName] = layerSize;

			prevLayer->setNextLayer(layer);
			prevLayer = layer;
			prevLayerSize = layerSize;
		}
	}

	for(unsigned i=0; i<conf.numHiddenUnits.size(); ++i)
	{
		std::string layerName = "fc_" + std::to_string(i);

		std::vector<unsigned> layerSize = {_batchSize, conf.numHiddenUnits[i]};
		layer = new DenseLayer(layerName, RELU, layerSize, prevLayer, conf.numHiddenUnits[i]);
		_layers.push_back(layer);
		_layerSizes[layerName] = layerSize;
  	
		prevLayer->setNextLayer(layer);
		prevLayer = layer;
	}
	
	// Output layer	
	std::string layerName = "output";

	unsigned outputSize = 1 << conf.numActions;
	std::vector<unsigned> layerSize = {_batchSize, outputSize};
	layer = new OutputLayer(layerName, RELU, layerSize, prevLayer, outputSize);
	_layers.push_back(layer);
	_layerSizes[layerName] = layerSize;
	prevLayer->setNextLayer(layer);

//	this->init_kaiming();

	std::string prevLayerName;
	for(auto it=this->_layers.begin(); it!=_layers.end(); it++)
	{
		std::string layerName = (*it)->layerName();
		std::cerr << layerName << " [ ";
		for(unsigned i=0; i<_layerSizes[layerName].size(); ++i)
			std::cerr << _layerSizes[layerName][i] << " ";
		std::cerr << "] [ ";
		if((*it)->layerType() == NetworkLayer::CONV)
			std::cerr << ((Conv3dLayer*)*it)->filterDim() << ", " << ((Conv3dLayer*)*it)->filterDim() << ", " << ((Conv3dLayer*)*it)->filterDepth();
		else if((*it)->layerType() == NetworkLayer::MAX_POOL)
			std::cerr << ((Pool3dLayer*)*it)->poolDim() << ", " << ((Pool3dLayer*)*it)->poolDim();
		else if((*it)->layerType() == NetworkLayer::FC)
		{
			std::vector<unsigned> prevLayerSize = _layerSizes[prevLayerName];
			int s = 0;
			for(unsigned i=1; i<prevLayerSize.size(); ++i)
				s += prevLayerSize[i];
			std::cerr << ((DenseLayer*)*it)->numHiddenUnits()*s;
		}
		std::cerr << " ]" << std::endl;

		prevLayerName = layerName;
	}
	
	doomDebug.end("ActionValueNetwork::ActionValueNetwork", 2, start);
}

void ActionValueNetwork::initInput(const std::vector<vizdoom::BufferPtr> &states)
{
	auto start = doomDebug.start("ActionValueNetwork::initInput", 3);
	
	InputLayer *inputLayer = (InputLayer*)*_layers.begin();
	inputLayer->setState(states);
	
	doomDebug.end("ActionValueNetwork::initInput", 3, start);
}

void ActionValueNetwork::cacheWeights()
{
	auto start = doomDebug.start("ActionValueNetwork::cacheWeights", 3);

	for(auto it=_layers.begin(); it!=_layers.end(); it++)
	{
		if((*it)->layerType() == NetworkLayer::CONV)
			((Conv3dLayer*)(*it))->cacheWeights();
		else if((*it)->layerType() == NetworkLayer::FC)
			((DenseLayer*)(*it))->cacheWeights();
	}		
	
	doomDebug.end("ActionValueNetwork::cacheWeights", 3, start);
}

// state -> conv3d -> max_pool -> conv3d -> max_pool -> fully_connected -> fully_connected -> softmax
std::vector<float> ActionValueNetwork::getActionValueSingle(vizdoom::BufferPtr state)
{
	auto start = doomDebug.start("ActionValueNetwork::getActionValueSingle", 2);

	std::vector<vizdoom::BufferPtr> states = {state};
	// Init first layer.
	initInput(states);
	for(auto it=_layers.begin(); it!=_layers.end(); it++)
		// activation(prev_layer*W + b)
		(*it)->forwardProp(NetworkLayer::SINGLE);

	std::vector<float> actionValues = _layers.back()->activations();
	
	doomDebug.end("ActionValueNetwork::getActionValueSingle", 2, start);
	return std::vector<float>(actionValues.begin(), actionValues.begin() + _layers.back()->layerSize()[1]);
}

std::vector<std::vector<float>> ActionValueNetwork::getActionValuePreds(const std::vector<ExperienceSample> &experiences)
{
	auto start = doomDebug.start("ActionValueNetwork::getActionValuePreds", 2);

	std::vector<vizdoom::BufferPtr> states;
	for(unsigned i=0; i<experiences.size(); ++i)
		states.push_back(experiences[i].state);

	initInput(states);
	for(auto it=_layers.begin(); it!=_layers.end(); it++)
		(*it)->forwardProp(NetworkLayer::PREDICTION);

	std::vector<float> actionValues = _layers.back()->activations();
	
	doomDebug.end("ActionValueNetwork::getActionValuePreds", 2, start);
	return {std::vector<float>(actionValues.begin(), actionValues.end())};
}


std::vector<std::vector<float>> ActionValueNetwork::getActionValueTargets(const std::vector<ExperienceSample> &experiences)
{
	auto start = doomDebug.start("ActionValueNetwork::getActionValueTargets", 2);

	std::vector<vizdoom::BufferPtr> nextStates;
	for(unsigned i=0; i<experiences.size(); ++i)
		nextStates.push_back(experiences[i].nextState);

	initInput(nextStates);
	for(auto it=_layers.begin(); it!=_layers.end(); it++)
		(*it)->forwardProp(NetworkLayer::TARGET);

	std::vector<float> actionValues = _layers.back()->activations();
	
	doomDebug.end("ActionValueNetwork::getActionValueTargets", 2, start);
	return {std::vector<float>(actionValues.begin(), actionValues.end())};
}

void ActionValueNetwork::getTDUpdate(const std::vector<ExperienceSample> &experiences, const std::vector<double> &deltaVec)
{
	auto start = doomDebug.start("ActionValueNetwork::getTDUpdate", 2);
	
	std::vector<std::vector<double>> actions;
	for(unsigned i=0; i<experiences.size(); ++i)
		actions.push_back(experiences[i].action);

	std::vector<std::vector<float>> deltaMat;
	NetworkLayer *output = _layers.back();
	for(auto it=_layers.rbegin(); it!=_layers.rend(); it++)
		(*it)->backProp(actions, deltaVec);
	
	doomDebug.end("ActionValueNetwork::getTDUpdate", 2, start);
}

std::list<NetworkLayer*> ActionValueNetwork::getLayers() const
{
	auto start = doomDebug.start("ActionValueNetwork::getLayers", 2);

	doomDebug.end("ActionValueNetwork::getLayers", 2, start);
	return _layers;
}

/*void ActionValueNetwork::init_saxe(unsigned num_rows, unsigned num_cols)
{
	std::cout << "ActionValueNetwork::init_saxe" << std::endl;
}*/

// Kaiming He initialization. Appropriate for ReLU and leaky ReLU activations.
/*void ActionValueNetwork::init_kaiming()
{
	std::cout << "ActionValueNetwork::init_kaiming" << std::endl;

	// Try using member random generator.
	std::default_random_engine g;

	for(auto it=weights_conv.begin(); it!=weights_conv.end(); it++)
	{
		int n = this->filter_dim[it->first]*this->filter_dim[it->first]*this->num_filters[it->first];
		std::normal_distribution<double> normal(0, std::sqrt(2.0/n));
		for(unsigned i=0; i<it->second.size(); ++i)
			for(unsigned j=0; j<it->second[i].size(); ++j)
				for(unsigned k=0; k<it->second[i][j].size(); ++k)
					for(unsigned l=0; l<it->second[i][j][k].size(); ++l)
						it->second[i][j][k][l] = normal(g);
	}

	// Check whether n should be the size of input or output of coresponding layer. 
	// It should be input (fan-in) so check which one of it->second.size or it->second[i].size should be used as n.
	for(auto it=weights_fc.begin(); it!=weights_fc.end(); it++)
	{
		int n = it->second.size();
		std::normal_distribution<double> normal(0, std::sqrt(2.0/n));
		for(unsigned i=0; i<it->second.size(); ++i)
			for(unsigned j=0; j<it->second[i].size(); ++j)
				it->second[i][j] = normal(g);
	}

	//Bias should already be initialized to 0.
}*/



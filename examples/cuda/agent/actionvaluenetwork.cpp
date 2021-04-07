#include "actionvaluenetwork.h"


ActionValueNetwork::ActionValueNetwork(const NetworkConfig &conf)
	:_stateDim(conf.state_dim), _numActions(conf.num_actions)
{
	std::cout << "ActionValueNetwork::ActionValueNetwork" << std::endl;

	NetworkLayer *prevLayer = nullptr;
	std::vector<unsigned> prevLayerSize = {_stateDim[0], _stateDim[1], _stateDim[2]};
	NetworkLayer *layer = new InputLayer("input", RELU, prevLayerSize, prevLayer);
	_layers.push_back(layer);
	_layerSizes["input"] = prevLayerSize;

	prevLayer = layer;
	for(unsigned i=0; i<conf.num_filters.size(); ++i)
	{
		std::string layerName = "conv_" + std::to_string(i);
		unsigned numFilter = conf.num_filters[i];
		unsigned filterDim = conf.filter_dim[i];
		unsigned filterStride = conf.filter_strides[i];
		ActivationType activation = conf.activations[i];

		// Add padding later.
		unsigned layerHeight = (prevLayerSize[0] - conf.filter_dim[i])/conf.filter_strides[i] + 1;
		unsigned layerWidth = (prevLayerSize[1] - conf.filter_dim[i])/conf.filter_strides[i] + 1;
		unsigned layerDepth = conf.num_filters[i];
		std::vector<unsigned> layerSize = {layerHeight, layerWidth, layerDepth};

		layer = new Conv3dLayer(layerName, activation, layerSize, prevLayer, filterDim, prevLayerSize[2], filterStride);
		_layers.push_back(layer);

		prevLayer = layer;
		prevLayerSize = layerSize;
		_layerSizes[layerName] = prevLayerSize;
		
		if(conf.pool_dim[i] > 0)
		{
		   	layerName = "max_pool_" + std::to_string(i);
			unsigned poolDim = conf.pool_dim[i];
			unsigned poolStride = conf.pool_strides[i];

			layerHeight = (prevLayerSize[0] - poolDim)/poolStride + 1;
			layerWidth = (prevLayerSize[1] - poolDim)/poolStride + 1;
			layerDepth = prevLayerSize[2];
			
			layer = new Pool3dLayer(layerName, activation, std::vector<unsigned>({layerHeight, layerWidth, layerDepth}), prevLayer, poolDim, poolStride);
			_layers.push_back(layer);
		
			prevLayer = layer;
			prevLayerSize = {layerHeight, layerWidth, layerDepth};
			_layerSizes[layerName] = prevLayerSize;
		}
	}

	int prevFcSize = prevLayerSize[0]*prevLayerSize[1]*prevLayerSize[2];
	unsigned i;
	for(i=0; i<conf.num_hidden_units.size(); ++i)
	{
		std::string layerName = "fc_" + std::to_string(i);
		unsigned numHiddenUnits = conf.num_hidden_units[i];

		layer = new DenseLayer(layerName, RELU, std::vector<unsigned>({numHiddenUnits}), prevLayer, numHiddenUnits);
		_layers.push_back(layer);
		_layerSizes[layerName] = {conf.num_hidden_units[i]};
  	
		prevLayer = layer;
		prevFcSize = conf.num_hidden_units[i];
	}
	
	// Output layer	
	std::string layerName = "fc_" + std::to_string(i);
	unsigned outputSize = 1 << conf.num_actions;
	unsigned numHiddenUnits = outputSize;

	layer = new DenseLayer(layerName, RELU, std::vector<unsigned>({numHiddenUnits}), prevLayer, numHiddenUnits);
	_layers.push_back(layer);
	_layerSizes[layerName] = {outputSize};
	
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
			for(unsigned i=0; i<prevLayerSize.size(); ++i)
				s += prevLayerSize[i];
			std::cerr << ((DenseLayer*)*it)->numHiddenUnits()*s;
		}
		std::cerr << " ]" << std::endl;

		prevLayerName = layerName;
	}
}

void ActionValueNetwork::initInput(vizdoom::BufferPtr s)
{
	std::cout << "ActionValueNetwork::init_input" << std::endl;
	
	InputLayer *inputLayer = (InputLayer*)*_layers.begin();
	inputLayer->setState(s);
}

// state -> conv3d -> max_pool -> conv3d -> max_pool -> fully_connected -> fully_connected -> softmax
std::vector<float> ActionValueNetwork::getActionValues(vizdoom::BufferPtr s)
{
	std::cout << "ActionValueNetwork::get_action_values" << std::endl;
	
	// Init first layer.
	initInput(s);
	auto it =_layers.begin();
	for(; it!=_layers.end(); it++)
		// activation(prev_layer*W + b)
		(*it)->forwardProp();

	std::vector<float> actionValues = _layers.back()->activations();
	return std::vector<float>(actionValues.begin(), actionValues.end()-1);
}

std::list<NetworkLayer*> ActionValueNetwork::getLayers() const
{
	std::cout << "ActionValueNetwork::get_layers" << std::endl;

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

/*std::vector<std::vector<double>> ActionValueNetwork::get_td_update(vizdoom::BufferPtr s, const std::vector<double> &delta_mat)
{
	std::cout << "ActionValueNetwork::get_td_update" << std::endl;

	// Continue here tomorrow.
}*/



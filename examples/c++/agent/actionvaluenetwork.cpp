#include "actionvaluenetwork.h"
#include <limits>


NetworkLayer::NetworkLayer(std::string n, ActivationType a, const std::vector<unsigned> &s)
	:_layerName(n), _activationType(a), _layerSize(s)
{
	std::cout << "NetworkLayer::NetworkLayer" << std::endl;
}

std::string NetworkLayer::layerName() const
{
	//std::cout << "NetworkLayer::layerName" << std::endl;
	return _layerName;
}

ActivationType NetworkLayer::activationType() const
{
	std::cout << "NetworkLayer::activationType" << std::endl;
	return _activationType;
}

std::vector<unsigned> NetworkLayer::layerSize() const
{
	//std::cout << "NetworkLayer::layerSize" << std::endl;
	return _layerSize;
}


//----------------------------------------------------------------------------------------------------------------------------------------//

Conv3dLayer::Conv3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned fd, unsigned fs)
	:NetworkLayer(ln, at, ls), _filterDim(fd), _filterStride(fs)
{
	std::cout << "Conv3dLayer::Conv3dLayer" << std::endl;

    Tensor3d *prevVertices;
	if(prevLayer != nullptr && prevLayer->layerType() == NetworkLayer::CONV)
		prevVertices = ((Conv3dLayer*)prevLayer)->vertices();
	else if(prevLayer != nullptr && prevLayer->layerType() == NetworkLayer::MAX_POOL)
		prevVertices = ((Pool3dLayer*)prevLayer)->vertices();
	else if(prevLayer != nullptr)
	{
		std::cout << "Unexpected previous layer type. Previous layer should be CONV or POOL." << std::endl;
		std::exit(1);
	}
	_vertices = new Tensor3d();
	std::vector<unsigned> curLayerSize = layerSize();
	for(unsigned i=0; i<curLayerSize[0]; ++i)
	{
		Tensor2d t2;
		for(unsigned j=0; j<curLayerSize[1]; ++j)
		{
			Tensor1d t1;
			for(unsigned k=0; k<curLayerSize[2]; ++k)
			{
				Vertex *v = new Vertex{at, 0, 0, 0, 0, 0, std::vector<Edge*>(), nullptr};
				if(prevLayer != nullptr)
				{
					unsigned filterDepth = prevLayer->layerSize()[2];
					unsigned numEdges = _filterDim*_filterDim*prevLayer->layerSize()[2];
					for(unsigned h=0; h<_filterDim; ++h)
						for(unsigned w=0; w<_filterDim; ++w)
							for(unsigned d=0; d<filterDepth; ++d)
							{
								Edge *e = new Edge{0, 0, 0, (*prevVertices)[i+h][j+w][d], v};
								v->in_edges.push_back(e);
							}
				}
				
				t1.push_back(v);
			}

			t2.push_back(t1);
		}
		_vertices->push_back(t2);
	}
}

NetworkLayer::LayerType Conv3dLayer::layerType() const
{
	//std::cout << "Conv3dLayer::layerType" << std::endl;
	return NetworkLayer::CONV;
}

void Conv3dLayer::forwardProp()
{
	std::cout << "Conv3dLayer::forwardProp" << std::endl;

	for(unsigned i=0; i<(*_vertices).size(); ++i)
		for(unsigned j=0; j<(*_vertices)[i].size(); ++j)
			for(unsigned k=0; k<(*_vertices)[i][j].size(); ++k)
			{
				// Parallelize this block.
				Vertex *v = (*_vertices)[i][j][k];
				v->inner_product = 0;
				for(unsigned h=0; h<v->in_edges.size(); ++h)
					for(unsigned w=0; w<v->in_edges[h].size(); ++w)
						for(unsigned d=0; d<v->in_edges[h][w].size(); ++d)
						{
							Edge *e = v->in_edges[h][w][d];
							v->in_products[h][w][d] = e->in_neuron->activation * e->weight;
							v->inner_product += v->in_products[h][w][d];
						}
				
				v->inner_product += v->bias;
                //switch(v->activation_type){...}
	            v->activation = std::max(0.0, v->inner_product);
			}
}

unsigned Conv3dLayer::filterDim() const
{
	std::cout << "Conv3dLayer::filterDim" << std::endl;	
}

unsigned Conv3dLayer::filterStride() const
{
	std::cout << "Conv3dLayer::filterStride" << std::endl;
}

Tensor3d* Conv3dLayer::vertices() const
{
	//std::cout << "Conv3dLayer::vertices" << std::endl;
	return _vertices;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------//
Pool3dLayer::Pool3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned pd, unsigned ps)
    :NetworkLayer(ln, at, ls), _poolDim(pd), _poolStride(ps)
{
	std::cout << "Pool3dLayer::Pool3dLayer" << std::endl;

	if(prevLayer == nullptr || prevLayer->layerType() != NetworkLayer::CONV)
	{
		std::cout << "Unexpected previous layer type. Previous layer must be conv." << std::endl;
		std::exit(1);
	}

	_vertices = new Tensor3d();
	std::vector<unsigned> curLayerSize = layerSize();
	unsigned poolDepth = prevLayer->layerSize()[2];
	unsigned numEdges = _poolDim*_poolDim*prevLayer->layerSize()[2];
	Tensor3d *prevVertices = ((Conv3dLayer*)prevLayer)->vertices();
	for(unsigned i=0; i<curLayerSize[0]; ++i)
	{
		Tensor2d t2;
		for(unsigned j=0; j<curLayerSize[1]; ++j)
		{
			Tensor1d t1;
			for(unsigned k=0; k<curLayerSize[2]; ++k)
			{
				// This vertex should be different from the one used in conv layers.
				Vertex *v = new Vertex{at, 0, 0, 0, 0, 0, std::vector<Edge*>(), nullptr};
				for(unsigned h=0; h<_poolDim; ++h)
					for(unsigned w=0; w<_poolDim; ++w)
						for(unsigned d=0; d<poolDepth; ++d)
						{
							Edge *e = new Edge{0, 0, 0, (*prevVertices)[i+h][j+w][d], v};
							v->in_edges.push_back(e);
						}				

				t1.push_back(v);
			}

			t2.push_back(t1);
		}

		_vertices->push_back(t2);
	}
}

NetworkLayer::LayerType Pool3dLayer::layerType() const
{
	//std::cout << "Pool3dLayer::layerType" << std::endl;
	return NetworkLayer::MAX_POOL;
}

void Pool3dLayer::forwardProp()
{
	std::cout << "Pool3dLayer::forwardProp" << std::endl;

    for(unsigned i=0; i<(*_vertices).size(); ++i)
        for(unsigned j=0; j<(*_vertices)[i].size(); ++j)
            for(unsigned k=0; k<(*_vertices)[i][j].size(); ++k)
            {
              	// Parallelize this block.
                Vertex *v = (*_vertices)[i][j][k];
				//v->inner_product = 0;
				v->activation = std::numeric_limits<double>::min();
                for(unsigned h=0; h<v->in_edges.size(); ++h)
                	for(unsigned w=0; w<v->in_edges[h].size(); ++w)
                    	for(unsigned d=0; d<v->in_edges[h][w].size(); ++d)
                        {
                            Edge *e = v->in_edges[h][w][d];
                           	//v->in_products[h][w][d] = e->in_neuron->activation * e->weight;
                            //v->inner_product += v->in_products[h][w][d];
							if(e->in_neuron->activation > v->activation)
								v->activation = e->in_neuron->activation;
                        }
                //v->inner_product += v->bias;
                //switch(v->activation_type){...}
				//v->activation = std::max(0.0, v->inner_product);
			}
}

Tensor3d* Pool3dLayer::vertices() const{
	//std::cout << "Pool3dLayer::vertices" << std::endl;
	return _vertices;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------//
DenseLayer::DenseLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned hu)
	:NetworkLayer(ln, at, ls), _numHiddenUnits(hu)
{
	std::cout << "DenseLayer::DenseLayer" << std::endl;

	_vertices = new Tensor1d();
	std::vector<unsigned> curLayerSize = layerSize();
	for(unsigned i=0; i<curLayerSize[0]; ++i)
	{
		Vertex *v = new Vertex{at, 0, 0, 0, 0, 0, std::vector<Edge*>(), nullptr};
		if(prevLayer != nullptr && prevLayer->layerType() == NetworkLayer::FC)
		{
			Tensor1d *prevVertices = ((DenseLayer*)prevLayer)->vertices();
            std::vector<unsigned> prevLayerSize = prevLayer->layerSize();
            for(unsigned j=0; j<prevLayerSize[0]; ++j)
			{
				Edge *e = new Edge{0, 0, 0, (*prevVertices)[j], v};
				v->in_edges.push_back(e);
			}
		}
	    else if(prevLayer != nullptr)
		{
	        Tensor3d *prevVertices = nullptr;
	        if(prevLayer->layerType() == NetworkLayer::CONV)
	            prevVertices = ((Conv3dLayer*)prevLayer)->vertices();
	        else if(prevLayer->layerType() == NetworkLayer::MAX_POOL)
	            prevVertices = ((Pool3dLayer*)prevLayer)->vertices();
        	else
		    {   
				std::cout << "Unexpected previous layer type. Previous layer should be CONV, POOL or FC." << std::endl;
				std::exit(1);
			}

			std::vector<unsigned> prevLayerSize = prevLayer->layerSize();
			for(unsigned h=0; h<prevLayerSize[0]; ++h)
				for(unsigned w=0; w<prevLayerSize[1]; ++w)
            		for(unsigned d=0; d<prevLayerSize[2]; ++d)
                	{
	           			Edge *e = new Edge{0, 0, 0, (*prevVertices)[h][w][d], v};
						v->in_edges.push_back(e);
                	}
		}

		_vertices->push_back(v);
	}
}

NetworkLayer::LayerType DenseLayer::layerType() const
{
	//std::cout << "DenseLayer::layerType" << std::endl;
	return NetworkLayer::FC;
}

void DenseLayer::forwardProp()
{
	std::cout << "DenseLayer::forwardProp" << std::endl;

	for(unsigned i=0; i<(*_vertices).size(); ++i)
	{
		// Parallelize this block.
		Vertex *v = (*_vertices)[i];
		v->inner_product = 0;
		for(unsigned j=0; j<v->in_edges.size(); ++j)
		{
			Edge *e = v->in_edges[j];
			v->in_products[j] = e->weight * e->in_neuron->activation;
			v->inner_product += v->in_products[j];
		}

		v->inner_product += v->bias;
		v->activation = std::max(0.0, v->inner_product);
	}
}

Tensor1d* DenseLayer::vertices() const
{
	//std::cout << "DenseLayer::vertices" << std::endl;
	return _vertices;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------//
/*double ActionValueNetwork::conv_product(vizdoom::BufferPtr state, std::vector<unsigned> prev_size, const matrix3d &filter, unsigned row, unsigned col) const
{
	//std::cout << "ActionValueNetwork::conv_product" << std::endl;

	unsigned input_height = prev_size[0];
	unsigned input_width = prev_size[1];
	unsigned input_depth = prev_size[2];

	std::vector<uint8_t> s = *state;
	double product = 0;
	for(unsigned i=0; i<filter.size(); ++i)
		for(unsigned j=0; j<filter[i].size(); ++j)
		{
			int current_index = (row+i)*(input_width*input_depth)+(col+j)*input_depth;
			double c1 = s[current_index]*filter[i][j][0];
			double c2 = s[current_index+1]*filter[i][j][1];
			double c3 = s[current_index+2]*filter[i][j][2];
			product += c1 + c2 + c3;
		}

	return product;
}*/


/*double ActionValueNetwork::max_pool(vizdoom::BufferPtr input, std::vector<unsigned> input_dim, int buffer_index, 
		unsigned pool_width, unsigned pool_height) const
{
	//std::cout << "ActionValueNetwork::max_pool" << std::endl;
	
	double result = std::numeric_limits<double>::min();
	for(unsigned i=0; i<pool_height; ++i)
		for(unsigned j=0; j<pool_width; ++j)
		{
			int current_index = buffer_index+(input_dim[2]*j)+(input_dim[1]*input_dim[2]*i);
			if(current_index >= (*input).size())
				std::cout << "current index out of bounds: " << current_index << " " << (*input).size() << std::endl;
			if((*input)[current_index] > result)
				result = (*input)[current_index];
		}		

	return result;
}*/

/*void ActionValueNetwork::conv3d(vizdoom::BufferPtr state, std::vector<unsigned> prev_size, std::string layer_name, 
		unsigned c, vizdoom::BufferPtr result)
{
	std::cout << "ActionValueNetwork::conv3d" << std::endl;

	unsigned input_height = prev_size[0];
	unsigned input_width = prev_size[1];
	unsigned input_depth = prev_size[2];
	
	unsigned result_height = this->layer_sizes[layer_name][0];
	unsigned result_width  = this->layer_sizes[layer_name][1];
	unsigned result_depth  = this->layer_sizes[layer_name][2];

	unsigned current_row = 0;
	unsigned current_col = 0;
	for(unsigned i=0; i<result_height; ++i)
	{
		current_row = i*this->filter_strides[layer_name];
		for(unsigned j=0; j<result_width; ++j)
		{
			current_col = j*this->filter_strides[layer_name]; 
			int output_index = i*result_width*result_depth + j*result_depth + c;
			// Also shoud check if segmentation fault can occur when accessing state neurons here with current row and col. 
			// Update: Should be ok now.
			double r = conv_product(state, prev_size, this->weights_conv[layer_name][c], current_row, current_col)+this->bias_conv[layer_name][c];
			switch(this->activations[layer_name]){
				case RELU:
					(*result)[output_index] = (uint8_t)std::max(0.0, r);
					break;
				case LRELU:
					if(r > 0)
						(*result)[output_index] = (uint8_t)r;
					else
						(*result)[output_index] = (uint8_t)(0.01*r);
					break;
				default:
					(*result)[output_index] = (uint8_t)r;
					break;
			}
		}
	}
}*/

/*void ActionValueNetwork::pool3d(vizdoom::BufferPtr input, std::vector<unsigned> prev_size, std::string layer_name, vizdoom::BufferPtr result)
{
	std::cout << "ActionValueNetwork::pool3d" << std::endl;

	unsigned output_height = this->layer_sizes[layer_name][0]; 
	unsigned output_width = this->layer_sizes[layer_name][1]; 
	unsigned output_depth = this->layer_sizes[layer_name][2];
	unsigned pool_size = this->pool_dim[layer_name];
	
	int buffer_index = 0;
	for(unsigned i=0; i<output_height; ++i)
	{
		int rows = i*output_width*output_depth;
		for(unsigned j=0; j<output_width; ++j)
		{			
			int cols = j*output_depth;
			for(unsigned k=0; k<output_depth; ++k)
			{
				(*result)[rows + cols + k] = max_pool(input, prev_size, buffer_index, pool_size, pool_size);
				buffer_index++;
			}
			if((buffer_index/prev_size[2])%prev_size[1] + pool_size > prev_size[1])
				buffer_index += prev_size[2];
			//if(buffer_index + pool_size exceedes number of input rows) ... This may not be needed since output buffer is already complete at this point.
			else
				buffer_index += prev_size[2]*(this->pool_strides[layer_name]-1);
		}
		buffer_index += prev_size[2]*prev_size[1]*(this->pool_strides[layer_name]-1);
	}
}*/

/*void ActionValueNetwork::fc_prop(vizdoom::BufferPtr input, std::string layer_name, vizdoom::BufferPtr result)
{
	std::cout << "ActionValueNetwork::fc_prop" << std::endl;

	for(unsigned j=0; j<(*result).size(); ++j)
	{
		(*result)[j] = this->bias_fc[layer_name];
		this->weights_fc[layer_name][0][j];
		for(unsigned i=0; i<(*input).size(); ++i)
			(*result)[j] += (*input)[i]*this->weights_fc[layer_name][i][j];		
	}
}*/

void ActionValueNetwork::firstLayerProp(vizdoom::BufferPtr s)
{
	std::cout << "ActionValueNetwork::firstLayerProp" << std::endl;

    //std::vector<unsigned> prevLayerSize = _stateDim;
	unsigned inputHeight = _stateDim[0];
	unsigned inputWidth = _stateDim[1];
	unsigned inputDepth = _stateDim[2];

	NetworkLayer *firstLayer = *_layers.begin();
	std::vector<unsigned> firstLayerSize = firstLayer->layerSize();
	Tensor3d *firstVertices = ((Conv3dLayer*)firstLayer)->vertices();
	unsigned filterDim = ((Conv3dLayer*)firstLayer)->filterDim();
	unsigned filterStride = ((Conv3dLayer*)firstLayer)->filterStride();

	for(unsigned i=0; i<(*firstVertices).size(); ++i)
		for(unsigned j=0; j<(*firstVertices)[i].size(); ++j)
		{
			// Place where the top left corner of the filter is placed in the input.
			int curInputIndex = (i*filterStride)*inputWidth*inputDepth + (j*filterStride)*inputDepth;
			for(unsigned k=0; k<(*firstVertices)[i][j].size(); ++k)
			{
				// Parallelize this block.
				Vertex *v = (*firstVertices)[i][j][k];
				for(unsigned h=0; h<filterDim; ++h)
					for(unsigned w=0; w<filterDim; ++w)
						for(unsigned d=0; d<inputDepth; ++d)
						{
							Edge *e = v->in_edges[h][w][d];
							double edgeProduct = e->weight * (*s)[i + h*inputWidth*inputDepth + w*inputDepth + d];
							v->in_products[h][w][d] = edgeProduct;
							v->inner_product += edgeProduct;
						}

				v->inner_product += v->bias;
				//switch(v->activation_type){...}
				v->activation = std::max(0.0, v->inner_product);
			}

		}
}

/*void ActionValueNetwork::singleLayerProp(NetworkLayer *layer)
{
	std::cout << "ActionValueNetwork::singleLayerProp" << std::endl;

	//if(layer->layerType == NetworkLayer::

}*/

ActionValueNetwork::ActionValueNetwork(const NetworkConfig &conf)
	:_stateDim(conf.state_dim), _numActions(conf.num_actions)
{
	std::cout << "ActionValueNetwork::ActionValueNetwork" << std::endl;

	NetworkLayer *prevLayer = nullptr;
	std::vector<unsigned> prevLayerSize = {_stateDim[0], _stateDim[1], _stateDim[2]};
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

		NetworkLayer *layer = new Conv3dLayer(layerName, activation, std::vector<unsigned>({layerHeight, layerWidth, layerDepth}), prevLayer, filterDim, filterStride);
		_layers.push_back(layer);

		prevLayer = layer;
		prevLayerSize = {layerHeight, layerWidth, layerDepth};
		_layerSizes[layerName] = prevLayerSize;
		
		if(conf.pool_dim[i] > 0)
		{
		   	layerName = "max_pool_" + std::to_string(i);
			unsigned poolDim = conf.pool_dim[i];
			unsigned poolStride = conf.pool_strides[i];

			layerHeight = (prevLayerSize[0] - poolDim)/poolStride + 1;
			layerWidth = (prevLayerSize[1] - poolDim)/poolStride + 1;
			layerDepth = prevLayerSize[2];
			
			NetworkLayer *layer = new Pool3dLayer(layerName, activation, std::vector<unsigned>({layerHeight, layerWidth, layerDepth}), prevLayer, poolDim, poolStride);
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

		NetworkLayer *layer = new DenseLayer(layerName, RELU, std::vector<unsigned>({numHiddenUnits}), prevLayer, numHiddenUnits);
		_layers.push_back(layer);
		_layerSizes[layerName] = {conf.num_hidden_units[i]};
  	
		prevLayer = layer;
		prevFcSize = conf.num_hidden_units[i];
	}
	
	// Output layer	
	std::string layerName = "fc_" + std::to_string(i);
	unsigned outputSize = 1 << conf.num_actions;
	unsigned numHiddenUnits = outputSize;

	NetworkLayer *layer = new DenseLayer(layerName, RELU, std::vector<unsigned>({numHiddenUnits}), prevLayer, numHiddenUnits);
	_layers.push_back(layer);
	_layerSizes[layerName] = {outputSize};
	
//	this->init_kaiming();

	for(auto it=this->_layers.begin(); it!=_layers.end(); it++)
	{
		std::string layerName = (*it)->layerName();
		std::cerr << layerName << " [ ";
		for(unsigned i=0; i<_layerSizes[layerName].size(); ++i)
			std::cerr << _layerSizes[layerName][i] << " ";
		std::cerr << "]" << std::endl;
	}
}

// state -> conv3d -> max_pool -> conv3d -> max_pool -> fully_connected -> fully_connected -> softmax
std::vector<double> ActionValueNetwork::get_action_values(vizdoom::BufferPtr s)
{
	std::cout << "ActionValueNetwork::get_action_values" << std::endl;
	
	// Init first layer.
	firstLayerProp(s);

	auto it=_layers.begin();
	it++;
	for(; it!=_layers.end(); it++)
	{
		// activation(prev_layer*W + b)
		(*it)->forwardProp();
		
		/*std::string layer_name = it->first;
		vizdoom::BufferPtr layer = it->second.second;
		switch(curLayer->layerType()){
			case CONV:
			{
				std::cout << "CONV" << std::endl;
				std::map<std::string, unsigned> num_f = this->num_filters;
				for(unsigned i=0; i<num_f[layer_name]; ++i)
					conv3d(prev_layer, prev_layer_size, layer_name, i, layer);
				conv3d(prevLayer, curLayer
				break;
			}
			case MAX_POOL:
			{
				std::cout << "MAX_POOL" << std::endl;
				pool3d(prev_layer, prev_layer_size, layer_name, layer);
				break;
			}
			case FC:
			{
				std::cout << "FC" << std::endl;
				fc_prop(prev_layer, layer_name, layer);
				break;
			}
			default:
				break;
		}

		prev_layer = layer;
		prev_layer_size = this->layer_sizes[layerName];*/

	}

	// return layers.back().second; // (BufferPtr)
	//return std::vector<double>((*prev_layer).begin(), (*prev_layer).end());
	return std::vector<double>();
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

/*std::list<std::pair<LayerType, std::pair<std::string, vizdoom::BufferPtr>>> ActionValueNetwork::get_layers() const
{
	std::cout << "ActionValueNetwork::get_layers" << std::endl;

	return this->layers;
}*/

/*std::map<std::string, matrix4d> ActionValueNetwork::get_weights_conv() const
{
	std::cout << "ActionValueNetwork::get_weights" << std::endl;

	return weights_conv;		
}


std::map<std::string, matrix1d> ActionValueNetwork::get_bias_conv() const
{
	std::cout << "ActionValueNetwork::get_weights" << std::endl;

	return bias_conv;		
}

std::map<std::string, matrix2d> ActionValueNetwork::get_weights_fc() const
{
	std::cout << "ActionValueNetwork::get_weights" << std::endl;

	return weights_fc;		
}


std::map<std::string, double> ActionValueNetwork::get_bias_fc() const
{
	std::cout << "ActionValueNetwork::get_weights" << std::endl;

	return bias_fc;		
}


void ActionValueNetwork::set_weights(const std::map<std::string, matrix4d> &w_conv, const std::map<std::string, matrix1d> &b_conv, 
		const std::map<std::string, matrix2d> &w_fc, const std::map<std::string, double> &b_fc)
{
	std::cout << "ActionValueNetwork::set_weights" << std::endl;

	this->weights_conv = w_conv;
	this->bias_conv = b_conv;
	this->weights_fc = w_fc;
	this->bias_fc = b_fc;
}*/

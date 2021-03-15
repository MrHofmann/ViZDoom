#include "actionvaluenetwork.h"
#include <limits>

double ActionValueNetwork::conv_product(vizdoom::BufferPtr state, std::vector<unsigned> prev_size, const matrix3d &filter, unsigned row, unsigned col) const
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
}

double ActionValueNetwork::max_pool(vizdoom::BufferPtr input, std::vector<unsigned> input_dim, int buffer_index, 
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
}

void ActionValueNetwork::conv3d(vizdoom::BufferPtr state, std::vector<unsigned> prev_size, std::string layer_name, 
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
}

void ActionValueNetwork::pool3d(vizdoom::BufferPtr input, std::vector<unsigned> prev_size, std::string layer_name, vizdoom::BufferPtr result)
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
}

void ActionValueNetwork::fc_prop(vizdoom::BufferPtr input, std::string layer_name, vizdoom::BufferPtr result)
{
	std::cout << "ActionValueNetwork::fc_prop" << std::endl;

	for(unsigned j=0; j<(*result).size(); ++j)
	{
		(*result)[j] = this->bias_fc[layer_name];
		this->weights_fc[layer_name][0][j];
		for(unsigned i=0; i<(*input).size(); ++i)
			(*result)[j] += (*input)[i]*this->weights_fc[layer_name][i][j];		
	}
}

ActionValueNetwork::ActionValueNetwork(const NetworkConfig &conf)
	:state_dim(conf.state_dim), num_actions(conf.num_actions)
{
	std::cout << "ActionValueNetwork::ActionValueNetwork" << std::endl;

	std::vector<unsigned> prev_layer_size = {state_dim[0], state_dim[1], state_dim[2]};
	for(unsigned i=0; i<conf.num_filters.size(); ++i)
	{
		std::string layer_name = "conv_" + std::to_string(i);
		this->num_filters[layer_name] = conf.num_filters[i];
		this->filter_dim[layer_name] = conf.filter_dim[i];
		this->filter_strides[layer_name] = conf.filter_strides[i];
		this->activations[layer_name] = conf.activations[i];

		// Add padding later.
		unsigned layer_height = (prev_layer_size[0] - conf.filter_dim[i])/conf.filter_strides[i] + 1;
		unsigned layer_width = (prev_layer_size[1] - conf.filter_dim[i])/conf.filter_strides[i] + 1;
		unsigned layer_depth = conf.num_filters[i];
		vizdoom::BufferPtr layer(new std::vector<uint8_t>(layer_height*layer_width*layer_depth));
		this->layers.push_back(std::make_pair(CONV, std::make_pair(layer_name, layer)));
	
						// num_filters			num_rows 					num_columns 		num_channels = num_prev_channels
  		matrix4d filters(conf.num_filters[i], matrix3d(conf.filter_dim[i], matrix2d(conf.filter_dim[i], matrix1d(prev_layer_size[2]))));
  		weights_conv[layer_name] = filters;
  		matrix1d biases(conf.num_filters[i]);
  		bias_conv[layer_name] = biases;

		prev_layer_size = {layer_height, layer_width, layer_depth};
		this->layer_sizes[layer_name] = prev_layer_size;
		
		if(conf.pool_dim[i] > 0)
		{
			layer_name = "max_pool_" + std::to_string(i);
			this->pool_dim[layer_name] = conf.pool_dim[i];
			this->pool_strides[layer_name] = conf.pool_strides[i];

			layer_height = (prev_layer_size[0] - conf.pool_dim[i])/conf.pool_strides[i] + 1;
			layer_width = (prev_layer_size[1] - conf.pool_dim[i])/conf.pool_strides[i] + 1;
			layer_depth = prev_layer_size[2];
			
			vizdoom::BufferPtr layer(new std::vector<uint8_t>(layer_height*layer_width*layer_depth));
			this->layers.push_back(std::make_pair(MAX_POOL, std::make_pair(layer_name, layer)));
			
			prev_layer_size = {layer_height, layer_width, layer_depth};
			this->layer_sizes[layer_name] = prev_layer_size;
		}
	}

	int prev_fc_size = prev_layer_size[0]*prev_layer_size[1]*prev_layer_size[2];
	unsigned i;
	for(i=0; i<conf.num_hidden_units.size(); ++i)
	{
		std::string layer_name = "fc_" + std::to_string(i);
		this->num_hidden_units[layer_name] = conf.num_hidden_units[i];

		vizdoom::BufferPtr layer(new std::vector<uint8_t>(conf.num_hidden_units[i]));
		this->layers.push_back(std::make_pair(FC, std::make_pair(layer_name, layer)));
		this->layer_sizes[layer_name] = {conf.num_hidden_units[i]};	
  	
		matrix2d weights(prev_fc_size, matrix1d(conf.num_hidden_units[i]));
    weights_fc[layer_name] = weights;
    bias_fc[layer_name] = 0;

		prev_fc_size = conf.num_hidden_units[i];
	}
	
	// Output layer	
	std::string layer_name = "fc_" + std::to_string(i);
	unsigned output_size = 1 << conf.num_actions;
	this->num_hidden_units[layer_name] = output_size;

	vizdoom::BufferPtr layer(new std::vector<uint8_t>(output_size));
	this->layers.push_back(std::make_pair(FC, std::make_pair(layer_name, layer)));
	this->layer_sizes[layer_name] = {output_size};
  	
	matrix2d weights(prev_fc_size, matrix1d(output_size));
  weights_fc[layer_name] = weights;
  bias_fc[layer_name] = 0;

	this->init_kaiming();

	for(auto it=this->layers.begin(); it!=layers.end(); it++)
	{
		std::string layer_name = it->second.first;
		std::cerr << layer_name << " [ ";
		for(unsigned i=0; i<this->layer_sizes[layer_name].size(); ++i)
			std::cerr << layer_sizes[layer_name][i] << " ";
		std::cerr << "]" << std::endl;
	}
}

// state -> conv3d -> max_pool -> conv3d -> max_pool -> fully_connected -> fully_connected -> softmax
std::vector<double> ActionValueNetwork::get_action_values(vizdoom::BufferPtr s)
{
	std::cout << "ActionValueNetwork::get_action_values" << std::endl;
		
	vizdoom::BufferPtr prev_layer = s;
	std::vector<unsigned> prev_layer_size = this->state_dim;
	for(auto it=this->layers.begin(); it!=this->layers.end(); it++)
	{
		// activation(prev_layer*W + b)
		std::string layer_name = it->second.first;
		vizdoom::BufferPtr layer = it->second.second;
		switch(it->first){
			case CONV:
			{
				std::cout << "CONV" << std::endl;
				std::map<std::string, unsigned> num_f = this->num_filters;
				for(unsigned i=0; i<num_f[layer_name]; ++i)
					conv3d(prev_layer, prev_layer_size, layer_name, i, layer);
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
		prev_layer_size = this->layer_sizes[layer_name];
	}

	// return layers.back().second; // (BufferPtr)
	return std::vector<double>((*prev_layer).begin(), (*prev_layer).end());
}

void ActionValueNetwork::init_saxe(unsigned num_rows, unsigned num_cols)
{
	std::cout << "ActionValueNetwork::init_saxe" << std::endl;
}

// Kaiming He initialization. Appropriate for ReLU and leaky ReLU activations.
void ActionValueNetwork::init_kaiming()
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
}

std::vector<std::vector<double>> ActionValueNetwork::get_td_update(vizdoom::BufferPtr s, const std::vector<double> &delta_mat)
{
	std::cout << "ActionValueNetwork::get_td_update" << std::endl;

	// Continue here tomorrow.
}

std::list<std::pair<LayerType, std::pair<std::string, vizdoom::BufferPtr>>> ActionValueNetwork::get_layers() const
{
	std::cout << "ActionValueNetwork::get_layers" << std::endl;

	return this->layers;
}

std::map<std::string, matrix4d> ActionValueNetwork::get_weights_conv() const
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
}

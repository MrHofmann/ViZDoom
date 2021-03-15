#ifndef ACTIONVALUENETWORK_H
#define ACTIONVALUENETWORK_H

#include <iostream>
#include "datastructures.h"

class ActionValueNetwork{
private:
	// Convolutional network with n conv layers, m fully connected layers and softmax output layer.
	std::vector<unsigned> state_dim;							// [H, W, C]
	std::map<std::string, unsigned> num_filters;        		// [F1, F2, ... , Fn]
	std::map<std::string, unsigned> filter_dim;         		// [F1s, F2s, ... , Fns]
	std::map<std::string, unsigned> filter_strides;     		// [Sf1, Sf2, ... , Sfn]
	std::map<std::string, ActivationType> activations;     		// [A1, A2, ... , An]
	std::map<std::string, unsigned> pool_dim;           		// [P1, P2, ... , Pn]
	std::map<std::string, unsigned> pool_strides;       		// [Sp1, Sp2, ... , Spn]
	std::map<std::string, unsigned> num_hidden_units; 			// [H1, H2, ... , Hm]
	std::map<std::string, std::vector<unsigned>> layer_sizes;
	unsigned num_actions;                     							// O

	//self.rand_generator = np.random.RandomState(network_config.get("seed"))
	std::list<std::pair<LayerType, std::pair<std::string, vizdoom::BufferPtr>>> layers;
	std::map<std::string, matrix4d> weights_conv;			// [Wc1, Wc2, ... , Wcn]
	std::map<std::string, matrix1d> bias_conv;				// [Bc1, Bc2, ... , Bcn]
	std::map<std::string, matrix2d> weights_fc;				// [Wfc1, Wfc2, ... , Wfcm]
	std::map<std::string, double> bias_fc;					// [Bfc1, Bfc2, ... , Bfcm]

	// This can be parallelized.
	double conv_product(vizdoom::BufferPtr state, std::vector<unsigned> prev_size, const matrix3d &filter, unsigned row, unsigned col) const;
	double max_pool(vizdoom::BufferPtr state, std::vector<unsigned> input_dim, int buffer_index, unsigned pool_width, unsigned pool_height) const;
	void conv3d(vizdoom::BufferPtr state, std::vector<unsigned> prev_size, std::string layer_name, unsigned c, vizdoom::BufferPtr result);
	void pool3d(vizdoom::BufferPtr input, std::vector<unsigned> prev_size, std::string layer_name, vizdoom::BufferPtr result);
	void fc_prop(vizdoom::BufferPtr input, std::string layer_name, vizdoom::BufferPtr result);

public:
	ActionValueNetwork(){}
	ActionValueNetwork(const NetworkConfig &conf);
	std::vector<double> get_action_values(vizdoom::BufferPtr s);
	std::vector<std::vector<double>> get_td_update(vizdoom::BufferPtr s, const std::vector<double> &delta_mat);
	void init_saxe(unsigned num_rows, unsigned num_cols);
	void init_kaiming();

	std::map<std::string, matrix4d> get_weights_conv() const;
	std::map<std::string, matrix1d> get_bias_conv() const;
	std::map<std::string, matrix2d> get_weights_fc() const;
	std::map<std::string, double> get_bias_fc() const;
	void set_weights(const std::map<std::string, matrix4d> &w_conv, const std::map<std::string, matrix1d> &b_conv, 
			const std::map<std::string, matrix2d> &w_fc, const std::map<std::string, double> &b_fc);
	std::list<std::pair<LayerType, std::pair<std::string, vizdoom::BufferPtr>>> get_layers() const;
};

#endif // ACTIONVALUENETWORK_H

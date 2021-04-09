#ifndef ACTIONVALUENETWORK_H
#define ACTIONVALUENETWORK_H

#include "networklayer.h"


class ActionValueNetwork{
private:
	// Convolutional network with n conv layers, m fully connected layers and softmax output layer.
	std::vector<unsigned> _stateDim;							// [H, W, C]
	//std::map<std::string, unsigned> num_filters;        		// [F1, F2, ... , Fn]
	//std::map<std::string, unsigned> filter_dim;         		// [F1s, F2s, ... , Fns]
	//std::map<std::string, unsigned> filter_strides;     		// [Sf1, Sf2, ... , Sfn]
	std::map<std::string, ActivationType> _activations;     		// [A1, A2, ... , An]
	//std::map<std::string, unsigned> pool_dim;           		// [P1, P2, ... , Pn]
	//std::map<std::string, unsigned> pool_strides;       		// [Sp1, Sp2, ... , Spn]
	//std::map<std::string, unsigned> num_hidden_units; 			// [H1, H2, ... , Hm]
	std::map<std::string, std::vector<unsigned>> _layerSizes;
	unsigned _numActions;                     							// O

	//self.rand_generator = np.random.RandomState(network_config.get("seed"))
	std::list<NetworkLayer*> _layers;

public:
	ActionValueNetwork(){}
	ActionValueNetwork(const NetworkConfig &conf);
	void initInput(vizdoom::BufferPtr s);
	void cacheWeights();
	std::vector<float> getActionValuePreds(vizdoom::BufferPtr s);
	std::vector<float> getActionValueTargets(vizdoom::BufferPtr s);
	std::list<NetworkLayer*> getLayers() const;
	void getTDUpdate(vizdoom::BufferPtr s, const std::vector<float> &delta_mat);
	//void init_saxe(unsigned num_rows, unsigned num_cols);
	//void init_kaiming();
};

#endif // ACTIONVALUENETWORK_H

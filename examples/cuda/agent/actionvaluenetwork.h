#ifndef ACTIONVALUENETWORK_H
#define ACTIONVALUENETWORK_H

#include <iostream>
//#include <thrust/device_vector.h>
//#include <thrust/device_ptr.h>
#include "datastructures.h"
#include "networkelements.h"

class NetworkLayer{
protected:
	std::string _layerName;
	ActivationType _activationType;
	std::vector<unsigned> _layerSize;
	NetworkLayer *_prevLayer;
	std::vector<float> _activations;
	BiasVertex *_bias;
		
public:
	enum LayerType {INPUT, CONV, MAX_POOL, FC};

	NetworkLayer(){}
	NetworkLayer(std::string n, ActivationType a, const std::vector<unsigned> &s, NetworkLayer *pl);
	virtual ~NetworkLayer(){std::cout << "Delete NetworkLayer: " << this->layerName() << std::endl;}
	virtual LayerType layerType() const = 0;
	virtual void forwardProp() = 0;
	std::string layerName() const;
	ActivationType activationType() const;
	std::vector<unsigned> layerSize() const;
	BiasVertex *biasVertex() const;
	std::vector<float> activations() const;
};

class InputLayer : public NetworkLayer{
private:
	vizdoom::BufferPtr _state;
	//std::vector<float> _activations;
	Tensor3d<Input3dVertex*> *_vertices;

public:
	InputLayer(){}
	InputLayer(std::string n, ActivationType a, const std::vector<unsigned> &s, NetworkLayer *pl);
	virtual LayerType layerType() const;
	virtual void forwardProp();

	Tensor3d<Input3dVertex*>* vertices() const;
	void setState(vizdoom::BufferPtr s);
};

class Conv3dLayer : public NetworkLayer{
private:
	unsigned _filterDim;
	unsigned _filterDepth;
	unsigned _filterStride;

	std::vector<float> _weights;
	std::vector<float> _dotProducts;
	//std::vector<float> _activations;
	std::vector<float> _TDUpdates;
	Tensor3d<Conv3dVertex*> *_vertices;

public:
	Conv3dLayer(){}
	Conv3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned fdi, unsigned fde, unsigned fs);
	virtual LayerType layerType() const;
	virtual void forwardProp();
	//Tensor1d<Dense1dVertex*> *_vertices;
	//thrust::device_ptr<float> weightsToDevice() const;
	//thrust::device_ptr<float> activationsToDevice() const;

	unsigned filterDim() const;
	unsigned filterDepth() const;
	unsigned filterStride() const;
	//std::vector<float> activations() const;
	Tensor3d<Conv3dVertex*>* vertices() const;
};

class Pool3dLayer : public NetworkLayer{
private:
	unsigned _poolDim;
	unsigned _poolDepth;
	unsigned _poolStride;

	//std::vector<float> _activations;
	Tensor3d<Pool3dVertex*> *_vertices;

public:
	Pool3dLayer(){}
	Pool3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned pdi, unsigned pde, unsigned fs);
	virtual LayerType layerType() const;
	virtual void forwardProp();

	unsigned poolDim() const;
	unsigned poolDepth() const;
	Tensor3d<Pool3dVertex*> *vertices() const;
};

class DenseLayer : public NetworkLayer{
private:
	unsigned _numHiddenUnits;

	std::vector<float> _weights;
	std::vector<float> _dotProducts;
	//std::vector<float> _activations;
	std::vector<float> _TDUpdates;
	Tensor1d<Dense1dVertex*> *_vertices;

public:
	DenseLayer(){}
	DenseLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned hu);
	virtual LayerType layerType() const;
	virtual void forwardProp();

	unsigned numHiddenUnits() const;
	Tensor1d<Dense1dVertex*> *vertices() const;
};

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
	void init_input(vizdoom::BufferPtr s);
	std::vector<float> get_action_values(vizdoom::BufferPtr s);
	//std::vector<std::vector<double>> get_td_update(vizdoom::BufferPtr s, const std::vector<double> &delta_mat);
	//void init_saxe(unsigned num_rows, unsigned num_cols);
	//void init_kaiming();
};

#endif // ACTIONVALUENETWORK_H

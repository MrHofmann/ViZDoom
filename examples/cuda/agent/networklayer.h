#ifndef NETWORKLAYER_H
#define NETWORKLAYER_H

#include <iostream>
#include "datastructures.h"
#include "networkelements.h"

class NetworkLayer{
protected:
	std::string _layerName;
	ActivationType _activationType;
	std::vector<unsigned> _layerSize;
	NetworkLayer *_prevLayer;
	NetworkLayer *_nextLayer;
	std::vector<float> _activations;
	BiasVertex *_bias;
		
public:
	enum LayerType {INPUT, CONV, MAX_POOL, FC, OUTPUT};
	enum PropagationType {SINGLE, PREDICTION, TARGET};

	NetworkLayer(){}
	NetworkLayer(std::string n, ActivationType a, const std::vector<unsigned> &s, NetworkLayer *pl);
	virtual ~NetworkLayer(){std::cout << "Delete NetworkLayer: " << this->layerName() << std::endl;}
	virtual LayerType layerType() const = 0;
	virtual void forwardProp(PropagationType p) = 0;
	virtual void backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec) = 0;
	std::string layerName() const;
	ActivationType activationType() const;
	std::vector<unsigned> layerSize() const;
	BiasVertex *biasVertex() const;
	std::vector<float> activations() const;
	void setNextLayer(NetworkLayer *nl);
};

class InputLayer : public NetworkLayer{
private:
	std::vector<vizdoom::BufferPtr> _states;
	Tensor3d<Input3dVertex*> *_vertices;

public:
	InputLayer(){}
	InputLayer(std::string n, ActivationType a, const std::vector<unsigned> &s, NetworkLayer *pl);
	virtual LayerType layerType() const;
	virtual void forwardProp(PropagationType p);
	virtual void backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec);

	Tensor3d<Input3dVertex*>* vertices() const;
	void setState(const std::vector<vizdoom::BufferPtr> &s);
};

class Conv3dLayer : public NetworkLayer{
private:
	unsigned _filterDim;
	unsigned _filterDepth;
	unsigned _filterStride;
	// Consider removing bias from _weights, _cachedWeights and _TDUpdates. It is not needed
	// when using batch normalization.
	std::vector<float> _weights;
	std::vector<float> _gamma;
	std::vector<float> _beta;
	std::vector<float> _cachedWeights;
	std::vector<float> _cachedGamma;
	std::vector<float> _cachedBeta;
	std::vector<float> _dotProducts;
	std::vector<float> _zHat;
	std::vector<float> _normed;
	std::vector<float> _means;
	std::vector<float> _vars;
	std::vector<float> _p1Sum;
	std::vector<float> _movingMeans;
	std::vector<float> _movingVars;
	std::vector<float> _TDUpdates;
	//std::vector<float> _TDGamma;
	//std::vector<float> _TDBeta;
	std::vector<float> _outGrads;
	Tensor3d<Conv3dVertex*> *_vertices;

public:
	Conv3dLayer(){}
	Conv3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned fdi, unsigned fde, unsigned fs);
	virtual LayerType layerType() const;
	virtual void forwardProp(PropagationType p);
	virtual void backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec);
	void cacheWeights();
	void updateStatistics(double momentum);


	unsigned filterDim() const;
	unsigned filterDepth() const;
	unsigned filterStride() const;	
	std::vector<float> weights() const;
	std::vector<float> dotProducts() const;
	std::vector<float> outGrads() const;
	Tensor3d<Conv3dVertex*>* vertices() const;
	void setWeights(const std::vector<float> &w);
};

class Pool3dLayer : public NetworkLayer{
private:
	unsigned _poolDim;
	unsigned _poolStride;

	std::vector<float> _outGrads;
	Tensor3d<Pool3dVertex*> *_vertices;

public:
	Pool3dLayer(){}
	Pool3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned pdi, unsigned fs);
	virtual LayerType layerType() const;
	virtual void forwardProp(PropagationType p);
	virtual void backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec);

	unsigned poolDim() const;
	std::vector<float> outGrads() const;
	Tensor3d<Pool3dVertex*> *vertices() const;
};

class DenseLayer : public NetworkLayer{
private:
	unsigned _numHiddenUnits;

	std::vector<float> _weights;
	std::vector<float> _gamma;
	std::vector<float> _beta;
	std::vector<float> _means;
	std::vector<float> _vars;
	std::vector<float> _p1Sum;
	std::vector<float> _cachedWeights;
	std::vector<float> _cachedGamma;
	std::vector<float> _cachedBeta;
	std::vector<float> _movingMeans;
	std::vector<float> _movingVars;
	std::vector<float> _dotProducts;
	std::vector<float> _zHat;
	std::vector<float> _normed;
	std::vector<float> _TDUpdates;
	std::vector<float> _outGrads;
	Tensor1d<Dense1dVertex*> *_vertices;

public:
	DenseLayer(){}
	DenseLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned hu);
	virtual LayerType layerType() const;
	virtual void forwardProp(PropagationType p);
	virtual void backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec);
	void cacheWeights();
	void updateStatistics(double momentum);

	unsigned numHiddenUnits() const;
	std::vector<float> weights() const;
	std::vector<float> dotProducts() const;
	std::vector<float> outGrads() const;
	Tensor1d<Dense1dVertex*> *vertices() const;
	void setWeights(const std::vector<float> &w);
};

class OutputLayer : public NetworkLayer{
private:
	unsigned _numHiddenUnits;

	std::vector<float> _weights;
	std::vector<float> _cachedWeights;
	std::vector<float> _dotProducts;
	std::vector<float> _TDUpdates;
	std::vector<float> _outGrads;
	Tensor1d<Dense1dVertex*> *_vertices;

public:
	OutputLayer(){}
	OutputLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned hu);
	virtual LayerType layerType() const;
	virtual void forwardProp(PropagationType p);
	virtual void backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec);
	std::vector<float> actionsToGrads(const std::vector<std::vector<double>> &actions) const;
	void cacheWeights();

	unsigned numHiddenUnits() const;
	std::vector<float> weights() const;
	std::vector<float> dotProducts() const;
	std::vector<float> outGrads() const;
	Tensor1d<Dense1dVertex*> *vertices() const;
	void setWeights(const std::vector<float> &w);
};


#endif // NETWORKLAYER_H

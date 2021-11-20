#ifndef ACTIONVALUENETWORK_H
#define ACTIONVALUENETWORK_H

#include "networklayer.h"


class ActionValueNetwork{
private:
	// Convolutional network with n conv layers, m fully connected layers and softmax output layer.
    //std::vector<unsigned> _stateDim;
	std::vector<unsigned> _inputDim;							
	std::map<std::string, ActivationType> _activations;     	
	std::map<std::string, std::vector<unsigned>> _layerSizes;
	unsigned _batchSize;
	unsigned _numActions;                     							

	//self.rand_generator = np.random.RandomState(network_config.get("seed"))
	std::list<NetworkLayer*> _layers;

public:
	ActionValueNetwork(){}
	ActionValueNetwork(const AgentConfig &agentConf, const NetworkConfig &conf);
	//void init_saxe(unsigned num_rows, unsigned num_cols);
	//void init_kaiming();
	void initInput(const std::vector<vizdoom::BufferPtr> &s);
	void cacheWeights();
	std::vector<float> getActionValueSingle(vizdoom::BufferPtr state);
	std::vector<std::vector<float>> getActionValuePreds(const std::vector<ExperienceSample> &experiences);
	std::vector<std::vector<float>> getActionValueTargets(const std::vector<ExperienceSample> &experiences);
	void getTDUpdate(const std::vector<ExperienceSample> &experiences, const std::vector<double> &deltaVec);
	
	std::list<NetworkLayer*> getLayers() const;
    std::vector<unsigned> getInputDim() const;
};

#endif // ACTIONVALUENETWORK_H

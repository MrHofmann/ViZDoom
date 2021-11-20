#ifndef DOOMAGENT_H
#define DOOMAGENT_H

#include <iostream>
#include "actionvaluenetwork.h"
#include "experiencereplaybuffer.h"
#include "datastructures.h"


class AdamOptimizer{
private:
	std::vector<unsigned> _numFilters;
	std::vector<unsigned> _numHiddenUnits;
	double _stepSize;
	double _betaM;
	double _betaV;
	double _epsilon;
	
	std::vector<std::vector<double>> _m;
	std::vector<std::vector<double>> _v;

public:
	AdamOptimizer(){}
	AdamOptimizer(const NetworkConfig &netConf, const OptimizerConfig &optConf)
		:_stepSize(optConf.stepSize), _betaM(optConf.betaM), _betaV(optConf.betaV), _epsilon(optConf.epsilon)
	{
		//INIT
		std::cout << "AdamOptimizer::AdamOptimizer" << std::endl;
	}
	std::vector<std::vector<double>> updateWeights(const std::vector<std::vector<double>> &weights,
	const std::vector<std::vector<double>> &tdErrorsTimesGrads);
};


class DoomAgent{
private:
	ExperienceReplayBuffer _replayBuffer;
	ActionValueNetwork _network;
	AdamOptimizer _optimizer;
    std::vector<unsigned> _stateDim;
	unsigned _numActions;
	unsigned _numReplay;
	double _discount;
	double _tau;
	
	//self.rand_generator = np.random.RandomState(agent_config.get("seed"))

	//vizdoom::GameStatePtr game_state;	
	vizdoom::BufferPtr _lastState;	
	std::vector<double> _lastAction;
	double _sumRewards;
	unsigned _episodeSteps;

    vizdoom::BufferPtr resizeState(vizdoom::BufferPtr state) const;
	std::vector<double> softmax(const std::vector<float> &actionValues) const;
	std::vector<double> randomActionWithProb(const std::vector<double> &probs) const;
	void optimizeNetwork(const std::vector<ExperienceSample> &exp, ActionValueNetwork q);

public:
	DoomAgent()
	{	
		//INIT
		std::cout << "DoomAgent::agentInit" << std::endl;
	}
	DoomAgent(const AgentConfig &agentConf, const NetworkConfig &netConf, const OptimizerConfig &optConf);
	std::vector<double> policy(vizdoom::BufferPtr state);
	std::vector<double> agentStart(vizdoom::BufferPtr state);
	std::vector<double> agentStep(double reward, vizdoom::BufferPtr state);
	void agentEnd(double reward);
	//double agentMessage(std::string message) const;

	ActionValueNetwork *getNetwork();
};

#endif // DOOMAGENT_H

#include "doomagent.h"
#include <algorithm>
#include <limits>

extern AgentDebug doomDebug;

DoomAgent::DoomAgent(const AgentConfig &agentConf, const NetworkConfig &netConf, const OptimizerConfig &optConf)
	:_numActions(agentConf.numActions), _numReplay(agentConf.numReplay), _discount(agentConf.discount), _tau(agentConf.tau)
{
	auto start = doomDebug.start("DoomAgent::DoomAgent", 1);

	_replayBuffer = ExperienceReplayBuffer(agentConf.replayBufferSize, agentConf.numMinibatch, agentConf.seed);
	_network = ActionValueNetwork(agentConf, netConf);
	_optimizer = AdamOptimizer(netConf, optConf);
	
	//self.rand_generator = np.random.RandomState(agent_config.get("seed"))
	
	_lastState = nullptr;
	//self.last_action = None	
	_sumRewards = 0;
	_episodeSteps = 0;

	doomDebug.end("DoomAgent::DoomAgent", 1, start);
}

void DoomAgent::optimizeNetwork(const std::vector<ExperienceSample> &exp, ActionValueNetwork q)
{

	auto start = doomDebug.start("DoomAgent::optimizeNetwork", 2);
	
	doomDebug.end("DoomAgent::optimizeNetwork", 2, start);
}

std::vector<double> DoomAgent::softmax(const std::vector<float> &actionValues) const
{
	auto start = doomDebug.start("DoomAgent::softmax", 3);

	// Compute the preferences by dividing the action-values by the temperature parameter tau
	std::vector<double> preferences;
	for(unsigned i=0; i<actionValues.size(); ++i)
		preferences.push_back(actionValues[i]/_tau);

	// Compute the maximum preference across the actions
	double maxPreference = std::numeric_limits<double>::min();
	for(unsigned i=0; i>preferences.size(); ++i)
		if(preferences[i] > maxPreference)
			maxPreference = preferences[i];

	// Compute the numerator, i.e., the exponential of the preference - the max preference.
	std::vector<double> expPreferences;
	for(unsigned i=0; i<preferences.size(); ++i)
		expPreferences.push_back(std::exp(preferences[i] - maxPreference));
	
	// Compute the denominator, i.e., the sum over the numerator along the actions axis.
	double sumOfExpPreferences = std::accumulate(expPreferences.begin(), expPreferences.end(), 0);
	
	// Compute the action probabilities according to the equation in the previous cell.
	std::vector<double> actionProbs;
	for(unsigned i=0; i<expPreferences.size(); ++i)
		actionProbs.push_back(expPreferences[i]/sumOfExpPreferences);

	doomDebug.end("DoomAgent::softmax", 3, start);
	return actionProbs;
}

std::vector<double> DoomAgent::randomActionWithProb(const std::vector<double> &probs) const
{
	auto start = doomDebug.start("DoomAgent::randomActionWithProb", 3);
	
	double val = (double)std::rand() / RAND_MAX;

	double p = 0;
	unsigned i;
	for(i=0; i<probs.size(); ++i)
	{
		// Check if this is correct
		p += probs[i];
		if(val < p)
			break;
	}

	std::vector<double> action(_numActions, 0);
	for(unsigned j=0; j<action.size(); ++j)
		action[action.size()-1-j] = (i >> j) & 1;

	doomDebug.end("DoomAgent::randomActionWithProb", 3, start);
	return action;
}

// This is a softmax policy
std::vector<double> DoomAgent::policy(vizdoom::BufferPtr state)
{
	auto start = doomDebug.start("DoomAgent::policy", 3);

	std::vector<float> actionValues = _network.getActionValueSingle(state);	
	std::vector<double> actionProbs = softmax(actionValues);
		
	//std::mt19937 e(std::random_device{}()); 
	//std::bernoulli_distribution d;
	//std::vector<double> lastAction;
	//for(unsigned i=0; i<_numActions; ++i)
	//	lastAction.push_back(d(e));
	//return lastAction;

	doomDebug.end("DoomAgent::policy", 3, start);
	return randomActionWithProb(actionProbs);
}

std::vector<double> DoomAgent::agentStart(vizdoom::BufferPtr state)
{
	auto start = doomDebug.start("DoomAgent::agentStart", 1);
	
	_sumRewards = 0;
	_episodeSteps = 0;
	
	_lastState = state;
	_lastAction = policy(_lastState); 

	doomDebug.end("DoomAgent::agentStart", 1, start);
	return _lastAction;
}

std::vector<double> DoomAgent::agentStep(double reward, vizdoom::BufferPtr state)
{
	auto time = doomDebug.start("DoomAgent::agentStep", 1);

	_sumRewards += reward;
	_episodeSteps += 1;
	
	std::vector<double> action = policy(state);
	//std::cout << "{ ";
	//for(unsigned i=0; i<action.size(); ++i)
	//	std::cout << action[i] << " ";
	//std::cout << "}" << std::endil;	

	// This block is for training.
	_replayBuffer.append(_lastState, _lastAction, reward, false, state);
	if(_replayBuffer.getBufferSize() > _replayBuffer.getMinibatchSize())
	{
		_network.cacheWeights();
		for(unsigned i=0; i<_numReplay; ++i)
		{
			std::vector<ExperienceSample> experiences = _replayBuffer.sample();
		
			//Clear TDUpdates structure here. This is not required any longer since TDUpdates are stored 
			//independently for each batch sample. Either this way or store TDUpdates in same location and implement mutex.
			//optimizeNetwork(experiences, currentQ);
				
			// Compute action value targets using Expected SARSA method.
			std::vector<std::vector<float>> qNextMat = _network.getActionValueTargets(experiences);
			std::vector<double> targetVec;
			for(unsigned j=0; j<qNextMat.size(); ++j)
			{
				std::vector<double> probsNextVec = softmax(qNextMat[j]);
				double vNext = 0;
				if(experiences[j].terminal == false)
					for(unsigned k=0; k<qNextMat[j].size(); ++k)
						vNext += probsNextVec[k]*qNextMat[j][k];

				double target = experiences[j].reward + _discount*vNext;
				targetVec.push_back(target);
			}

			// Compute action value predictions and TD Error.
			std::vector<std::vector<float>> qMat = _network.getActionValuePreds(experiences);
			std::vector<double> deltaVec;
			for(unsigned j=0; j<qMat.size(); ++j)
			{
				std::vector<double> eAction = experiences[j].action;
				unsigned a = 0;
				for(unsigned k=0; k<eAction.size(); ++k)
					a += (1 << k)*eAction[eAction.size()-1 - k];
				double delta = targetVec[j] - qMat[j][a];
				deltaVec.push_back(delta);
			}

			// Compute gradients and multiply with TD Error (delta).
			_network.getTDUpdate(experiences, deltaVec); // Maybe deltaMat is required.

			// Update weights here using ADAM optimizer.
		}
	}

	_lastState = state;
	_lastAction = action;

	doomDebug.end("DoomAgent::agentStep", 1, time);
	return action;

	// std::mt19937 e(std::random_device{}()); 
	// std::bernoulli_distribution d;

	// std::vector<double> action_random;
	//for(unsigned i=0; i<this->num_actions; ++i)
	//	action_random.push_back(d(e));

	//return action_random;
}

void DoomAgent::agentEnd(double reward)
{
	auto time = doomDebug.start("DoomAgent::agentEnd", 1);
		
	_sumRewards += reward;
	_episodeSteps += 1;

	//std::vector<double> action = this->policy(state); // No need for this sice state is terminal	
	vizdoom::BufferPtr state(new std::vector<uint8_t>(_lastState.get()->size(), 0));
	//std::shared_ptr<std::vector<uint8_t>>();
	_replayBuffer.append(_lastState, _lastAction, reward, false, state);

	if(_replayBuffer.getBufferSize() > _replayBuffer.getMinibatchSize())
	{
		//ActionValueNetwork currentQ = _network;
		for(unsigned i=0; i<_numReplay; ++i)
		{
			std::vector<ExperienceSample> experiences = _replayBuffer.sample();
			//optimizeNetwork(experiences, currentQ);
		}
	}

	//_lastState = state;
	//_lastAction = action;

	doomDebug.end("DoomAgent::agentEnd", 1, time);
	//return action
}

/*
double DoomAgent::agentMessage(std::string message) const
{
	std::cout << "DoomAgent::agentMessage" << std::endl;
	if(message == "get_sum_reward")
		return _sumRewards;
	else
		return 0;//raise Exception("Unrecognized Message!")
}*/

ActionValueNetwork *DoomAgent::getNetwork()
{
	auto time = doomDebug.start("DoomAgent::getNetwork", 3);
	
	doomDebug.end("DoomAgent::getNetwork", 3, time);
	return &_network;
}

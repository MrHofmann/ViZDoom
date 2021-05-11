#include "doomagent.h"
#include <algorithm>
#include <limits>


DoomAgent::DoomAgent(const AgentConfig &agentConf, const NetworkConfig &netConf, const OptimizerConfig &optConf)
	:_numActions(agentConf.numActions), _numReplay(agentConf.numReplay), _discount(agentConf.discount), _tau(agentConf.tau)
{
	std::cout << "DoomAgent::DoomAgent" << std::endl;

	_replayBuffer = ExperienceReplayBuffer(agentConf.replayBufferSize, agentConf.numMinibatch, agentConf.seed);
	_network = ActionValueNetwork(netConf);
	_optimizer = AdamOptimizer(netConf, optConf);
	
	//self.rand_generator = np.random.RandomState(agent_config.get("seed"))
	
	_lastState = nullptr;
	//self.last_action = None	
	_sumRewards = 0;
	_episodeSteps = 0;
}

void DoomAgent::optimizeNetwork(const std::vector<ExperienceSample> &exp, ActionValueNetwork q)
{
	std::cout << "DoomAgent::optimizeNetwork" << std::endl;

}

std::vector<double> DoomAgent::softmax(const std::vector<float> &actionValues) const
{
	//std::cout << "DoomAgent::softmax" << std::endl;

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

	return actionProbs;
}

std::vector<double> DoomAgent::randomActionWithProb(const std::vector<double> &probs) const
{
	//std::cout << "DoomAgent::randomActionWithProb" << std::endl;

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

	return action;
}

// This is a softmax policy
std::vector<double> DoomAgent::policy(vizdoom::BufferPtr state)
{
	//std::cout << "DoomAgent::policy" << std::endl;

	std::vector<float> actionValues = _network.getActionValuePreds(state);	
	std::vector<double> actionProbs	= softmax(actionValues);

	return randomActionWithProb(actionProbs);
}

std::vector<double> DoomAgent::agentStart(vizdoom::BufferPtr state)
{
	std::cout << "DoomAgent::agentStart" << std::endl;

	_sumRewards = 0;
	_episodeSteps = 0;
	
	//std::mt19937 e(std::random_device{}()); 
	//std::bernoulli_distribution d;
	//std::vector<double> lastAction;
	//for(unsigned i=0; i<_numActions; ++i)
	//	lastAction.push_back(d(e));

	_lastState = state;
	_lastAction = policy(_lastState); 

	return _lastAction;
}

std::vector<double> DoomAgent::agentStep(double reward, vizdoom::BufferPtr state)
{
	std::cout << "DoomAgent::agentStep" << std::endl;
	
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
		
			//Clear TDUpdates structure here.

			//optimizeNetwork(experiences, currentQ);
			////std::vector<std::vector<float>> qNextMat;
			////std::vector<std::vector<float>> qMat;
			std::vector<double> deltaVec;
			for(unsigned j=0; j<experiences.size(); ++j)
			{
				// Compute action value targets using Expected SARSA method.
				//std::vector<float> qNextVec = currentQ.getActionValues(experiences[j].nextState);
				std::vector<float> qNextVec = _network.getActionValueTargets(experiences[j].nextState);
				////qNextMat.push_back(qNextVec);
				std::vector<double> probsNextVec = softmax(qNextVec);
				double vNext = 0;
				for(unsigned k=0; k<qNextVec.size(); ++k)
					vNext += probsNextVec[k]*qNextVec[k];
				double target = experiences[j].reward + _discount*vNext;
				
				// Compute action value predictions and TD Error.
				std::vector<float> qVec = _network.getActionValuePreds(experiences[j].state);
				////qMat.push_back(qVec);
				std::vector<double> eAction = experiences[j].action;
				unsigned a = 0;
				for(unsigned k=0; k<eAction.size(); ++k)
					a += (1 << k)*eAction[eAction.size()-1 - k];
				double delta = target - qVec[a];
				deltaVec.push_back(delta);

				// Compute gradients and multiply with TD Error (delta).
				// Also clear TDUpdates structures in layers before beginning of loop.
				_network.getTDUpdate(j, eAction, delta);
			}

			// Update weights here using ADAM optimizer.
		}
	}

	_lastState = state;
	_lastAction = action;

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
	std::cout << "DoomAgent::agentEnd" << std::endl;
		
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
	return &_network;
}

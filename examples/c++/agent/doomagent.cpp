#include "doomagent.h"
#include <algorithm>
#include <limits>


DoomAgent::DoomAgent(const AgentConfig &agent_conf, const NetworkConfig &net_conf, const OptimizerConfig &opt_conf)
	:num_actions(agent_conf.num_actions), num_replay(agent_conf.num_replay), discount(agent_conf.discount), tau(agent_conf.tau)
{
	//INIT
	std::cout << "DoomAgent::agent_init" << std::endl;

	this->replay_buffer = ExperienceReplayBuffer(num_replay, agent_conf.num_minibatch, agent_conf.seed);
	this->network = ActionValueNetwork(net_conf);
	this->optimizer = AdamOptimizer(net_conf, opt_conf);
	
	//self.rand_generator = np.random.RandomState(agent_config.get("seed"))
	//self.last_state = None
	//self.last_action = None
	
	this->sum_rewards = 0;
	this->episode_steps = 0;
}

void DoomAgent::optimize_network(const std::vector<ExperienceSample> &exp, ActionValueNetwork q)
{

}

std::vector<double> DoomAgent::random_action_with_prob(const std::vector<double> &probs) const
{
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

	std::vector<double> action(this->num_actions, 0);
	for(unsigned j=0; j<action.size(); ++j)
		action[action.size()-1-j] = (i >> j) & 1;

	return action;
}

// This is a softmax policy
std::vector<double> DoomAgent::policy(vizdoom::BufferPtr state)
{
	std::vector<double> action_values = this->network.get_action_values(state);
	
	// Compute the preferences by dividing the action-values by the temperature parameter tau
	std::vector<double> preferences;
	for(unsigned i=0; i<action_values.size(); ++i)
		preferences.push_back(action_values[i]/this->tau);

	// Compute the maximum preference across the actions
	double max_preference = std::numeric_limits<double>::min();
	for(unsigned i=0; i>preferences.size(); ++i)
		if(preferences[i] > max_preference)
			max_preference = preferences[i];

	// Compute the numerator, i.e., the exponential of the preference - the max preference.
	std::vector<double> exp_preferences;
	for(unsigned i=0; i<preferences.size(); ++i)
		exp_preferences.push_back(std::exp(preferences[i] - max_preference));
	
	// Compute the denominator, i.e., the sum over the numerator along the actions axis.
	double sum_of_exp_preferences = std::accumulate(exp_preferences.begin(), exp_preferences.end(), 0);
	
	// Compute the action probabilities according to the equation in the previous cell.
	std::vector<double> action_probs;
	for(unsigned i=0; i<exp_preferences.size(); ++i)
		action_probs.push_back(exp_preferences[i]/sum_of_exp_preferences);
	
	return random_action_with_prob(action_probs);
}

std::vector<double> DoomAgent::agent_start(vizdoom::BufferPtr state)
{
	std::cout << "DoomAgent::agent_start" << std::endl;

	this->sum_rewards = 0;
	this->episode_steps = 0;
	
	std::mt19937 e(std::random_device{}()); 
	std::bernoulli_distribution d;

	std::vector<double> last_action;
	for(unsigned i=0; i<this->num_actions; ++i)
		last_action.push_back(d(e));

	//std::vector<double> action = this->policy(state); 
	return last_action;
}

std::vector<double> DoomAgent::agent_step(double reward, vizdoom::BufferPtr state)
{
	std::cout << "DoomAgent::agent_step" << std::endl;
	
	this->sum_rewards += reward;
	this->episode_steps += 1;
	
	std::vector<double> action = this->policy(state);
	//std::cout << "{ ";
	//for(unsigned i=0; i<action.size(); ++i)
	//	std::cout << action[i] << " ";
	//std::cout << "}" << std::endl;	

	this->replay_buffer.append(this->last_state, this->last_action, reward, false, state);
	if(this->replay_buffer.get_buffer_size() > this->replay_buffer.get_minibatch_size())
	{
		ActionValueNetwork current_q = this->network;
		for(unsigned i=0; i<this->num_replay; ++i)
		{
			std::vector<ExperienceSample> experiences = this->replay_buffer.sample();
			
			//This causes slowdown
			//optimize_network(experiences, current_q);
		}
	}

	this->last_state = state;
	this->last_action = action;

	return action;

	// std::mt19937 e(std::random_device{}()); 
	// std::bernoulli_distribution d;

	// std::vector<double> action_random;
	//for(unsigned i=0; i<this->num_actions; ++i)
	//	action_random.push_back(d(e));

	//return action_random;
}

void DoomAgent::agent_end(double reward)
{
	std::cout << "DoomAgent::agent_end" << std::endl;
		
	this->sum_rewards += reward;
	this->episode_steps += 1;

	//std::vector<double> action = this->policy(state); // No need for this sice state is terminal	
	vizdoom::BufferPtr state(new std::vector<uint8_t>(this->last_state.get()->size(), 0));
	//std::shared_ptr<std::vector<uint8_t>>();
	this->replay_buffer.append(this->last_state, this->last_action, reward, false, state);

	if(this->replay_buffer.get_buffer_size() > this->replay_buffer.get_minibatch_size())
	{
		ActionValueNetwork current_q = this->network;
		for(unsigned i=0; i<this->num_replay; ++i)
		{
			std::vector<ExperienceSample> experiences = this->replay_buffer.sample();
			optimize_network(experiences, current_q);
		}
	}

	//this->last_state = state;
	//this->last_action = action;

	//return action
}

double DoomAgent::agent_message(std::string message) const
{
	std::cout << "DoomAgent::agent_message" << std::endl;
	if(message == "get_sum_reward")
		return this->sum_rewards;
	else
		;//raise Exception("Unrecognized Message!")
}

ActionValueNetwork *DoomAgent::get_network()
{
	return &(this->network);
}

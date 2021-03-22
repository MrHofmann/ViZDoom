#ifndef DOOMAGENT_H
#define DOOMAGENT_H

#include <iostream>
#include "actionvaluenetwork.h"
#include "experiencereplaybuffer.h"
#include "datastructures.h"


class AdamOptimizer{
private:
	std::vector<unsigned> num_filters;
	std::vector<unsigned> num_hidden_units;
	double step_size;
	double beta_m;
	double beta_v;
	double epsilon;
	
	std::vector<std::vector<double>> m;
	std::vector<std::vector<double>> v;

public:
	AdamOptimizer(){}
	AdamOptimizer(const NetworkConfig &net_conf, const OptimizerConfig &opt_conf)
		:step_size(opt_conf.step_size), beta_m(opt_conf.beta_m), beta_v(opt_conf.beta_v), epsilon(opt_conf.epsilon)
	{
		//INIT
		std::cout << "AdamOptimizer init" << std::endl;
	}
	std::vector<std::vector<double>> update_weights(const std::vector<std::vector<double>> weights,
	const std::vector<std::vector<double>> td_errors_times_grads);
};


class DoomAgent{
private:
	ExperienceReplayBuffer replay_buffer;
	ActionValueNetwork network;
	AdamOptimizer optimizer;
	unsigned num_actions;
	unsigned num_replay;
	double discount;
	double tau;
	
	//self.rand_generator = np.random.RandomState(agent_config.get("seed"))

	//vizdoom::GameStatePtr game_state;	
	vizdoom::BufferPtr last_state;	
	std::vector<double> last_action;
	double sum_rewards;
	unsigned episode_steps;

	std::vector<double> random_action_with_prob(const std::vector<double> &probs) const;
	void optimize_network(const std::vector<ExperienceSample> &exp, ActionValueNetwork q);

public:
	DoomAgent()
	{	
		//INIT
		std::cout << "DoomAgent::agent_init" << std::endl;
	}
	DoomAgent(const AgentConfig &agent_conf, const NetworkConfig &net_conf, const OptimizerConfig &opt_conf);
	std::vector<double> policy(vizdoom::BufferPtr state);
	std::vector<double> agent_start(vizdoom::BufferPtr state);
	std::vector<double> agent_step(double reward, vizdoom::BufferPtr state);
	void agent_end(double reward);
	double agent_message(std::string message) const;

	ActionValueNetwork *get_network();
};

#endif // DOOMAGENT_H

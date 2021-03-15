#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include "ViZDoom.h"
#include <random>
#include <vector>
#include <string>
#include <map>


//enum LayerType {CONV, MAX_POOL, FC};
enum ActivationType {RELU, LRELU, TANH, SIGMOID};

struct Vertex;
struct Edge;

//typedef std::vector<std::vector<std::vector<Vertex*>>> Tensor3d;
//typedef std::vector<std::vector<Vertex*>> Tensor2d;
//typedef std::vector<Vertex*> Tensor1d;

struct NetworkConfig{
	//self.rand_generator = np.random.RandomState(network_config.get("seed"))
	std::vector<unsigned> state_dim;                // [W, H, C]
	std::vector<unsigned> num_filters;  			// [F1, F2, ... , Fn]
	std::vector<unsigned> filter_dim;       		// [F1s, F2s, ... , Fns]
	std::vector<unsigned> filter_strides;   		// [Sf1, Sf2, ... , Sfn]
	std::vector<ActivationType> activations; 		// [A1, A2, ... , An]
	std::vector<unsigned> pool_dim;         		// [P1, P2, ... , Pn]
	std::vector<unsigned> pool_strides;     		// [Sp1, Sp2, ... , Spn]
	std::vector<unsigned> num_hidden_units; 		// [H1, H2, ... , Hm]
	unsigned num_actions;
};

struct OptimizerConfig{
	double step_size;
	double beta_m;
	double beta_v;
	double epsilon;
};

struct AgentConfig{
	unsigned num_actions;
	unsigned num_replay;
	unsigned num_minibatch;
	unsigned seed;
	double discount;
	double tau;
};

struct ExperienceSample{
	vizdoom::BufferPtr state;
	std::vector<double> action;
	double reward;
	bool terminal;
	vizdoom::BufferPtr next_state;
};

struct Edge{
	double weight;
	double in_local_grad;
	double w_total_grad;
	Vertex *in_neuron;
	Vertex *out_neuron;
};

struct Vertex{
	ActivationType type;
	double activation;
	double inner_product;
	double in_products;
	double bias;
	double b_total_grad;
	std::vector<Edge*> in_edges;
	Edge* out_edge;
};

#endif // DATASTRUCTURES_H

#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include "ViZDoom.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <map>


enum ActivationType {RELU, LRELU, TANH, SIGMOID};

class Vertex;
class Conv3dVertex;

template <typename T>
struct Tensor3d {
	std::vector<std::vector<std::vector<T>>> elements;
	
	Tensor3d(){}
	Tensor3d(unsigned h, unsigned w, unsigned d)
	{
		elements = std::vector<std::vector<std::vector<T>>>(h, std::vector<std::vector<T>>(w, std::vector<T>(d, 0)));
	}
	
	Tensor3d(const std::vector<std::vector<std::vector<T>>> &e)
	{
		elements = e;
	}
	// Test this constructor.
	template <typename U>
	Tensor3d(Tensor3d<U> *t)
	{
		for(unsigned i=0; i<t->elements.size(); ++i)
	    {
	      std::vector<std::vector<Vertex*>> v1;
	      for(unsigned j=0; j<t->elements[i].size(); ++j)
	          v1.push_back(std::vector<Vertex*>(t->elements[i][j].begin(), t->elements[i][j].end()));

	      elements.push_back(v1);
	    }

	    //for(unsigned i=0; i<V->size(); ++i)
	        //for(unsigned j=0; j<(*V)[i].size(); ++j)
	            //for(unsigned k=0; k<(*V)[i][j].size(); ++k)
	                //if((*V)[i][j][k] == nullptr || (*V)[i][j][k] != (*C)[i][j][k])
	                    //std::cout << "FALSE" << std::endl;
	                //else
	                    //std::cout << "TRUE" << std::endl;	
	}

	// Test this also. (Tested this struct otside of project and it looked fine.)
	std::vector<std::vector<T>>& operator[](const int i)
	{
		return elements[i];
	}

	int size() const
	{
		return elements.size();
	}

	//template <typename U>
	//Tensor3d<T> & operator=(const Tensor3d<U> &t)
	//{
	//	std::cout << "HEREEEE" << std::endl;
	//	return *this;
	//}
};

template <typename T>
struct Tensor2d {
	std::vector<std::vector<T>> elements;
	
	Tensor2d<T>(){}
    Tensor2d<T>(unsigned h, unsigned w)
	{
		elements = std::vector<std::vector<T>>(h, std::vector<T>(w, 0));
	}
	//Tensor3d(unsigned h, unsigned w, unsigned d)
	//{
	//	elements = std::vector<std::vector<std::vector<T>>>(h, std::vector<std::vector<T>>(w, std::vector<T>(d, 0)));
	//}
	
	Tensor2d(const std::vector<std::vector<T>> &e)
	{
		elements = e;
	}

	// Test this constructor.
	template <typename U>
	Tensor2d(Tensor2d<U> *t)
	{
		for(unsigned i=0; i<t->elements.size(); ++i)
	    {
	      //std::vector<Vertex*> v1(t->elements[i].begin(), t->elements[i].end());
	      //v1.push_back(std::vector<Vertex*>(t->elements[i][j].begin(), t->elements[i][j].end()));

	      elements.push_back(std::vector<Vertex*>(t->elements[i].begin(), t->elements[i].end()));
	    }

	    //for(unsigned i=0; i<V->size(); ++i)
	        //for(unsigned j=0; j<(*V)[i].size(); ++j)
	            //for(unsigned k=0; k<(*V)[i][j].size(); ++k)
	                //if((*V)[i][j][k] == nullptr || (*V)[i][j][k] != (*C)[i][j][k])
	                    //std::cout << "FALSE" << std::endl;
	                //else
	                    //std::cout << "TRUE" << std::endl;	
	}

	// Test this also. (Tested this struct otside of project and it looked fine.)
	std::vector<T>& operator[](const int i)
	{
		return elements[i];
	}

	int size() const
	{
		return elements.size();
	}
};


template <typename T>
struct Tensor1d
{
	std::vector<T> elements;
	
	Tensor1d<T>(){}
	Tensor1d<T>(int l)
	{
		elements = std::vector<T>(l, 0);
	}
	Tensor1d(const std::vector<T> &e)
	{
		elements = e;
    }
    template <typename U>
    Tensor1d(Tensor1d<U> *t)
    {
		elements = std::vector<Vertex*>(t->elements.begin(), t->elements.end());
    }

	T& operator[](const int i)
	{
		return elements[i];
	}

	int size() const
	{
		return elements.size();
	}
};

struct NetworkConfig{
	//self.rand_generator = np.random.RandomState(network_config.get("seed"))
	std::vector<unsigned> stateDim;                // [W, H, C]
	std::vector<unsigned> numFilters;  			// [F1, F2, ... , Fn]
	std::vector<unsigned> filterDim;       		// [F1s, F2s, ... , Fns]
	std::vector<unsigned> filterStrides;   		// [Sf1, Sf2, ... , Sfn]
	std::vector<ActivationType> activations; 		// [A1, A2, ... , An]
	std::vector<unsigned> poolDim;         		// [P1, P2, ... , Pn]
	std::vector<unsigned> poolStrides;     		// [Sp1, Sp2, ... , Spn]
	std::vector<unsigned> numHiddenUnits; 		// [H1, H2, ... , Hm]
	unsigned numActions;
};

struct OptimizerConfig{
	double stepSize;
	double betaM;
	double betaV;
	double epsilon;
};

struct AgentConfig{
	unsigned numActions;
	unsigned replayBufferSize;
	unsigned numReplay;
	unsigned numMinibatch;
	unsigned seed;
	double discount;
	double tau;
};

struct ExperienceSample{
	vizdoom::BufferPtr state;
	std::vector<double> action;
	double reward;
	bool terminal;
	vizdoom::BufferPtr nextState;
};

#endif // DATASTRUCTURES_H

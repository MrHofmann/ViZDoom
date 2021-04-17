#ifndef NETWORKVERTEX_H
#define NETWORKVERTEX_H

#include "datastructures.h"

class Vertex;

class Edge{
private:
	Vertex *_inputVertex;
	float *_outGrad;
	//Vertex *_outputVertex;

public:
	enum EdgeType {WEIGHTED, UNWEIGHTED};

	Edge(){}
	Edge(Vertex *u/*, Vertex *v*/, float *_outGrad);
	virtual EdgeType edgeType() const = 0;
	Vertex *inputVertex() const;
	float outGrad() const;
};

class WeightedEdge : public Edge{
private:
	float *_weight;
	float *_TDUpdate;
	//float *_inputGrad;

public:
	WeightedEdge(){}
	WeightedEdge(Vertex *u, /*Vertex *v,*/ float *og, float *w, float *tdu);
	virtual Edge::EdgeType edgeType() const;
	float weight() const;
};

class UnweightedEdge : public Edge{
public:
	using Edge::Edge;
	virtual EdgeType edgeType() const;
};


class Vertex{
private:
    float *_activation;
	float _inputGrad;
	std::vector<Edge*> _outputEdges;

public:
    enum VertexType {BIAS, INPUT, A_RELU, A_LRELU, A_TANH, A_SIGM, P_MAX, P_AVG};
    
	Vertex(){}
	Vertex(float *a, float ing);
    virtual VertexType vertexType() const = 0;
	float activation() const;
	void setActivation(float a);
	void addOutputEdge(Edge *e);
	std::vector<Edge*> outputEdges() const;
};

class BiasVertex : public Vertex{
public:
	BiasVertex(float *a, float ing);
	virtual VertexType vertexType() const;
};

class Input3dVertex : public Vertex{
public:
    using Vertex::Vertex;
    virtual VertexType vertexType() const;
};

class Conv3dVertex : public Vertex{
private:
    float *_dotProduct;
	float _actGrad;
    Tensor1d<WeightedEdge*> *_inputEdges;

public:
    Conv3dVertex(){}
	Conv3dVertex(float *a, float ing, float *dp, float ag, Tensor1d<WeightedEdge*> *ie);
	Tensor1d<WeightedEdge*> *inputEdges() const;
	void setDotProduct(float dp);
};

class Relu3dUnit : public Conv3dVertex{
public:
    using Conv3dVertex::Conv3dVertex;
    virtual VertexType vertexType() const;
};

class Pool3dVertex : public Vertex{
private:
    Tensor2d<UnweightedEdge*> *_inputEdges;

public:
	Pool3dVertex(float *a, float ing, Tensor2d<UnweightedEdge*> *ie);
	Tensor2d<UnweightedEdge*> *inputEdges() const;
};

class MaxPool3dUnit : public Pool3dVertex{
public:
    using Pool3dVertex::Pool3dVertex;
    virtual VertexType vertexType() const;
};

class Dense1dVertex : public Vertex{
private:
    float *_dotProduct;
    Tensor1d<WeightedEdge*> *_inputEdges;

public:
    Dense1dVertex(){}
	Dense1dVertex(float *a, float ing, float *dp, Tensor1d<WeightedEdge*> *ie);
	Tensor1d<WeightedEdge*> *inputEdges() const;
	void setDotProduct(float dp);
};

class Relu1dUnit : public Dense1dVertex{
public:
    using Dense1dVertex::Dense1dVertex;
    virtual VertexType vertexType() const;
};

#endif //NETWORKVERTEX_H

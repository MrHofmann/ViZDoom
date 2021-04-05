#ifndef NETWORKVERTEX_H
#define NETWORKVERTEX_H

#include "datastructures.h"

class Vertex;

class Edge{
private:
	Vertex *_inputVertex;
	Vertex *_outputVertex;

public:
	enum EdgeType {WEIGHTED, UNWEIGHTED};

	Edge(){}
	Edge(Vertex *u, Vertex *v);
	virtual EdgeType edgeType() const = 0;
	Vertex *inputVertex() const;
};

class WeightedEdge : public Edge{
private:
	float *_weight;
	float *_TDUpdate;
	//float *_inputGrad;

public:
	WeightedEdge(){}
	WeightedEdge(Vertex *u, Vertex *v, float *w, float *tdu);
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
	// Output edge pointer is probably redundant.
    //Edge *_outputEdge;

public:
    //enum VertexType {1D, 3D};
    enum VertexType {BIAS, INPUT, A_RELU, A_LRELU, A_TANH, A_SIGM, P_MAX, P_AVG};
    
	Vertex(){}
    //Vertex(double a, Edge *o);
	Vertex(float *a, float ing);
    virtual VertexType vertexType() const = 0;
	float activation() const;
	void setActivation(float a);
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
	//double _bias;
    Tensor1d<WeightedEdge*> *_inputEdges;

public:
    Conv3dVertex(){}
    //Conv3dVertex(double ip, double b, double a, Tensor3d<WeightedEdge*> *ie, Edge *o);
	Conv3dVertex(float *a, float ing, float *dp, float ag, Tensor1d<WeightedEdge*> *ie);
	Tensor1d<WeightedEdge*> *inputEdges() const;
	//double bias() const;
	void setDotProduct(float dp);
};

class Relu3dUnit : public Conv3dVertex{
public:
    using Conv3dVertex::Conv3dVertex;
    virtual VertexType vertexType() const;
};

class Pool3dVertex : public Vertex{
private:
    Tensor3d<UnweightedEdge*> *_inputEdges;

public:
    //Pool3dVertex(double a, Tensor3d<WeightlessEdge*> *ie, Edge *o);
	Pool3dVertex(float *a, float ing, Tensor3d<UnweightedEdge*> *ie);
	Tensor3d<UnweightedEdge*> *inputEdges() const;
};

class MaxPool3dUnit : public Pool3dVertex{
public:
    using Pool3dVertex::Pool3dVertex;
    virtual VertexType vertexType() const;
};

class Dense1dVertex : public Vertex{
private:
    float *_dotProduct;
	//double _bias;
    Tensor1d<WeightedEdge*> *_inputEdges;

public:
    Dense1dVertex(){}
    //Dense1dVertex(double ip, double b, double a, Tensor1d<WeightedEdge*> *ie, Edge *o);
	Dense1dVertex(float *a, float ing, float *dp, Tensor1d<WeightedEdge*> *ie);
	Tensor1d<WeightedEdge*> *inputEdges() const;
	void setDotProduct(float dp);
	//double bias() const;
};

class Relu1dUnit : public Dense1dVertex{
public:
    using Dense1dVertex::Dense1dVertex;
    virtual VertexType vertexType() const;
};

#endif //NETWORKVERTEX_H

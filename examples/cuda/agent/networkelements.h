#ifndef NETWORKVERTEX_H
#define NETWORKVERTEX_H

#include "datastructures.h"

class Vertex;

class Edge{
private:
	Vertex *_inputVertex;
	Vertex *_outputVertex;

public:
	enum EdgeType {WEIGHTED, WEIGHTLESS};
	Edge(){}
	Edge(Vertex *u, Vertex *v);
	virtual EdgeType edgeType() const = 0;
	Vertex *inputVertex() const;
};

class WeightedEdge : public Edge{
private:
	double _weight;
	double _inLocalGrad;
	double _wTotalGrad;

public:
	WeightedEdge(){}
	WeightedEdge(double w, double ilg, double wtg, Vertex *u, Vertex *v);
	virtual Edge::EdgeType edgeType() const;
	double weight() const;
};

class WeightlessEdge : public Edge{
public:
	using Edge::Edge;
	virtual EdgeType edgeType() const;
};


class Vertex{
private:
    double _activation;
	// Output edge pointer is probably redundant.
    Edge *_outputEdge;

public:
    //enum VertexType {1D, 3D};
    enum VertexType {A_RELU, A_LRELU, A_TANH, A_SIGM, P_MAX, P_AVG};
    
	Vertex(){}
    Vertex(double a, Edge *o);
    virtual VertexType vertexType() const = 0;
	double activation() const;
	void setActivation(double a);
};

class Conv3dVertex : public Vertex{
private:
    //Tensor3d<double> *_inputProducts;
    Tensor3d<WeightedEdge*> *_inputEdges;
    double _innerProduct;
	double _bias;

public:
    Conv3dVertex(){}
    Conv3dVertex(double ip, double b, double a, Tensor3d<WeightedEdge*> *ie, Edge *o);
	Tensor3d<WeightedEdge*> *inputEdges() const;
	//Tensor3d<double> *inputProducts() const;
	double bias() const;
	//void setInputProduct(unsigned h, unsigned w, unsigned d, double ip);
	void setInnerProduct(double ip);
};

class Relu3dUnit : public Conv3dVertex{
public:
    using Conv3dVertex::Conv3dVertex;
    virtual VertexType vertexType() const;
};

class Pool3dVertex : public Vertex{
private:
    Tensor3d<WeightlessEdge*> *_inputEdges;

public:
    Pool3dVertex(double a, Tensor3d<WeightlessEdge*> *ie, Edge *o);
	Tensor3d<WeightlessEdge*> *inputEdges() const;
};

class MaxPool3dUnit : public Pool3dVertex{
public:
    using Pool3dVertex::Pool3dVertex;
    virtual VertexType vertexType() const;
};

class Dense1dVertex : public Vertex{
private:
    //Tensor1d<double> *_inputProducts;
    Tensor1d<WeightedEdge*> *_inputEdges;
    double _innerProduct;
	double _bias;

public:
    Dense1dVertex(){}
    Dense1dVertex(double ip, double b, double a, Tensor1d<WeightedEdge*> *ie, Edge *o);
	Tensor1d<WeightedEdge*> *inputEdges() const;
	//void setInputProduct(int i, double ip);
	void setInnerProduct(double ip);
	double bias() const;
};

class Relu1dUnit : public Dense1dVertex{
public:
    using Dense1dVertex::Dense1dVertex;
    virtual VertexType vertexType() const;
};

#endif //NETWORKVERTEX_H

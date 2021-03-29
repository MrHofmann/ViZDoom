#include "networkelements.h"

Edge::Edge(Vertex *u, Vertex *v)
	:_inputVertex(u), _outputVertex(v)
{
	//std::cout << "Edge::Edge" << std::endl;
}

Vertex *Edge::inputVertex() const
{
	//std::cout << "Edge::inputVertex" << std::endl;

	return _inputVertex;
}

WeightedEdge::WeightedEdge(double w, double ilg, double wtg, Vertex *u, Vertex *v)
	:Edge(u, v), _weight(w), _inLocalGrad(ilg), _wTotalGrad(wtg)
{
	//std::cout << "WeightedEdge::WeightedEdge" << std::endl;
}

Edge::EdgeType WeightedEdge::edgeType() const
{
	std::cout << "WeightedEdge::WeightedEdge" << std::endl;
	
	return Edge::WEIGHTED;
}

double WeightedEdge::weight() const
{
	//std::cout << "WeightedEdge::weight" << std::endl;

	return _weight;
}

Edge::EdgeType WeightlessEdge::edgeType() const
{
	std::cout << "WeightlessEdge::WeightlessEdge" << std::endl;
	
	return Edge::WEIGHTLESS;
}

Vertex::Vertex(double a, Edge *o)
	:_activation(a), _outputEdge(o)
{
	//std::cout << "Vertex::Vertex" << std::endl;
}

double Vertex::activation() const
{
	//std::cout << "Vertex::activation" << std::endl;

	return _activation;
}

void Vertex::setActivation(double a)
{
	//std::cout << "Vertex::setActivation" << std::endl;

	_activation = a;
}

Vertex::VertexType Input3dVertex::vertexType() const
{
	return Vertex::INPUT;
}

Conv3dVertex::Conv3dVertex(double ip, double b, double a, Tensor3d<WeightedEdge*> *ie, Edge *o)
	:Vertex(a, o), _inputEdges(ie), _innerProduct(ip), _bias(b)
{
	//std::cout << "Conv3dVertex::Conv3dVertex" << std::endl;
	
	//unsigned inputHeight = ie->size();
	//unsigned inputWidth = (*ie)[0].size();
	//unsigned inputDepth = (*ie)[0][0].size();
	//_inputProducts = new Tensor3d<double>(inputHeight, inputWidth, inputDepth);
}

Tensor3d<WeightedEdge*> *Conv3dVertex::inputEdges() const
{
	//std::cout << "Conv3dVertex::inputEdges" << std::endl;
	
	return _inputEdges;
}

double Conv3dVertex::bias() const
{
	//std::cout << "Conv3dVertex::bias" << std::endl;

	return _bias;
}

/*void Conv3dVertex::setInputProduct(unsigned h, unsigned w, unsigned d, double ip)
{
	//std::cout << "Conv3dVertex::setInputProduct" << std::endl;

	(*_inputProducts)[h][w][d] = ip;
}*/

void Conv3dVertex::setInnerProduct(double ip)
{
	//std::cout << "Conv3dVertex::setInnerProduct" << std::endl;

	_innerProduct = ip;
}

/*Tensor3d<double> *Conv3dVertex::inputProducts() const
{
	//std::cout << "Conv3dVertex::inputProducts" << std::endl;

	return _inputProducts;
}*/

Vertex::VertexType Relu3dUnit::vertexType() const
{
	std::cout << "Relu3dUnit::vertexType" << std::endl;
	
	return Vertex::A_RELU;
}

Pool3dVertex::Pool3dVertex(double a, Tensor3d<WeightlessEdge*> *ie, Edge *o)
	:Vertex(a, o), _inputEdges(ie)
{
	//std::cout << "Pool3dVertex::Pool3dVertex" << std::endl;
}

Tensor3d<WeightlessEdge*> *Pool3dVertex::inputEdges() const
{
	//std::cout << "Pool3dVertex::inputEdges" << std::endl;

	return _inputEdges;
}

Vertex::VertexType MaxPool3dUnit::vertexType() const
{
	std::cout << "MaxPool3dUnit::vertexType" << std::endl; 
	
	return Vertex::P_MAX;
}

Dense1dVertex::Dense1dVertex(double ip, double b, double a, Tensor1d<WeightedEdge*> *ie, Edge *o)
	:Vertex(a, o), _innerProduct(ip), _bias(b), _inputEdges(ie)
{
	//std::cout << "Dense1dVertex::Dense1dVertex" << std::endl;

	//unsigned inputSize = ie->size();
	//_inputProducts = new Tensor1d<double>(inputSize);
}

Tensor1d<WeightedEdge*> *Dense1dVertex::inputEdges() const
{
	//std::cout << "Dense1dVertex::inputEdges" << std::endl;

	return _inputEdges;
}
/*
void Dense1dVertex::setInputProduct(int i, double ip)
{
	//std::cout << "Dense1dVertex::setInputProduct" << std::endl;

	(*_inputProducts)[i] = ip;
}*/

void Dense1dVertex::setInnerProduct(double ip)
{
    //std::cout << "Dense1dVertex::setInnerProduct" << std::endl;
	
	_innerProduct = ip;
}

double Dense1dVertex::bias() const
{
	//std::cout << "Dense1dVertex::bias" << std::endl;

	return _bias;
}

Vertex::VertexType Relu1dUnit::vertexType() const
{
	std::cout << "Relu1dUnit::vertexType" << std::endl;
	
	return Vertex::A_RELU;
}

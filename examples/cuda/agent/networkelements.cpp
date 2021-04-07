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

WeightedEdge::WeightedEdge(Vertex *u, Vertex *v, float *w, float *tdu)
	:Edge(u, v), _weight(w), _TDUpdate(tdu)
{
	//std::cout << "WeightedEdge::WeightedEdge" << std::endl;
}

Edge::EdgeType WeightedEdge::edgeType() const
{
	//std::cout << "WeightedEdge::WeightedEdge" << std::endl;
	
	return Edge::WEIGHTED;
}

float WeightedEdge::weight() const
{
	//std::cout << "WeightedEdge::weight" << std::endl;

	return *_weight;
}

Edge::EdgeType UnweightedEdge::edgeType() const
{
	//std::cout << "UnweightedEdge::UnweightedEdge" << std::endl;
	
	return Edge::UNWEIGHTED;
}

Vertex::Vertex(float *a, float ing)
	:_activation(a), _inputGrad(ing)
{
	//std::cout << "Vertex::Vertex" << std::endl;
}

float Vertex::activation() const
{
	//std::cout << "Vertex::activation" << std::endl;

	return *_activation;
}

void Vertex::setActivation(float a)
{
	//std::cout << "Vertex::setActivation" << std::endl;

	*_activation = a;
}

//-----------
// BIAS VERTEX

BiasVertex::BiasVertex(float *a, float ing)
	:Vertex(a, ing)
{
	setActivation(1.0f);
}

Vertex::VertexType BiasVertex::vertexType() const
{
	return Vertex::BIAS;
}

//----------
// INPUT VERTEX

Vertex::VertexType Input3dVertex::vertexType() const
{
	return Vertex::INPUT;
}

Conv3dVertex::Conv3dVertex(float *a, float ing, float *dp, float ag, Tensor1d<WeightedEdge*> *ie)
	:Vertex(a, ing), _dotProduct(dp), _actGrad(ag), _inputEdges(ie)
{
	//std::cout << "Conv3dVertex::Conv3dVertex" << std::endl;
}

Tensor1d<WeightedEdge*> *Conv3dVertex::inputEdges() const
{
	//std::cout << "Conv3dVertex::inputEdges" << std::endl;
	
	return _inputEdges;
}

void Conv3dVertex::setDotProduct(float dp)
{
	//std::cout << "Conv3dVertex::setInnerProduct" << std::endl;

	*_dotProduct = dp;
}

Vertex::VertexType Relu3dUnit::vertexType() const
{
	std::cout << "Relu3dUnit::vertexType" << std::endl;
	
	return Vertex::A_RELU;
}

Pool3dVertex::Pool3dVertex(float *a, float ing, Tensor2d<UnweightedEdge*> *ie)
	:Vertex(a, ing), _inputEdges(ie)
{
	//std::cout << "Pool3dVertex::Pool3dVertex" << std::endl;
}

Tensor2d<UnweightedEdge*> *Pool3dVertex::inputEdges() const
{
	//std::cout << "Pool3dVertex::inputEdges" << std::endl;

	return _inputEdges;
}

Vertex::VertexType MaxPool3dUnit::vertexType() const
{
	std::cout << "MaxPool3dUnit::vertexType" << std::endl; 
	
	return Vertex::P_MAX;
}

Dense1dVertex::Dense1dVertex(float *a, float ing, float *dp, Tensor1d<WeightedEdge*> *ie)
	:Vertex(a, ing), _dotProduct(dp), _inputEdges(ie)
{
	//std::cout << "Dense1dVertex::Dense1dVertex" << std::endl;
}

Tensor1d<WeightedEdge*> *Dense1dVertex::inputEdges() const
{
	//std::cout << "Dense1dVertex::inputEdges" << std::endl;

	return _inputEdges;
}

void Dense1dVertex::setDotProduct(float dp)
{
    //std::cout << "Dense1dVertex::setInnerProduct" << std::endl;
	
	*_dotProduct = dp;
}

Vertex::VertexType Relu1dUnit::vertexType() const
{
	std::cout << "Relu1dUnit::vertexType" << std::endl;
	
	return Vertex::A_RELU;
}

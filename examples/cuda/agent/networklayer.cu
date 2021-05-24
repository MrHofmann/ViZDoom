#include "networklayer.h"
#include <limits>
#include <thrust/device_vector.h>

extern AgentDebug doomDebug;

NetworkLayer::NetworkLayer(std::string n, ActivationType a, const std::vector<unsigned> &s, NetworkLayer *pl)
	:_layerName(n), _activationType(a), _layerSize(s), _prevLayer(pl), _nextLayer(nullptr)
{
	auto start = doomDebug.start("NetworkLayer::NetworkLayer", 2);
	
	doomDebug.end("NetworkLayer::NetworkLayer", 2, start);
}

std::string NetworkLayer::layerName() const
{
	//std::cout << "NetworkLayer::layerName" << std::endl;
	return _layerName;
}

ActivationType NetworkLayer::activationType() const
{
	//std::cout << "NetworkLayer::activationType" << std::endl;
	
	return _activationType;
}

std::vector<unsigned> NetworkLayer::layerSize() const
{
	//std::cout << "NetworkLayer::layerSize" << std::endl;
	return _layerSize;
}

BiasVertex *NetworkLayer::biasVertex() const
{
	//std::cout << "NetworkLayer::biasVertex" << std::endl;

	return _bias;
}

std::vector<float> NetworkLayer::activations() const
{
	//std::cout << "NetworkLayer::activations" << std::endl;

	return _activations;
}

void NetworkLayer::setNextLayer(NetworkLayer *nl)
{
	//std::cout << "NetworkLayer::setNextLayer" << std::endl;

	_nextLayer = nl;
}

//----------------------------------------------------------------------------------------------------------------------------------------//

InputLayer::InputLayer(std::string ln, ActivationType at, const std::vector<unsigned> &ls, NetworkLayer *pl)
	:NetworkLayer(ln, at, ls, pl)
{
	auto start = doomDebug.start("InputLayer::InputLayer", 2);

	_states = {};
	int layerTotalSize = _layerSize[1]*_layerSize[2]*_layerSize[3];
	_activations = std::vector<float>(_layerSize[0]*layerTotalSize);
	_vertices = nullptr;
	_bias = nullptr;

	//_vertices = new Tensor3d<Input3dVertex*>(_layerSize[1], _layerSize[2], _layerSize[3]); 	// Remove later.
	//_bias = new BiasVertex(nullptr, 0);														// Remove later.
	//for(unsigned i=0; i<_layerSize[1]; ++i)													// Remove later.
	//	for(unsigned j=0; j<_layerSize[2]; ++j)												// Remove later.
	//		for(unsigned k=0; k<_layerSize[3]; ++k)											// Remove later.
	//		{																				// Remove later.
	//			int actIndex = i*_layerSize[1]*_layerSize[2] + j*_layerSize[3] + k;			// Remove later.
	//			(*_vertices)[i][j][k] = new Input3dVertex(&_activations[actIndex], 0);		// Remove later.
	//		}
	
	doomDebug.end("InputLayer::InputLayer", 2, start);
}

NetworkLayer::LayerType InputLayer::layerType() const
{
	//std::cout << "InputLayer::layerType" << std::endl;
	
	return NetworkLayer::INPUT;
}

void InputLayer::forwardProp(PropagationType p)
{
	auto start = doomDebug.start("InputLayer::forwardProp", 3);

	unsigned batch = _states.size();
	unsigned height = _layerSize[1];
	unsigned width = _layerSize[2];
	unsigned depth = _layerSize[3];

	for(unsigned i=0; i<batch; ++i)
	{
	/*	for(unsigned h=0; h<height; ++h)
			for(unsigned w=0; w<width; ++w)
				for(unsigned d=0; d<depth; ++d)
				{
					//(*_vertices)[i][j][k]->setActivation((*_state)[i*width*depth + j*depth + k]);
					_activations[i*height*width*depth + h*width*depth + w*depth + d] = (*_states[i])[h*width*depth + w*depth + d];
				}
	*/
		std::copy((*_states[i]).begin(), (*_states[i]).end(), _activations.begin() + i*height*width*depth);
	}

	//REMOVE BIAS!!!
	//_activations[batch*height*width*depth] = 1.0f;
	
	doomDebug.end("InputLayer::forwardProp", 3, start);
}

void InputLayer::backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec)
{
	auto start = doomDebug.start("InputLayer::backProp", 3);
	
	doomDebug.end("InputLayer::backProp", 3, start);
}

Tensor3d<Input3dVertex*>* InputLayer::vertices() const
{
	//std::cout << "InputLayer::vertices" << std::endl;

	return _vertices;
}


void InputLayer::setState(const std::vector<vizdoom::BufferPtr> &s)
{
	//std::cout << "InputLayer::setState" << std::endl;

	_states = s;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------//

Conv3dLayer::Conv3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned fdi, unsigned fde, unsigned fs)
	:NetworkLayer(ln, at, ls, prevLayer), _filterDim(fdi), _filterDepth(fde), _filterStride(fs)
{
	auto start = doomDebug.start("Conv3dLayer::Conv3dLayer", 2);

	// Cannot cast Tensor3d<Conv3dVertex*>* / Tensor3d<Pool3dVertex*>* to Tensor3d<Vertex*>* since overload of pointer assignment operator is not allowed. The result
	// of this is that *prevVertices is not pointing to original Tensor3d<Conv3dVertex*>/Tensor3d<Pool3dVertex*> struct but to a new struct whcih holds pointers
	// identical to the ones in the original struct. This new struct has to be deleted upon exit of this constructor to avoid memory leak, but carefuly not to
	// delete the vertices to which pointers inside this struct point to.
    //Tensor3d<Vertex*> *prevVertices = nullptr;																				// Remove later.
	//if(prevLayer->layerType() == NetworkLayer::INPUT)																		// Remove later.
	//	prevVertices = new Tensor3d<Vertex*>(((InputLayer*)prevLayer)->vertices());											// Remove later.
	//else if(prevLayer->layerType() == NetworkLayer::CONV)																	// Remove later.
	//	//prevVertices = ((Conv3dLayer*)prevLayer)->vertices();																// Remove later.
	//	prevVertices = new Tensor3d<Vertex*>(((Conv3dLayer*)prevLayer)->vertices());										// Remove later.
	//else if(prevLayer->layerType() == NetworkLayer::MAX_POOL)																// Remove later.
	//	//prevVertices = ((Pool3dLayer*)prevLayer)->vertices();																// Remove later.
	//	prevVertices = new Tensor3d<Vertex*>(((Pool3dLayer*)prevLayer)->vertices());										// Remove later.
	if(_prevLayer == nullptr || _prevLayer->layerType() == NetworkLayer::FC)
	{
		std::cout << "Unexpected previous layer type. Previous layer should be INPUT, CONV or POOL." << std::endl;
		std::exit(1);
	}

	int layerTotalSize = _layerSize[1]*_layerSize[2]*_layerSize[3];
	int filterTotalSize = _filterDim*_filterDim*_filterDepth;
	// Total number of weights is 3d-filter size times number of filters plus one bias for each filter. That is for each filter (depth dimension) there are
	// (filterTotalSize + 1) weights. (filterTotalSize + 1)*_layerSize[3]
	_weights = std::vector<float>(filterTotalSize*_layerSize[3] + _layerSize[3]);
	_cachedWeights = std::vector<float>(filterTotalSize*_layerSize[3] + _layerSize[3]);
	_TDUpdates = std::vector<float>(_layerSize[0]*(filterTotalSize*_layerSize[3] + _layerSize[3]));
	_outGrads = std::vector<float>(_layerSize[0]*(filterTotalSize + 1)*layerTotalSize);										// Not sure if outGrad of bias is required. Check later.
	_dotProducts = std::vector<float>(_layerSize[0]*layerTotalSize);	
	_activations = std::vector<float>(_layerSize[0]*layerTotalSize);
	_vertices = nullptr;
	_bias = nullptr;

	//_vertices = new Tensor3d<Conv3dVertex*>(_layerSize[1], _layerSize[2], _layerSize[3]);									// Remove later.
	//_bias = new BiasVertex(nullptr, 0);																						// Remove later.
	//for(unsigned i=0; i<_layerSize[1]; ++i)																					// Remove later.
	//	for(unsigned j=0; j<_layerSize[2]; ++j)																				// Remove later.
	//		for(unsigned k=0; k<_layerSize[3]; ++k)																			// Remove later.
	//		{																												// Remove later.
	//			int vIndex = i*_layerSize[2]*_layerSize[3] + j*_layerSize[3] + k;											// Remove later.
	//			//Tensor3d<WeightedEdge*> *inputEdges = new Tensor3d<WeightedEdge*>(_filterDim, _filterDim, _filterDepth);	// Remove later.
	//			Tensor1d<WeightedEdge*> *inputEdges = new Tensor1d<WeightedEdge*>(filterTotalSize + 1);						// Remove later.
	//			Conv3dVertex *v = new Relu3dUnit(&_activations[vIndex], 0.0f, &_dotProducts[vIndex], 0.0f, inputEdges);		// Remove later.
	//			for(unsigned h=0; h<_filterDim; ++h)																		// Remove later.
	//				for(unsigned w=0; w<_filterDim; ++w)																	// Remove later.
	//					for(unsigned d=0; d<_filterDepth; ++d)																// Remove later.
	//					{																									// Remove later.
	//						int eIndex = k*(filterTotalSize + 1) + h*_filterDim*_filterDepth + w*_filterDepth + d;			// Remove later.
	//						int gIndex = vIndex*(filterTotalSize + 1) + h*_filterDim*_filterDepth + w*_filterDepth + d;		// Remove later.
	//						WeightedEdge *e = new WeightedEdge((*prevVertices)[i+h][j+w][d], /*v,*/ &_outGrads[gIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
	//						//(*inputEdges)[h][w][d] = e;																	// Remove later.
	//						(*inputEdges)[h*_filterDim*_filterDepth + w*_filterDepth + d] = e;								// Remove later.
	//						(*prevVertices)[i+h][j+w][d]->addOutputEdge(e);													// Remove later.
	//					}																									// Remove later.
	//
	//			int eIndex = k*(filterTotalSize + 1) + filterTotalSize;														// Remove later.
	//			int gIndex = vIndex*(filterTotalSize + 1) + filterTotalSize;												// Remove later.
	//			WeightedEdge *e = new WeightedEdge(biasVertex(), /*v,*/ &_outGrads[gIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
	//			(*inputEdges)[filterTotalSize] = e;																			// Remove later.
	//			biasVertex()->addOutputEdge(e);																				// Remove later.
	//			(*_vertices)[i][j][k] = v;																					// Remove later.
	//		}																												// Remove later.
	
	doomDebug.end("Conv3dLayer::Conv3dLayer", 2, start);
}

NetworkLayer::LayerType Conv3dLayer::layerType() const
{
	//std::cout << "Conv3dLayer::layerType" << std::endl;
	
	return NetworkLayer::CONV;
}

struct Conv3dTransform{
	float *_input;
	float *_weights;
	int _filterDim;
	int _filterDepth;
	int _inputHeight;
	int _inputWidth;
	int _inputDepth;
	int _layerHeight;
	int _layerWidth;
	int _layerDepth;

	Conv3dTransform(float *i, float *w, int fdi, int fde, int ih, int iw, int id, int lh, int lw, int ld)
		:_input(i), _weights(w), _filterDim(fdi), _filterDepth(fde), 
		_inputHeight(ih), _inputWidth(iw), _inputDepth(id), 
		_layerHeight(lh), _layerWidth(lw), _layerDepth(ld)
	{
			
	}
	__host__ __device__ thrust::tuple<float, float> operator()(size_t vidx)
	{
		// vidx = b*outputHeight*outputWidth*outputDepth + i*outputWidth*outputDepth + j*outputDepth + k
		int b = vidx/(_layerHeight*_layerWidth*_layerDepth);
		int ijk = vidx - b*_layerHeight*_layerWidth*_layerDepth;
		int i = ijk/(_layerWidth*_layerDepth);
		int jk = ijk - i*_layerWidth*_layerDepth;
		int j = jk/_layerDepth;
		int k = jk - j*_layerDepth;

		int widx = k*(_filterDim*_filterDim*_filterDepth + 1);
		// Input vertex index should be the same as the output one? No, this was a bug. The input vertex index is always in the
		// first input channel. Thats why there is no + k at the end of right hand side of the expression bellow.
		int iidx = b*_inputHeight*_inputWidth*_inputDepth + i*_inputWidth*_inputDepth + j*_inputDepth;
		float dotProduct = 0.0f;
		float activation;
		for(unsigned h=0; h<_filterDim; ++h)
			for(unsigned w=0; w<_filterDim; ++w)
				for(unsigned d=0; d<_filterDepth; ++d)
				{
					//int wx = k*(filterTotalSize + 1) + h*_filterDim*_filterDepth + w*_filterDepth + d;
					int wx = widx + h*_filterDim*_filterDepth + w*_filterDepth + d;
					int ix = iidx + h*_inputWidth*_inputDepth + w*_inputDepth + d;
					dotProduct += _input[ix]*_weights[wx];
				}

		dotProduct += _weights[widx + _filterDim*_filterDim*_filterDepth];
		// Batch normalization goes here.
		activation = (dotProduct > 0)? dotProduct : 0;
		return thrust::make_tuple(dotProduct, activation);
	}
};

void Conv3dLayer::forwardProp(PropagationType p)
{
	auto start = doomDebug.start("Conv3dLayer::forwardProp", 3);

	int inputHeight = _prevLayer->layerSize()[1];
	int inputWidth = _prevLayer->layerSize()[2];
	int inputDepth = _prevLayer->layerSize()[3];
	int inputSize, layerSize;
	if(p == SINGLE)
	{
		inputSize = inputHeight*inputWidth*inputDepth;
		layerSize = _layerSize[1]*_layerSize[2]*_layerSize[3];
	}
	else
	{
		inputSize = _prevLayer->layerSize()[0]*inputHeight*inputWidth*inputDepth;
		layerSize = _layerSize[0]*_layerSize[1]*_layerSize[2]*_layerSize[3];
	}

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.begin() + inputSize);
	thrust::device_vector<float> weights(_weights.size());	
	thrust::device_vector<float> dotProducts(layerSize);
	thrust::device_vector<float> activations(layerSize);
	if(p == TARGET)
		thrust::copy(_cachedWeights.begin(), _cachedWeights.end(), weights.begin());
	else
		thrust::copy(_weights.begin(), _weights.end(), weights.begin());

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(layerSize), 
		thrust::make_zip_iterator(thrust::make_tuple(dotProducts.begin(), activations.begin())), 
		Conv3dTransform(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(weights.data()), 
			_filterDim, _filterDepth, inputHeight, inputWidth, inputDepth, _layerSize[1], _layerSize[2], _layerSize[3]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vectors as well. Also, make sure location of output vector is not changed.
	thrust::copy(dotProducts.begin(), dotProducts.end(), _dotProducts.begin());
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
	
	doomDebug.end("Conv3dLayer::forwardProp", 3, start);
}

struct Conv3dGetInGrads{
	float *_nextOutGrads;
	int _layerHeight;
	int _layerWidth;
	int _layerDepth;
	int _nextPoolDim;
	int _nextLayerHeight;
	int _nextLayerWidth;
	int _nextLayerDepth;

	Conv3dGetInGrads(float *nog, int lh, int lw, int ld, int npd, int nlh, int nlw, int nld)
		:_nextOutGrads(nog), _layerHeight(lh), _layerWidth(lw), _layerDepth(ld), 
		_nextPoolDim(npd), 	_nextLayerHeight(nlh), _nextLayerWidth(nlw), _nextLayerDepth(nld)
	{}
	__host__ __device__ float operator()(size_t vidx)
	{
		int b = vidx/(_layerHeight*_layerWidth*_layerDepth);
		int ijk = vidx - b*_layerHeight*_layerWidth*_layerDepth; 
		int i = ijk/(_layerWidth*_layerDepth);
		int jk = ijk - i*_layerWidth*_layerDepth;
		int j = jk/_layerDepth;
		int k = jk - j*_layerDepth;

		float inGrad = 0.0f;
		//int vidx = vi*_layerWidth*_layerDepth + vj*_layerDepth + c;
		int hMin = (i > _nextPoolDim - 1)? i - (_nextPoolDim - 1) : 0;
		int wMin = (j > _nextPoolDim - 1)? j - (_nextPoolDim - 1) : 0;
		int hMax = (hMin + _nextPoolDim > _nextLayerHeight)? _nextLayerHeight - hMin : _nextPoolDim;
		int wMax = (wMin + _nextPoolDim > _nextLayerWidth)? _nextLayerWidth - wMin : _nextPoolDim;
		for(unsigned h=0; h<hMax; ++h)
			for(unsigned w=0; w<wMax; ++w)
			{
				// Scope of nVidx is one batch sample.
				int nVidx = (hMin + h)*_nextLayerWidth*_nextLayerDepth + (wMin + w)*_nextLayerDepth + k;
				//_outGrads = _batchSize*_poolDim*_poolDim*_layerHeight*_layerWidth*_layerDepth;
				int gidx = b*_nextPoolDim*_nextPoolDim*_nextLayerHeight*_nextLayerWidth*_nextLayerDepth + 
					nVidx*(_nextPoolDim*_nextPoolDim) + (_nextPoolDim - 1 - h)*_nextPoolDim + (_nextPoolDim - 1 - w);
//				printf("%d %d %d %d %d %d %d, %d %d %d %d %d %d, %d %d %d\n", nVidx, hMin, hMax, h, wMin, wMax, w, gidx, b, _nextPoolDim, _nextLayerHeight, _nextLayerWidth, _nextLayerDepth, i, j, k);
				inGrad += _nextOutGrads[gidx];
			}			

		return inGrad;
	}
};


struct Conv3dBack{
	float *_inGrads;
	float *_dotProducts;
	float *_prevAct;
	float *_weights;
	float *_deltas;
	float *_outGrads;
	unsigned _filterDim;
	unsigned _filterDepth;
	unsigned _layerHeight;
	unsigned _layerWidth;
	unsigned _layerDepth;
	unsigned _prevLayerWidth;
	unsigned _prevLayerDepth;

	Conv3dBack(float *ig, float *dp, float *pa, float *w, float *d, float *og, 
			unsigned fdi, unsigned fde, unsigned lh, unsigned lw, unsigned ld, 
			unsigned plw, unsigned pld)
		:_inGrads(ig), _dotProducts(dp), _prevAct(pa), _weights(w), _deltas(d), _outGrads(og),
		_filterDim(fdi), _filterDepth(fde), _layerHeight(lh), _layerWidth(lw), _layerDepth(ld), 
		_prevLayerWidth(plw), _prevLayerDepth(pld)
	{}
	__host__ __device__ float operator()(size_t tidx)
	{
		// actGrad = gradRelu(inGrad, dotProduct)
		// outGrad = weight*actGrad = weight*gradRelu(inGrad, dotProduct)
		// weightGrad = prevAct*actGrad = prevAct*gradRelu(inGrad, dotProduct) 
		// tdUpdate = weightGrad*delta

		int weightsTotalSize = _filterDim*_filterDim*_filterDepth + 1;
		int bidx = tidx/(_layerDepth*weightsTotalSize);
		int widx = tidx - bidx*_layerDepth*weightsTotalSize;
		// The channel in which weight widx is located.
		int c = widx/weightsTotalSize;
		// 1d location of widx and its edge in input edges of each vertex that contains it in each channel.
		int r = widx - c*weightsTotalSize;

		// 3d location of widx in input edges.
		// If b == 1 then widx is bias.
		int b = r/(_filterDim*_filterDim*_filterDepth);
		int ijk = r - b*(_filterDim*_filterDim*_filterDepth);
		int i = ijk/_filterDim*_filterDepth;
		int jk = ijk - i*_filterDim*_filterDepth;
		int j = jk/_filterDepth;
		int k = jk - j*_filterDepth;

		// Fixating the k-th channel because weight widx is common only for that channel.
		float tdUpdate = 0;
		//float inGrad = 0.0f;
		for(unsigned vi=0; vi<_layerHeight; ++vi)
			for(unsigned vj=0; vj<_layerWidth; ++vj)
			{
				// inGrads size is one element shorter than _activations for current layer because of bias. Check this block.
				//int vidx = vi*_layerWidth*_layerDepth + vj*_layerDepth + c;
				int vidx = bidx*_layerHeight*_layerWidth*_layerDepth +  vi*_layerWidth*_layerDepth + vj*_layerDepth + c;
				float actGrad = (_dotProducts[vidx] >= 0)? _inGrads[vidx] : 0;
				int gidx = vidx*weightsTotalSize + r;
				_outGrads[gidx] = _weights[widx]*actGrad;

				// If current weight widx is bias then previous activation is one.
				if(b == 1)
					tdUpdate += actGrad*_deltas[bidx];
				else
				{
					// This is the vertex from previous layer that is connected via edge of widx. First is calculated the top left vertex 
					// in previous layer (vidx + skipped vertices because of filter size) and then adding the precise location of wanted
					// vertex in convolution block with respect to top left corner. It is easier to understand on a drawing.
					int aidx = vidx + vi*(_filterDim - 1)*_prevLayerDepth + i*_prevLayerWidth*_prevLayerDepth + j*_prevLayerDepth + k;
					tdUpdate += _prevAct[aidx]*actGrad*_deltas[bidx];					
				}
			}

		return tdUpdate;
	}
};

void Conv3dLayer::backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec)
{
	auto start = doomDebug.start("Conv3dLayer::backProp", 3);

	std::vector<float> grad = ((Pool3dLayer*)_nextLayer)->outGrads();
	thrust::device_vector<float> nextOutGrads(grad.begin(), grad.end());
	thrust::device_vector<float> inGrads(_activations.size(), 0);
	unsigned nextPoolDim = ((Pool3dLayer*)_nextLayer)->poolDim();
	std::vector<unsigned> nextLayerSize = _nextLayer->layerSize();
	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(inGrads.size()), inGrads.begin(), 
		Conv3dGetInGrads(thrust::raw_pointer_cast(nextOutGrads.data()), _layerSize[1], _layerSize[2], _layerSize[3], 
			nextPoolDim, nextLayerSize[1], nextLayerSize[2], nextLayerSize[3]));
	cudaDeviceSynchronize();
	
	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> prevAct(act.begin(), act.end());
	thrust::device_vector<float> dotProducts(_dotProducts.begin(), _dotProducts.end());
	thrust::device_vector<float> weights(_weights.begin(), _weights.end());
	thrust::device_vector<float> deltas(deltaVec.begin(), deltaVec.end());

	thrust::device_vector<float> outGrads(_outGrads.size());
	thrust::device_vector<float> tdUpdates(_TDUpdates.size());

	std::vector<unsigned> prevLayerSize = _prevLayer->layerSize();
	
	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(tdUpdates.size()), tdUpdates.begin(), 
		Conv3dBack(thrust::raw_pointer_cast(inGrads.data()), thrust::raw_pointer_cast(dotProducts.data()), thrust::raw_pointer_cast(prevAct.data()), 
			thrust::raw_pointer_cast(weights.data()), thrust::raw_pointer_cast(deltas.data()), thrust::raw_pointer_cast(outGrads.data()),
			_filterDim, _filterDepth, _layerSize[1], _layerSize[2], _layerSize[3], prevLayerSize[2], prevLayerSize[3]));
	cudaDeviceSynchronize();

	thrust::copy(outGrads.begin(), outGrads.end(), _outGrads.begin());
	thrust::copy(tdUpdates.begin(), tdUpdates.end(), _TDUpdates.begin());

	doomDebug.end("Conv3dLayer::backProp", 3, start);
}

void Conv3dLayer::cacheWeights()
{
	//std::cout << "Conv3dLayer::cacheWeights" << std::endl;

	_cachedWeights = _weights;
}

unsigned Conv3dLayer::filterDim() const
{
	//std::cout << "Conv3dLayer::filterDim" << std::endl;	

	return _filterDim;
}

unsigned Conv3dLayer::filterDepth() const
{	
	//std::cout << "Conv3dLayer::filterDepth" << std::endl;	

	return _filterDepth;
}

unsigned Conv3dLayer::filterStride() const
{
	//std::cout << "Conv3dLayer::filterStride" << std::endl;
	
	return _filterStride;
}

std::vector<float> Conv3dLayer::weights() const
{
	//std::cout << "Conv3dLayer::weights" << std::endl;

	return _weights;
}

std::vector<float> Conv3dLayer::dotProducts() const
{
	//std::cout << "Conv3dLayer::dotProducts" << std::endl;

	return _dotProducts;
}

std::vector<float> Conv3dLayer::outGrads() const
{
	//std::cout << "Conv3dLayer::outGrads" << std::endl;

	return _outGrads;
}

Tensor3d<Conv3dVertex*> *Conv3dLayer::vertices() const
{
	//std::cout << "Conv3dLayer::vertices" << std::endl;
	return _vertices;
}

void Conv3dLayer::setWeights(const std::vector<float> &w)
{
	//std::cout << "Conv3dLayer::setWeights" << std::endl;

	_weights.clear();
	std::copy(w.begin(), w.end(), std::back_inserter(_weights));
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------//
Pool3dLayer::Pool3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned pdi, unsigned ps)
    :NetworkLayer(ln, at, ls, prevLayer), _poolDim(pdi), _poolStride(ps)
{
	auto start = doomDebug.start("Pool3dLayer::Pool3dLayer", 2);

	if(prevLayer->layerType() != NetworkLayer::CONV)
	{
		std::cout << "Unexpected previous layer type. Previous layer must be CONV." << std::endl;
		std::exit(1);
	}

	//Tensor3d<Conv3dVertex*> *prevVertices = ((Conv3dLayer*)prevLayer)->vertices();											// Remove later.
	int layerTotalSize = _layerSize[1]*_layerSize[2]*_layerSize[3];
	_outGrads = std::vector<float>(_layerSize[0]*_poolDim*_poolDim*layerTotalSize);
	_activations = std::vector<float>(_layerSize[0]*layerTotalSize);
	_vertices = nullptr;
	_bias = nullptr;

	//_vertices = new Tensor3d<Pool3dVertex*>(_layerSize[1], _layerSize[2], _layerSize[3]);									// Remove later.
	//_bias = new BiasVertex(nullptr, 0);																						// Remove later.
	//for(unsigned i=0; i<_layerSize[1]; ++i)																					// Remove later.
	//	for(unsigned j=0; j<_layerSize[2]; ++j)																				// Remove later.
	//		for(unsigned k=0; k<_layerSize[3]; ++k)																			// Remove later.
	//		{																												// Remove later.
	//			int vIndex = i*_layerSize[2]*_layerSize[3] + j*_layerSize[3] + k;											// Remove later.
	//			Tensor2d<UnweightedEdge*> *inputEdges = new Tensor2d<UnweightedEdge*>(_poolDim, _poolDim);					// Remove later.
	//			MaxPool3dUnit *v = new MaxPool3dUnit(&_activations[vIndex], 0, inputEdges);									// Remove later.
	//			for(unsigned h=0; h<_poolDim; ++h)																			// Remove later.
	//				for(unsigned w=0; w<_poolDim; ++w)																		// Remove later.
	//				{																										// Remove later.
	//					int gIndex = vIndex*_poolDim*_poolDim + h*_poolDim + w;												// Remove later.
	//					UnweightedEdge *e = new UnweightedEdge((*prevVertices)[i+h][j+w][k]/*, v*/, &_outGrads[gIndex]);	// Remove later.
	//					(*inputEdges)[h][w] = e;																			// Remove later.
	//					(*prevVertices)[i+h][j+w][k]->addOutputEdge(e);														// Remove later.
	//				}																										// Remove later.
	//																														// Remove later.
	//			(*_vertices)[i][j][k] = v;																					// Remove later.
	//		}																												// Remove later.
	
	doomDebug.end("Pool3dLayer::Pool3dLayer", 2, start);
}

NetworkLayer::LayerType Pool3dLayer::layerType() const
{
	//std::cout << "Pool3dLayer::layerType" << std::endl;
	
	return NetworkLayer::MAX_POOL;
}

struct Pool3dTransform{
	float *_input;
	int _poolDim;
	int _inputHeight;
	int _inputWidth;
	int _inputDepth;
	int _layerHeight;
	int _layerWidth;
	int _layerDepth;

	Pool3dTransform(float *i, int pdi, int ih, int iw, int id, int lh, int lw, int ld)
		:_input(i), _poolDim(pdi), 
		_inputHeight(ih), _inputWidth(iw), _inputDepth(id), 
		_layerHeight(lh), _layerWidth(lw), _layerDepth(ld){}
	__host__ __device__  float operator()(size_t vidx)
	{
		// vidx = i*outputWidth*outputDepth + j*outputDepth + k
		int b = vidx/(_layerHeight*_layerWidth*_layerDepth);
		int ijk = vidx - b*_layerHeight*_layerWidth*_layerDepth;
		int i = ijk/(_layerWidth*_layerDepth);
		int jk = ijk - i*_layerWidth*_layerDepth;
		int j = jk/_layerDepth;
		int k = jk - j*_layerDepth;

		// Input vertex index should be the same as the output one? Yes, because unlike convolution, 
		// pooling is applied to each channel independently.
		int iidx = b*_inputHeight*_inputWidth*_inputDepth + i*_inputWidth*_inputDepth + j*_inputDepth + k;
		float activation = _input[iidx];
		for(unsigned h=0; h<_poolDim; ++h)
			for(unsigned w=0; w<_poolDim; ++w)
			{	
				int ix = iidx + h*_inputWidth*_inputDepth + w*_inputDepth;
				if(_input[ix] > activation)
					activation = _input[ix];
			}	
			
		// Is batch normalization required in pooling layers? I think not, since there are no learnable parameters or activation functions.
		// NO IT IS NOT REQUIRED!
		return activation;
	}

};

void Pool3dLayer::forwardProp(PropagationType p)
{
	auto start = doomDebug.start("Pool3dLayer::forwardProp", 3);
	
	int inputHeight = _prevLayer->layerSize()[1];
	int inputWidth = _prevLayer->layerSize()[2];
	int inputDepth = _prevLayer->layerSize()[3];
	int inputSize, layerSize;
	if(p == SINGLE)
	{
		inputSize = inputHeight*inputWidth*inputDepth;
		layerSize = _layerSize[1]*_layerSize[2]*_layerSize[3];
	}
	else
	{
		inputSize = _prevLayer->layerSize()[0]*inputHeight*inputWidth*inputDepth;
		layerSize = _layerSize[0]*_layerSize[1]*_layerSize[2]*_layerSize[3];
	}

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.begin() + inputSize);
	thrust::device_vector<float> activations(layerSize);

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(layerSize), 
		activations.begin(), Pool3dTransform(thrust::raw_pointer_cast(input.data()), _poolDim, //_poolDepth, 
			inputHeight, inputWidth, inputDepth, _layerSize[1], _layerSize[2], _layerSize[3]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vector as well. Also, make sure location of output vector is not changed.
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
	
	doomDebug.end("Pool3dLayer::forwardProp", 3, start);
}

struct Pool3dBackFromConv{
	float *_nextOutGrads;
	unsigned _poolDim;
	unsigned _nextFilterDim;
	unsigned _layerHeight;
	unsigned _layerWidth;
	unsigned _layerDepth;
	unsigned _nextLayerHeight;
	unsigned _nextLayerWidth;
	unsigned _nextLayerDepth;

	Pool3dBackFromConv(float *nog, unsigned pd, unsigned nfd, unsigned lh, unsigned lw, unsigned ld, unsigned nlh, unsigned nlw, unsigned nld)
		:_nextOutGrads(nog), _poolDim(pd), _nextFilterDim(nfd),
		_layerHeight(lh), _layerWidth(lw), _layerDepth(ld),
		_nextLayerHeight(nlh), _nextLayerWidth(nlw), _nextLayerDepth(nld)
	{}
	__host__ __device__ float operator()(size_t vidx)
	{
		// outGrad = gradPool(inGrad, prevAct)

		// vidx is index of vertex in current layer in current batch.
		int bidx = vidx/(_layerHeight*_layerWidth*_layerDepth);
		int ijk = vidx - bidx*_layerHeight*_layerWidth*_layerDepth;
		int i = ijk/(_layerWidth*_layerDepth);
		int jk = ijk - i*_layerWidth*_layerDepth;
		int j = jk/_layerDepth;
		int k = jk - j*_layerDepth;

		float inGrad = 0.0f;
		int hMin = (i > _nextFilterDim - 1)? i - (_nextFilterDim - 1) : 0;
		int wMin = (j > _nextFilterDim - 1)? j - (_nextFilterDim - 1) : 0;
		int hMax = (hMin + _nextFilterDim > _nextLayerHeight)? _nextLayerHeight - hMin : _nextFilterDim;
		int wMax = (wMin + _nextFilterDim > _nextLayerWidth)? _nextLayerWidth - wMin : _nextFilterDim;
		int bix = bidx*_nextLayerHeight*_nextLayerWidth*_nextLayerDepth;
		//printf("%d %d %d %d, %d %d %d %d", hMin, hMax, wMin, wMax, _poolDim, _nextLayerHeight, _nextLayerWidth, _nextLayerDepth);
		for(unsigned h=0; h<hMax; ++h)
			for(unsigned w=0; w<wMax; ++w)
				for(unsigned n=0; n<_nextLayerDepth; ++n)
				{
					int nVidx = bix + (hMin + h)*_nextLayerWidth*_nextLayerDepth + (wMin + w)*_nextLayerDepth + n;
					int gidx = nVidx*(_nextFilterDim*_nextFilterDim*_layerDepth + 1) + 
						(_nextFilterDim - 1 - h)*_nextFilterDim*_layerDepth + (_nextFilterDim - 1 - w)*_layerDepth + k;
					inGrad += _nextOutGrads[gidx];
				}

		return inGrad;
	}
};

struct Pool3dBackFromDense{
	float *_nextOutGrads;
	unsigned _poolDim;
	unsigned _layerHeight;
	unsigned _layerWidth;
	unsigned _layerDepth;
	unsigned _nextLayerSize;

	Pool3dBackFromDense(float *nog, unsigned pd, unsigned lh, unsigned lw, unsigned ld, unsigned nls)
		:_nextOutGrads(nog), _poolDim(pd), _layerHeight(lh), _layerWidth(lw), _layerDepth(ld), _nextLayerSize(nls)
	{}
	__host__ __device__ float operator()(size_t vidx)
	{
		// outGrad = gradPool(inGrad, prevAct)
		// vidx is index of vertex in current layer in current batch.

		float inGrad = 0.0f;
		int layerTotalSize = (_layerHeight*_layerWidth*_layerDepth + 1);
		for(int h=0; h<_nextLayerSize; ++h)
		{
			int gidx = h*layerTotalSize + vidx;
			inGrad += _nextOutGrads[gidx];
		}

		return inGrad;
	}
};

struct Pool3dBack{
	float *_prevAct;
	float *_inGrads;
	unsigned _poolDim;
	unsigned _layerHeight;
	unsigned _layerWidth;
	unsigned _layerDepth;
	unsigned _prevLayerHeight;
	unsigned _prevLayerWidth;
	unsigned _prevLayerDepth;

	Pool3dBack(float *pa, float *ig, unsigned pd, unsigned lh, unsigned lw, unsigned ld, unsigned plh, unsigned plw, unsigned pld)
		:_prevAct(pa), _inGrads(ig), _poolDim(pd), 
		_layerHeight(lh), _layerWidth(lw), _layerDepth(ld),
		_prevLayerHeight(plh), _prevLayerWidth(plw), _prevLayerDepth(pld)
	{}
	__host__ __device__ float operator()(size_t beidx)
	{
		// outGrad = gradPool(inGrad, prevAct)

		// vidx is index of vertex in current layer.
		int bidx = beidx/(_layerHeight*_layerWidth*_layerDepth*_poolDim*_poolDim);
		int eidx = beidx - bidx*_layerHeight*_layerWidth*_layerDepth*_poolDim*_poolDim;
		int vidx = eidx/(_poolDim*_poolDim);
		int i = vidx/(_layerWidth*_layerDepth);

		int hwp = eidx - vidx*_poolDim*_poolDim;
		int hp = hwp/_poolDim;
		int wp = hwp - hp*_poolDim;
		int vix = bidx*_layerHeight*_layerWidth*_layerDepth + vidx;		
		int aidx = vidx + i*(_poolDim - 1)*_layerDepth;
		int aix = bidx*_prevLayerHeight*_prevLayerWidth*_prevLayerDepth + aidx;
		// Check wether _layerDepth or _prevLayerDepth is used here.
		//int eix = aidx + hp*_poolDim*_layerDepth + wp*_layerDepth;
		int eix = aix + hp*_poolDim*_layerDepth + wp*_layerDepth;

		float outGrad = _inGrads[vix];
		for(unsigned h=0; h<_poolDim; ++h)
			for(unsigned w=0; w<_poolDim; ++w)
			{
				// Check wether _layerDepth or _prevLayerDepth is used here.
				//unsigned aix = aidx + h*_poolDim*_layerDepth + w*_layerDepth;
				int ai = aix + h*_poolDim*_layerDepth + w*_layerDepth;
				if(_prevAct[ai] > _prevAct[eix])
					return 0;
			}

		return outGrad;
	}
};

void Pool3dLayer::backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec)
{
	auto start1 = doomDebug.start("Pool3dLayer::backProp", 3);

	thrust::device_vector<float> inGrads(_layerSize[0]*_layerSize[1]*_layerSize[2]*_layerSize[3]);
	std::vector<unsigned> nextLayerSize = _nextLayer->layerSize();
	if(_nextLayer->layerType() == FC)
	{
		auto start2 = doomDebug.start("Pool3dLayer::backFromDense", 4);
		
		thrust::device_vector<float> grad = ((DenseLayer*)_nextLayer)->outGrads();
		thrust::device_vector<float> nextOutGrads(grad);
		thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(inGrads.size()), inGrads.begin(), 
			Pool3dBackFromDense(thrust::raw_pointer_cast(nextOutGrads.data()), _poolDim, 
				_layerSize[1], _layerSize[2], _layerSize[3], nextLayerSize[1]));
	
		doomDebug.end("Pool3dLayer::backFromDense", 4, start2);
	}
	else
	{
		auto start2 = doomDebug.start("Pool3dLayer::backFromConv", 4);
		
		unsigned nextFilterDim = ((Conv3dLayer*)_nextLayer)->filterDim();
		thrust::device_vector<float> grad = ((Conv3dLayer*)_nextLayer)->outGrads();
		thrust::device_vector<float> nextOutGrads(grad);
		thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(inGrads.size()), inGrads.begin(), 
			Pool3dBackFromConv(thrust::raw_pointer_cast(nextOutGrads.data()),  _poolDim, nextFilterDim, 
				_layerSize[1], _layerSize[2], _layerSize[3], nextLayerSize[1], nextLayerSize[2], nextLayerSize[3]));
		
		doomDebug.end("Pool3dLayer::backFromConv", 4, start2);
	}
	cudaDeviceSynchronize();

	std::vector<float> act = _prevLayer->activations();
	std::vector<unsigned> prevLayerSize = _prevLayer->layerSize();
	thrust::device_vector<float> prevAct(act.begin(), act.end());
	thrust::device_vector<float> outGrads(_outGrads.size());
	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(outGrads.size()), outGrads.begin(), 
		Pool3dBack(thrust::raw_pointer_cast(prevAct.data()), thrust::raw_pointer_cast(inGrads.data()), 
			_poolDim, _layerSize[1], _layerSize[2], _layerSize[3], prevLayerSize[1], prevLayerSize[2], prevLayerSize[3]));
	cudaDeviceSynchronize();

	thrust::copy(outGrads.begin(), outGrads.end(), _outGrads.begin());		
	
	doomDebug.end("Pool3dLayer::backProp", 3, start1);
}

unsigned Pool3dLayer::poolDim() const
{
	//std::cout << "Pool3dLayer::poolDim" << std::endl;	

	return _poolDim;
}

std::vector<float> Pool3dLayer::outGrads() const
{
	//std::cout << "Pool3dLayer::outGrads" << std::endl;

	return _outGrads;
}

Tensor3d<Pool3dVertex*>* Pool3dLayer::vertices() const
{
	//std::cout << "Pool3dLayer::vertices" << std::endl;
	
	return _vertices;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------//
DenseLayer::DenseLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned hu)
	:NetworkLayer(ln, at, ls, prevLayer), _numHiddenUnits(hu)
{
	auto start = doomDebug.start("DenseLayer::DenseLayer", 2);

	std::vector<unsigned> prevLayerSize = prevLayer->layerSize();
	int prevTotalSize = 1;
	for(unsigned i=1; i<prevLayerSize.size(); ++i)
		prevTotalSize *= prevLayerSize[i];

	std::vector<unsigned> curLayerSize = layerSize();
	_weights = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
	_cachedWeights = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
	_TDUpdates = std::vector<float>(_layerSize[0]*(prevTotalSize + 1)*_numHiddenUnits);
	_outGrads = std::vector<float>(_layerSize[0]*(prevTotalSize + 1)*_numHiddenUnits);									// Not sure if bias is required. Check later.
	_dotProducts = std::vector<float>(_layerSize[0]*_numHiddenUnits);
	_activations = std::vector<float>(_layerSize[0]*_numHiddenUnits);
	_vertices = nullptr;
	_bias = nullptr;

	//_vertices = new Tensor1d<Dense1dVertex*>(curLayerSize[1]);																	// Remove later.
	//_bias = new BiasVertex(nullptr, 0);																							// Remove later.
	//BiasVertex *prevBias = prevLayer->biasVertex();																				// Remove later.
	//for(unsigned i=0; i<curLayerSize[1]; ++i)																					// Remove later.
	//{																															// Remove later.
	//	Dense1dVertex *v;																										// Remove later.
	//	if(prevLayer->layerType() == NetworkLayer::FC)																			// Remove later.
	//	{																														// Remove later.
	//		Tensor1d<Dense1dVertex*> *prevVertices = ((DenseLayer*)prevLayer)->vertices();										// Remove later.
    //      std::vector<unsigned> prevLayerSize = prevLayer->layerSize();														// Remove later.
	//		Tensor1d<WeightedEdge*> *inputEdges = new Tensor1d<WeightedEdge*>(prevLayerSize[1] + 1);							// Remove later.
	//		v = new Relu1dUnit(&_activations[i], 0, &_TDUpdates[i], inputEdges);												// Remove later.
    //        for(unsigned j=0; j<prevLayerSize[1]; ++j)																			// Remove later.
	//		{																													// Remove later.
	//			int eIndex = i*(prevTotalSize + 1) + j;																			// Remove later.
	//			Vertex *u = (*prevVertices)[j];																					// Remove later.
	//			WeightedEdge *e = new WeightedEdge(u, /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);		// Remove later.
	//			(*inputEdges)[j] = e;																							// Remove later.
	//			u->addOutputEdge(e);																							// Remove later.
	//		}																													// Remove later.
	//																															// Remove later.	
	//		int eIndex = i*(prevTotalSize + 1) + prevTotalSize;																	// Remove later.
	//		WeightedEdge *e = new WeightedEdge(prevBias, /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);	// Remove later.
	//		(*inputEdges)[prevTotalSize] = e;																					// Remove later.
	//		prevBias->addOutputEdge(e);																							// Remove later.
	//	}																														// Remove later.
	//    else																													// Remove later.
	//	{																														// Remove later.
	//		// Same issue as in Conv3dLayer::Conv3dLayer. Will have to decide at some point whether should prevVertices be used as structs or pointers to structs. In
	//		// case pointers are NOT used then consider replacing pointers to vertices structs with only structs in Conv3dLayer and DenseLayer.
	//        Tensor3d<Vertex*> *prevVertices = nullptr;																			// Remove later.
	//        if(prevLayer->layerType() == NetworkLayer::CONV)																	// Remove later.
	//            //prevVertices = ((Conv3dLayer*)prevLayer)->vertices();															// Remove later.
	//			prevVertices = new Tensor3d<Vertex*>(((Conv3dLayer*)prevLayer)->vertices());									// Remove later.
	//        else if(prevLayer->layerType() == NetworkLayer::MAX_POOL)															// Remove later.
	//        	//prevVertices = ((Pool3dLayer*)prevLayer)->vertices();															// Remove later.
	//			prevVertices = new Tensor3d<Vertex*>(((Pool3dLayer*)prevLayer)->vertices());									// Remove later.
    //    	else																												// Remove later.
	//	    {   																												// Remove later.
	//			std::cout << "Unexpected previous layer type. Previous layer should be CONV, POOL or FC." << std::endl;			// Remove later.
	//			std::exit(1);																									// Remove later.
	//		}																													// Remove later.
	//																															// Remove later.
	//		std::vector<unsigned> prevLayerSize = prevLayer->layerSize();														// Remove later.
	//		Tensor1d<WeightedEdge*> *inputEdges = new Tensor1d<WeightedEdge*>(prevTotalSize + 1);								// Remove later.
	//		v = new Relu1dUnit(&_activations[i], 0, &_dotProducts[i], inputEdges);												// Remove later.
	//		for(unsigned h=0; h<prevLayerSize[1]; ++h)																			// Remove later.
	//			for(unsigned w=0; w<prevLayerSize[2]; ++w)																		// Remove later.
    //        		for(unsigned d=0; d<prevLayerSize[3]; ++d)																	// Remove later.
    //            	{																											// Remove later.
	//					int eIndex = i*(prevTotalSize + 1) + h*prevLayerSize[2]*prevLayerSize[3] + w*prevLayerSize[3] + d;		// Remove later.
	//           			WeightedEdge *e = new WeightedEdge((*prevVertices)[h][w][d], /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
	//					(*inputEdges)[h*prevLayerSize[2]*prevLayerSize[3] + w*prevLayerSize[3] + d] = e;						// Remove later.
	//					(*prevVertices)[h][w][d]->addOutputEdge(e);																// Remove later.
	//				}																											// Remove later.
	//																															// Remove later.
	//		int eIndex = i*(prevTotalSize + 1) + prevTotalSize;																	// Remove later.
	//        WeightedEdge *e = new WeightedEdge(prevBias, /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);	// Remove later.
	//		(*inputEdges)[prevTotalSize] = e;																					// Remove later.
	//		prevBias->addOutputEdge(e);																							// Remove later.
	//	}																														// Remove later.
	//																															// Remove later.
	//	(*_vertices)[i] = v;																									// Remove later.
	//}																															// Remove later.
	
	doomDebug.end("DenseLayer::DenseLayer", 2, start);
}

NetworkLayer::LayerType DenseLayer::layerType() const
{
	//std::cout << "DenseLayer::layerType" << std::endl;
	
	return NetworkLayer::FC;
}

struct Dense1dTransform{
	float *_input;
	float *_weights;
	int _inputHeight;
	int _layerHeight;

	Dense1dTransform(float *i, float *w, int ih, int lh)
		:_input(i), _weights(w),
		_inputHeight(ih), _layerHeight(lh){}
	__host__ __device__ thrust::tuple<float, float> operator()(size_t vidx)
	{
		int b = vidx/_layerHeight;
		int r = vidx - b*_layerHeight;
		int widx = r*(_inputHeight + 1);
		float dotProduct = 0.0f;
		float activation;
		for(unsigned h=0; h<_inputHeight; ++h)
		{
			int wx = widx + h;
			dotProduct += _input[b*_inputHeight + h]*_weights[wx];
		}
		dotProduct += _weights[widx + _inputHeight];
		
		// Batch normalization goes here.
		activation = (dotProduct > 0)? dotProduct : 0;
		return thrust::make_tuple(dotProduct, activation);
	}
};

void DenseLayer::forwardProp(PropagationType p)
{
	auto start = doomDebug.start("DenseLayer::forwardProp", 3);
	
	std::vector<unsigned> prevLayerSize = _prevLayer->layerSize();
	int prevSize = 1;
	for(unsigned i=1; i<prevLayerSize.size(); ++i)
		prevSize *= prevLayerSize[i];

	int inputSize, layerSize;
	if(p == SINGLE)
	{
		inputSize = prevSize;
		layerSize = _layerSize[1];
	}
	else
	{
		inputSize = prevLayerSize[0]*prevSize;
		layerSize = _layerSize[0]*_layerSize[1];
	}
	
	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.begin() + inputSize);
	thrust::device_vector<float> dotProducts(layerSize);
	thrust::device_vector<float> activations(layerSize);	
	thrust::device_vector<float> weights(_weights.size());
	if(p == TARGET)
		thrust::copy(_cachedWeights.begin(), _cachedWeights.end(), weights.begin());
	else
		thrust::copy(_weights.begin(), _weights.end(), weights.begin());

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(layerSize), 
		thrust::make_zip_iterator(thrust::make_tuple(dotProducts.begin(), activations.begin())), 
		Dense1dTransform(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(weights.data()), 
			prevSize, _layerSize[1]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vectors as well. Also, make sure location of output vector is not changed.
	thrust::copy(dotProducts.begin(), dotProducts.end(), _dotProducts.begin());
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
	
	doomDebug.end("DenseLayer::forwardProp", 3, start);
}

struct Dense1dGetInGrads{
	float *_nextOutGrads;
	// Both without bias.
	int _layerSize;
	int _nextLayerSize;

	Dense1dGetInGrads(float *nog, int ls, int nls)
		:_nextOutGrads(nog), _layerSize(ls), _nextLayerSize(nls)
	{}
	__host__ __device__ float operator()(size_t vidx)
	{	
		//int bix = _bidx*_layerSize*_nextLayerSize;
		int bidx = vidx/_layerSize;
		int vix = vidx - bidx*_layerSize;
		float inGrad = 0;
		for(unsigned i=0; i<_nextLayerSize; ++i)
		{
			int gidx = bidx*(_layerSize + 1)*_nextLayerSize + i*(_layerSize + 1) + vix;
			inGrad += _nextOutGrads[gidx];
		}

		return inGrad;
	}
};

struct Dense1dBack{
	float *_inGrads;
	float *_dotProducts;
	float *_prevAct;
	float *_weights;
	float *_deltas;
	// Both without bias.
	int _prevLayerSize;
	int _layerSize;

	Dense1dBack(float *ig, float *dp, float *pa, float *w, float *d, int pls, int ls)
		:_inGrads(ig), _dotProducts(dp), _prevAct(pa), _weights(w), _deltas(d), 
		_prevLayerSize(pls), _layerSize(ls)
	{}
	__host__ __device__ thrust::tuple<float, float> operator()(size_t bwidx)
	{
		// actGrad = gradRelu(inGrad, dotProduct)
		// outGrad = weight*actGrad = weight*gradRelu(inGrad, dotProduct)
		// weightGrad = prevAct*actGrad = prevAct*gradRelu(inGrad, dotProduct) 
		// tdUpdate = weightGrad*delta

		int bidx = bwidx/((_prevLayerSize + 1)*_layerSize);
		int widx = bwidx - bidx*(_prevLayerSize + 1)*_layerSize;
		int aidx = widx/(_prevLayerSize + 1);							// Current layer activation index.
		int pidx = widx - aidx*(_prevLayerSize + 1);					// Previous layer activation index.

		// actGrad is the gradient of relu activation.
		//float actGrad = (_dotProducts[aidx] >= 0)? inGrad : 0;
//		printf("%d\n", aidx);
		int aix = bidx*_layerSize + aidx;
		float actGrad = 0;
		if(_dotProducts[aidx] >= 0)
			actGrad = _inGrads[aix];

		int pix = bidx*_prevLayerSize + pidx;
		float outGrad = _weights[widx]*actGrad;
		float tdUpdate;
		//If pidx is bias then prevAct is one.
		if(pidx == _prevLayerSize)
			tdUpdate = actGrad*_deltas[bidx];
		else
			tdUpdate = _prevAct[pix]*actGrad*_deltas[bidx];

		return thrust::make_tuple(outGrad, tdUpdate);
	}
};


void DenseLayer::backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec)
{
	auto start = doomDebug.start("DenseLayer::backProp", 3);

	std::vector<float> grad;
	if(_nextLayer->layerType() == FC)
		grad = ((DenseLayer*)_nextLayer)->outGrads();
	else
		grad = ((OutputLayer*)_nextLayer)->outGrads();

	thrust::device_vector<float> nextOutGrads(grad);
	thrust::device_vector<float> inGrads(_activations.size());
		thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(inGrads.size()), inGrads.begin(), 
		Dense1dGetInGrads(thrust::raw_pointer_cast(nextOutGrads.data()), _layerSize[1], _nextLayer->layerSize()[1]));
	cudaDeviceSynchronize();

	std::vector<float> pAct = _prevLayer->activations();
	thrust::device_vector<float> prevAct(pAct.begin(), pAct.end());
	thrust::device_vector<float> weights(_weights.begin(), _weights.end());
	thrust::device_vector<float> dotProducts(_dotProducts.begin(), _dotProducts.end());
	thrust::device_vector<float> deltas(deltaVec.begin(), deltaVec.end());

	thrust::device_vector<float> outGrads(_outGrads.size());
	thrust::device_vector<float> tdUpdates(_TDUpdates.size());

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(_weights.size()), 
		thrust::make_zip_iterator(thrust::make_tuple(outGrads.begin(), tdUpdates.begin())), 
		Dense1dBack(thrust::raw_pointer_cast(inGrads.data()), thrust::raw_pointer_cast(dotProducts.data()), 
			thrust::raw_pointer_cast(prevAct.data()), thrust::raw_pointer_cast(weights.data()), 
			thrust::raw_pointer_cast(deltas.data()), prevAct.size(), _layerSize[1]));
	cudaDeviceSynchronize();

	thrust::copy(outGrads.begin(), outGrads.end(), _outGrads.begin());
	thrust::copy(tdUpdates.begin(), tdUpdates.end(), _TDUpdates.begin());	

	doomDebug.end("DenseLayer::backProp", 3, start);
}

void DenseLayer::cacheWeights()
{
	//std::cout << "DenseLayer::cacheWeights" << std::endl;

	_cachedWeights = _weights;
}

unsigned DenseLayer::numHiddenUnits() const
{
	//std::cout << "DenseLayer::numHiddenUnits" << std::endl;	

	return _numHiddenUnits;
}

std::vector<float> DenseLayer::weights() const
{
	//std::cout << "DenseLayer::weights" << std::endl;

	return _weights;
}

std::vector<float> DenseLayer::dotProducts() const
{
	//std::cout << "DenseLayer::dotProducts" << std::endl;

	return _dotProducts;
}

std::vector<float> DenseLayer::outGrads() const
{
	//std::cout << "DenseLayer::outGrads" << std::endl;

	return _outGrads;
}

Tensor1d<Dense1dVertex*>* DenseLayer::vertices() const
{
	//std::cout << "DenseLayer::vertices" << std::endl;
	return _vertices;
}

void DenseLayer::setWeights(const std::vector<float> &w)
{
	//std::cout << "DenseLayer::setWeights" << std::endl;

	_weights.clear();
	std::copy(w.begin(), w.end(), std::back_inserter(_weights));
}

//-----------------------------------------------------------------------------------------------------------------------------------------//

OutputLayer::OutputLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned hu)
	:NetworkLayer(ln, at, ls, prevLayer), _numHiddenUnits(hu)
{
	auto start = doomDebug.start("OutputLayer::OutputLayer", 2);

	std::vector<unsigned> prevLayerSize = prevLayer->layerSize();
	int prevTotalSize = 1;
	for(unsigned i=1; i<prevLayerSize.size(); ++i)
		prevTotalSize *= prevLayerSize[i];

	std::vector<unsigned> curLayerSize = layerSize();
	_weights = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
	_cachedWeights = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
	_TDUpdates = std::vector<float>(_layerSize[0]*(prevTotalSize + 1)*_numHiddenUnits);
	_outGrads = std::vector<float>(_layerSize[0]*(prevTotalSize + 1)*_numHiddenUnits);
	_dotProducts = std::vector<float>(_layerSize[0]*_numHiddenUnits);
	_activations = std::vector<float>(_layerSize[0]*_numHiddenUnits);
	_vertices = nullptr;
	_bias = nullptr;

	//_vertices = new Tensor1d<Dense1dVertex*>(curLayerSize[1]);																	// Remove later.
	//_bias = new BiasVertex(nullptr, 0);																							// Remove later.
	//BiasVertex *prevBias = prevLayer->biasVertex();																				// Remove later.
	//for(unsigned i=0; i<curLayerSize[1]; ++i)																					// Remove later.
	//{																															// Remove later.
	//	Dense1dVertex *v;																										// Remove later.
	//	Tensor1d<Dense1dVertex*> *prevVertices = ((DenseLayer*)prevLayer)->vertices();											// Remove later.
    //    std::vector<unsigned> prevLayerSize = prevLayer->layerSize();															// Remove later.
	//	Tensor1d<WeightedEdge*> *inputEdges = new Tensor1d<WeightedEdge*>(prevLayerSize[1] + 1);								// Remove later.
	//	v = new Relu1dUnit(&_activations[i], 0, &_TDUpdates[i], inputEdges);													// Remove later.
    //    for(unsigned j=0; j<prevLayerSize[1]; ++j)																				// Remove later.
	//	{																														// Remove later.
	//		int eIndex = i*(prevTotalSize + 1) + j;																				// Remove later.
	//		Vertex *u = (*prevVertices)[j];																						// Remove later.
	//		WeightedEdge *e = new WeightedEdge(u, /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);			// Remove later.
	//		(*inputEdges)[j] = e;																								// Remove later.
	//		u->addOutputEdge(e);																								// Remove later.
	//	}																														// Remove later.
	//																															// Remove later.
	//	int eIndex = i*(prevTotalSize + 1) + prevTotalSize;																		// Remove later.
	//	WeightedEdge *e = new WeightedEdge(prevBias, /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);		// Remove later.
	//	(*inputEdges)[prevTotalSize] = e;																						// Remove later.
	//	prevBias->addOutputEdge(e);																								// Remove later.
	//																															// Remove later.
	//	(*_vertices)[i] = v;																									// Remove later.
	//}																															// Remove later.
	doomDebug.end("OutputLayer::OutputLayer", 2, start);
}

NetworkLayer::LayerType OutputLayer::layerType() const
{
	//std::cout << "DenseLayer::layerType" << std::endl;
	
	return NetworkLayer::OUTPUT;
}

void OutputLayer::forwardProp(PropagationType p)
{
	auto start = doomDebug.start("OutputLayer::forwardProp", 3);

	std::vector<unsigned> prevLayerSize = _prevLayer->layerSize();
	int inputSize, layerSize;
	if(p == SINGLE)
	{
		inputSize = prevLayerSize[1];
		layerSize = _layerSize[1];
	}
	else
	{
		inputSize = prevLayerSize[0]*prevLayerSize[1];
		layerSize = _layerSize[0]*_layerSize[1];
	}

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.begin() + inputSize);
	thrust::device_vector<float> weights(_weights.size());
	if(p == PREDICTION)
		thrust::copy(_weights.begin(), _weights.end(), weights.begin());
	else if(p == TARGET)
		thrust::copy(_cachedWeights.begin(), _cachedWeights.end(), weights.begin());

	thrust::device_vector<float> dotProducts(layerSize);
	thrust::device_vector<float> activations(layerSize);

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(layerSize), 
		thrust::make_zip_iterator(thrust::make_tuple(dotProducts.begin(), activations.begin())), 
		Dense1dTransform(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(weights.data()), 
			prevLayerSize[1], _layerSize[1]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vectors as well. Also, make sure location of output vector is not changed.
	thrust::copy(dotProducts.begin(), dotProducts.end(), _dotProducts.begin());
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
	
	doomDebug.end("OutputLayer::forwardProp", 3, start);
}


struct OutputBack{
	float *_inGrads;
	float *_dotProducts;
	float *_prevAct;
	float *_weights;
	float *_deltas;
	// Both without bias.
	int _prevLayerSize;
	int _layerSize;

	OutputBack(float *ig, float *dp, float *pa, float *w, float *d, int pls, int ls)
		:_inGrads(ig), _dotProducts(dp), _prevAct(pa), _weights(w), _deltas(d), 
		_prevLayerSize(pls), _layerSize(ls)
	{}
	__host__ __device__ thrust::tuple<float, float> operator()(size_t bwidx)
	{
		// actGrad = gradRelu(inGrad, dotProduct)
		// outGrad = weight*actGrad = weight*gradRelu(inGrad, dotProduct)
		// weightGrad = prevAct*actGrad = prevAct*gradRelu(inGrad, dotProduct) 
		// tdUpdate = weightGrad*delta

		int bidx = bwidx/((_prevLayerSize + 1)*_layerSize);
		int widx = bwidx - bidx*(_prevLayerSize + 1)*_layerSize;
		int aidx = widx/(_prevLayerSize + 1);
		int pidx = widx - aidx*(_prevLayerSize + 1);

		int aix = bidx*_layerSize + aidx;
		int pix = bidx*_prevLayerSize + pidx;
		// actGrad is the gradient of relu activation.
		float actGrad = (_dotProducts[aix] >= 0)? _inGrads[aidx] : 0;
		float outGrad = _weights[widx]*actGrad;
		float tdUpdate;
		// If pidx is bias then prevAct is one.
		if(pidx == _prevLayerSize)
			tdUpdate = actGrad*_deltas[bidx];
		else
			tdUpdate = _prevAct[pix]*actGrad*_deltas[bidx];

		return thrust::make_tuple(outGrad, tdUpdate);
	}
};


void OutputLayer::backProp(const std::vector<std::vector<double>> &actions, const std::vector<double> &deltaVec)
{
	auto start = doomDebug.start("OutputLayer::backProp", 3);

	std::vector<float> grad = actionsToGrads(actions);
	thrust::device_vector<float> inGrads(grad.begin(), grad.end());
	thrust::device_vector<float> dotProducts(_dotProducts.begin(), _dotProducts.end());

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> prevAct(act.begin(), act.end());
	thrust::device_vector<float> weights(_weights.begin(), _weights.end());
	thrust::device_vector<float> deltas(deltaVec.begin(), deltaVec.end());

	thrust::device_vector<float> outGrads(_outGrads.size());
	thrust::device_vector<float> tdUpdates(_TDUpdates.size());

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(_weights.size()), 
		thrust::make_zip_iterator(thrust::make_tuple(outGrads.begin(), tdUpdates.begin())), 
		OutputBack(thrust::raw_pointer_cast(inGrads.data()), thrust::raw_pointer_cast(dotProducts.data()), 
			thrust::raw_pointer_cast(prevAct.data()), thrust::raw_pointer_cast(weights.data()),
			thrust::raw_pointer_cast(deltas.data()), prevAct.size(), _layerSize[1]));
	cudaDeviceSynchronize();

	thrust::copy(outGrads.begin(), outGrads.end(), _outGrads.begin());
	thrust::copy(tdUpdates.begin(), tdUpdates.end(), _TDUpdates.begin());	

	doomDebug.end("OutputLayer::backProp", 3, start);
}

std::vector<float> OutputLayer::actionsToGrads(const std::vector<std::vector<double>> &actions) const
{
	auto start = doomDebug.start("OutputLayer::actionsToGrads", 4);

	std::vector<float> grad(_activations.size(), 0);
	for(unsigned i=0; i<actions.size(); ++i)
	{
		int aIdx = 0;
		for(unsigned j=0; j<actions[i].size(); ++j)
			aIdx += actions[i][actions[i].size()-1-j]*(1 << j);
		grad[i*_layerSize[1] + aIdx] = 1.0f;
	}

	doomDebug.end("OutputLayer::actionsToGrads", 4, start);
	return grad;
}

void OutputLayer::cacheWeights()
{
	//std::cout << "OutputLayer::cacheWeights" << std::endl;

	_cachedWeights = _weights;
}

unsigned OutputLayer::numHiddenUnits() const
{
	//std::cout << "OutputLayer::numHiddenUnits" << std::endl;	

	return _numHiddenUnits;
}

std::vector<float> OutputLayer::weights() const
{
	//std::cout << "OutputLayer::weights" << std::endl;

	return _weights;
}

std::vector<float> OutputLayer::dotProducts() const
{
	//std::cout << "OutputLayer::dotProducts" << std::endl;

	return _dotProducts;
}

std::vector<float> OutputLayer::outGrads() const
{
	//std::cout << "OutputLayer::outGrads" << std::endl;

	return _outGrads;
}

Tensor1d<Dense1dVertex*>* OutputLayer::vertices() const
{
	//std::cout << "OutputLayer::vertices" << std::endl;
	return _vertices;
}

void OutputLayer::setWeights(const std::vector<float> &w)
{
	//std::cout << "OutputLayer::setWeights" << std::endl;

	_weights.clear();
	std::copy(w.begin(), w.end(), std::back_inserter(_weights));
}

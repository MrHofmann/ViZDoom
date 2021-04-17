#include "networklayer.h"
#include <limits>
#include <thrust/device_vector.h>

NetworkLayer::NetworkLayer(std::string n, ActivationType a, const std::vector<unsigned> &s, NetworkLayer *pl)
	:_layerName(n), _activationType(a), _layerSize(s), _prevLayer(pl)
{
	std::cout << "NetworkLayer::NetworkLayer" << std::endl;
	
	int lts = 0;
	for(unsigned i=0; i<s.size(); ++i)
		lts += s[i];
	_activations = std::vector<float>(lts, 333);
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

//----------------------------------------------------------------------------------------------------------------------------------------//

InputLayer::InputLayer(std::string ln, ActivationType at, const std::vector<unsigned> &ls, NetworkLayer *pl)
	:NetworkLayer(ln, at, ls, pl)//, _activations(std::vector<float>(ls[0]*ls[1]*ls[2] + 1))
{
	std::cout << "InputLayer::InputLayer" << std::endl;

	int layerTotalSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	_activations = std::vector<float>(layerTotalSize + 1);
	_vertices = new Tensor3d<Input3dVertex*>(_layerSize[0], _layerSize[1], _layerSize[2]);
	_bias = new BiasVertex(&_activations[layerTotalSize], 0);
	for(unsigned i=0; i<_layerSize[0]; ++i)
		for(unsigned j=0; j<_layerSize[1]; ++j)
			for(unsigned k=0; k<_layerSize[2]; ++k)
			{
				int actIndex = i*_layerSize[1]*_layerSize[2] + j*_layerSize[2] + k;
				(*_vertices)[i][j][k] = new Input3dVertex(&_activations[actIndex], 0);
			}
}

NetworkLayer::LayerType InputLayer::layerType() const
{
	//std::cout << "InputLayer::layerType" << std::endl;
	
	return NetworkLayer::INPUT;
}

void InputLayer::forwardProp(PropagationType p)
{
	//std::cout << "InputLayer::forwardProp" << std::endl;

	unsigned height = _layerSize[0];
	unsigned width = _layerSize[1];
	unsigned depth = _layerSize[2];

	for(unsigned i=0; i<height; ++i)
		for(unsigned j=0; j<width; ++j)
			for(unsigned k=0; k<depth; ++k)
				(*_vertices)[i][j][k]->setActivation((*_state)[i*width*depth + j*depth + k]);

	//std::cout << "InputLayer::forwardProp" << std::endl;
}

void InputLayer::backProp(unsigned expNum, const std::vector<double> &action, double delta)
{
	//std::cout << "InputLayer::backProp" << std::endl;	
}

/*
void InputLayer::forwardPropTarget()
{
	std::cout << "InputLayer::forwardPropTarget" << std::endl;

	unsigned length = _layerSize[0];
	unsigned width = _layerSize[1];
	unsigned depth = _layerSize[2];
	for(unsigned i=0; i<length; ++i)
		for(unsigned j=0; j<width; ++j)
			for(unsigned k=0; k<depth; ++k)
				(*_vertices)[i][j][k]->setActivation((*_state)[i*width*depth + j*depth + k]);
	
}
*/

Tensor3d<Input3dVertex*>* InputLayer::vertices() const
{
	//std::cout << "InputLayer::vertices" << std::endl;

	return _vertices;
}


void InputLayer::setState(vizdoom::BufferPtr s)
{
	//std::cout << "InputLayer::setState" << std::endl;

	_state = s;
}



Conv3dLayer::Conv3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned fdi, unsigned fde, unsigned fs)
	:NetworkLayer(ln, at, ls, prevLayer), _filterDim(fdi), _filterDepth(fde), _filterStride(fs)
{
	std::cout << "Conv3dLayer::Conv3dLayer" << std::endl;

	// Cannot cast Tensor3d<Conv3dVertex*>* / Tensor3d<Pool3dVertex*>* to Tensor3d<Vertex*>* since overload of pointer assignment operator is not allowed. The result
	// of this is that *prevVertices is not pointing to original Tensor3d<Conv3dVertex*>/Tensor3d<Pool3dVertex*> struct but to a new struct whcih holds pointers
	// identical to the ones in the original struct. This new struct has to be deleted upon exit of this constructor to avoid memory leak, but carefuly not to
	// delete the vertices to which pointers inside this struct point to.
    Tensor3d<Vertex*> *prevVertices = nullptr;
	if(prevLayer->layerType() == NetworkLayer::INPUT)
		prevVertices = new Tensor3d<Vertex*>(((InputLayer*)prevLayer)->vertices());
	else if(prevLayer->layerType() == NetworkLayer::CONV)
		//prevVertices = ((Conv3dLayer*)prevLayer)->vertices();
		prevVertices = new Tensor3d<Vertex*>(((Conv3dLayer*)prevLayer)->vertices());
	else if(prevLayer->layerType() == NetworkLayer::MAX_POOL)
		//prevVertices = ((Pool3dLayer*)prevLayer)->vertices();
		prevVertices = new Tensor3d<Vertex*>(((Pool3dLayer*)prevLayer)->vertices());
	else
	{
		std::cout << "Unexpected previous layer type. Previous layer should be INPUT, CONV or POOL." << std::endl;
		std::exit(1);
	}

	int layerTotalSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	int filterTotalSize = _filterDim*_filterDim*_filterDepth;
	// Total number of weights is 3d-filter size times number of filters plus one bias for each filter. That is for each filter (depth dimension) there are
	// (filterTotalSize + 1) weights. (filterTotalSize + 1)*_layerSize[2]
	_weights = std::vector<float>(filterTotalSize*_layerSize[2] + _layerSize[2]);
	_cachedWeights = std::vector<float>(filterTotalSize*_layerSize[2] + _layerSize[2]);
	_TDUpdates = std::vector<float>(filterTotalSize*_layerSize[2] + _layerSize[2]);
	_outGrads = std::vector<float>((filterTotalSize + 1)*layerTotalSize);
	_dotProducts = std::vector<float>(layerTotalSize);
	_activations = std::vector<float>(layerTotalSize + 1);

	_vertices = new Tensor3d<Conv3dVertex*>(_layerSize[0], _layerSize[1], _layerSize[2]);
	_bias = new BiasVertex(&_activations[layerTotalSize], 0);
	for(unsigned i=0; i<_layerSize[0]; ++i)
		for(unsigned j=0; j<_layerSize[1]; ++j)
			for(unsigned k=0; k<_layerSize[2]; ++k)
			{
				int vIndex = i*_layerSize[1]*_layerSize[2] + j*_layerSize[2] + k;
				//Tensor3d<WeightedEdge*> *inputEdges = new Tensor3d<WeightedEdge*>(_filterDim, _filterDim, _filterDepth);
				Tensor1d<WeightedEdge*> *inputEdges = new Tensor1d<WeightedEdge*>(filterTotalSize + 1);
				Conv3dVertex *v = new Relu3dUnit(&_activations[vIndex], 0.0f, &_dotProducts[vIndex], 0.0f, inputEdges);
				for(unsigned h=0; h<_filterDim; ++h)
					for(unsigned w=0; w<_filterDim; ++w)
						for(unsigned d=0; d<_filterDepth; ++d)
						{
							int eIndex = k*(filterTotalSize + 1) + h*_filterDim*_filterDepth + w*_filterDepth + d;
							int gIndex = vIndex*(filterTotalSize + 1) + h*_filterDim*_filterDepth + w*_filterDepth + d;
							WeightedEdge *e = new WeightedEdge((*prevVertices)[i+h][j+w][d], /*v,*/ &_outGrads[gIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
							//(*inputEdges)[h][w][d] = e;
							(*inputEdges)[h*_filterDim*_filterDepth + w*_filterDepth + d] = e;
							(*prevVertices)[i+h][j+w][d]->addOutputEdge(e);
						}

				int eIndex = k*(filterTotalSize + 1) + filterTotalSize;
				int gIndex = vIndex*(filterTotalSize + 1) + filterTotalSize;
				WeightedEdge *e = new WeightedEdge(biasVertex(), /*v,*/ &_outGrads[gIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
				(*inputEdges)[filterTotalSize] = e;
				biasVertex()->addOutputEdge(e);
				(*_vertices)[i][j][k] = v;
			}
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
		// vidx = i*outputWidth*outputDepth + j*outputDepth + k
		int i = vidx/(_layerWidth*_layerDepth);
		int jk = vidx - (i*_layerWidth*_layerDepth);
		int j = jk/_layerDepth;
		int k = jk - (j*_layerDepth);

		int widx = k*(_filterDim*_filterDim*_filterDepth + 1);
		// Input vertex index should be the same as the output one? No, this was a bug. The input vertex index is always in the
		// first input channel. Thats why there is no + k at the end of right hand side of the expression bellow.
		int iidx = i*_inputWidth*_inputDepth + j*_inputDepth;
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
		activation = (dotProduct > 0)? dotProduct : 0;
		return thrust::make_tuple(dotProduct, activation);
	}
};

void Conv3dLayer::forwardProp(PropagationType p)
{
	//std::cout << "Conv3dLayer::forwardProp" << std::endl;

	int layerSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	int inputHeight = _prevLayer->layerSize()[0];
	int inputWidth = _prevLayer->layerSize()[1];
	int inputDepth = _prevLayer->layerSize()[2];

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.end());
	thrust::device_vector<float> weights(_weights.size());
	if(p == PREDICTION)
		thrust::copy(_weights.begin(), _weights.end(), weights.begin());
	else if(p == TARGET)
		thrust::copy(_cachedWeights.begin(), _cachedWeights.end(), weights.begin());
	
	thrust::device_vector<float> dotProducts(layerSize);
	thrust::device_vector<float> activations(layerSize);

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(layerSize), 
		thrust::make_zip_iterator(thrust::make_tuple(dotProducts.begin(), activations.begin())), 
		Conv3dTransform(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(weights.data()), 
			_filterDim, _filterDepth, inputHeight, inputWidth, inputDepth, _layerSize[0], _layerSize[1], _layerSize[2]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vectors as well. Also, make sure location of output vector is not changed.
	thrust::copy(dotProducts.begin(), dotProducts.end(), _dotProducts.begin());
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
	
	//std::cout << "Conv3dLayer::forwardProp" << std::endl;
}

struct Conv3dBack{
	float *_inGrad;
	float *_dotProducts;
	float *_prevAct;
	float *_weights;
	float *_outGrads;
	float _delta;
	unsigned _filterDim;
	unsigned _filterDepth;
	unsigned _layerHeight;
	unsigned _layerWidth;
	unsigned _layerDepth;
	unsigned _prevLayerWidth;
	unsigned _prevLayerDepth;

	Conv3dBack(float *ig, float *dp, float *pa, float *w, float *og, float d, 
			unsigned fdi, unsigned fde, unsigned lh, unsigned lw, unsigned ld, unsigned plw, unsigned pld)
		:_inGrad(ig), _dotProducts(dp), _prevAct(pa), _weights(w), _outGrads(og), _delta(d),
		_filterDim(fdi), _filterDepth(fde), _layerHeight(lh), _layerWidth(lw), _layerDepth(ld), _prevLayerWidth(plw), _prevLayerDepth(pld)
	{}
	__host__ __device__ float operator()(size_t widx)
	{
		// actGrad = gradRelu(inGrad, dotProduct)
		// outGrad = weight*actGrad = weight*gradRelu(inGrad, dotProduct)
		// weightGrad = prevAct*actGrad = prevAct*gradRelu(inGrad, dotProduct) 
		// tdUpdate = weightGrad*delta

		int weightsTotalSize = _filterDim*_filterDim*_filterDepth + 1;
		// The channel in which weight widx is located.
		int k = widx/weightsTotalSize;
		// Location of widx and its edge in input edges of each vertex.
		int r = widx - k*weightsTotalSize;

		int h = r/_filterDim*_filterDepth;
		int wd = r - h*_filterDim*_filterDepth;
		int w = wd/_filterDepth;
		int d = wd - w*_filterDepth;

		// Fixating the k-th channel because weight widx is common only for that channel.
		float tdUpdate = 0;
		for(unsigned i=0; i<_layerHeight; ++i)
			for(unsigned j=0; j<_layerWidth; ++j)
			{
				int vidx = i*_layerWidth*_layerDepth + j*_layerDepth + k;
				float actGrad = (_dotProducts[vidx] >= 0)? _inGrad[vidx] : 0;

				int gidx = vidx*weightsTotalSize + r;
				_outGrads[gidx] = _weights[widx]*actGrad;

				// If current weight widx is bias then previous activation is one.
				if(r == weightsTotalSize - 1)
					tdUpdate += actGrad*_delta;
				else
				{
					// This is the vertex from previous layer that is connected via edge of widx. First is calculated the top left vertex 
					// in previous layer (vidx + skipped vertices because of filter size) and then adding the precise location of wanted
					// vertex in convolution block with respect to top left corner. It is easier to understand on a drawing.
					int aidx = vidx + i*(_filterDim - 1)*_prevLayerDepth + h*_prevLayerWidth*_prevLayerDepth + w*_prevLayerDepth + d;
					tdUpdate += _prevAct[aidx]*actGrad*_delta;					
				}
			}

		return tdUpdate;
	}
};

void Conv3dLayer::backProp(unsigned expNum, const std::vector<double> &action, double delta)
{
	//std::cout << "Conv3dLayer::backProp" << std::endl;

	int layerTotalSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	std::vector<float> grad(layerTotalSize);
	for(unsigned i=0; i<_layerSize[0]; ++i)
		for(unsigned j=0; j<_layerSize[1]; ++j)
			for(unsigned k=0; k<_layerSize[2]; ++k)
			{
				int gidx = i*_layerSize[1]*_layerSize[2] + j*_layerSize[2] + k;
				std::vector<Edge*> outputEdges = (*_vertices)[i][j][k]->outputEdges();
				float inGrad = 0;
				for(int l=0; l<outputEdges.size(); ++l)
					inGrad += outputEdges[l]->outGrad();
		
				grad[gidx] = inGrad;
			}

	thrust::device_vector<float> inGrads(grad.begin(), grad.end());
	thrust::device_vector<float> dotProducts(_dotProducts.begin(), _dotProducts.end());
	//thrust::device_vector<float> actGrad(_activations.size());

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> prevAct(act.begin(), act.end());
	thrust::device_vector<float> weights(_weights.begin(), _weights.end());

	thrust::device_vector<float> outGrads(_outGrads.size());
	thrust::device_vector<float> tdUpdates(_TDUpdates.size());

	std::vector<unsigned> prevLayerSize = _prevLayer->layerSize();
	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(_weights.size()), tdUpdates.begin(), 
		Conv3dBack(thrust::raw_pointer_cast(inGrads.data()), thrust::raw_pointer_cast(dotProducts.data()), 
			thrust::raw_pointer_cast(prevAct.data()), thrust::raw_pointer_cast(weights.data()), thrust::raw_pointer_cast(outGrads.data()), delta,
			_filterDim, _filterDepth, _layerSize[0], _layerSize[1], _layerSize[2], prevLayerSize[1], prevLayerSize[2]));
	cudaDeviceSynchronize();

	thrust::copy(outGrads.begin(), outGrads.end(), _outGrads.begin());
	thrust::copy(tdUpdates.begin(), tdUpdates.end(), _TDUpdates.begin());		
}

void Conv3dLayer::cacheWeights()
{
	//std::cout << "Conv3dLayer::cacheWeights" << std::endl;

	_cachedWeights = _weights;
}

/*thrust::device_ptr<float> Conv3dLayer::weightsToDevice() const
{
	//#include <thrust/device_ptr.h>
	//#include <thrust/fill.h>
	//#include <cuda.h>

	size_t N = 10;
	int layerSize = _layerSize[0]*_layerSize[1]*_layerSize[2];

	// allocate pointer to device memory
	float *rawPtr;
	cudaMalloc((void **) &rawPtr, N*sizeof(float));

	// wrap raw pointer with a device_ptr
	thrust::device_ptr<float> devPtr(rawPtr);

	// copy memory to a new device_vector (which automatically allocates memory)
	//thrust::device_vector<float> vec(devPtr, devPtr + N);

	// use vec
	
	// free user-allocated memory
	//cudaFree(rawPtr);
	
	return devPtr;
}*/

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
	std::cout << "Conv3dLayer::weights" << std::endl;

	return _weights;
}

std::vector<float> Conv3dLayer::dotProducts() const
{
	std::cout << "Conv3dLayer::dotProducts" << std::endl;

	return _dotProducts;
}

Tensor3d<Conv3dVertex*> *Conv3dLayer::vertices() const
{
	//std::cout << "Conv3dLayer::vertices" << std::endl;
	return _vertices;
}

void Conv3dLayer::setWeights(const std::vector<float> &w)
{
	std::cout << "Conv3dLayer::setWeights" << std::endl;

	_weights.clear();
	std::copy(w.begin(), w.end(), std::back_inserter(_weights));
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------//
Pool3dLayer::Pool3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned pdi, unsigned ps)
    :NetworkLayer(ln, at, ls, prevLayer), _poolDim(pdi), _poolStride(ps)
{
	std::cout << "Pool3dLayer::Pool3dLayer" << std::endl;

	if(prevLayer->layerType() != NetworkLayer::CONV)
	{
		std::cout << "Unexpected previous layer type. Previous layer must be CONV." << std::endl;
		std::exit(1);
	}

	Tensor3d<Conv3dVertex*> *prevVertices = ((Conv3dLayer*)prevLayer)->vertices();
	int layerTotalSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	_outGrads = std::vector<float>(_poolDim*_poolDim*layerTotalSize);
	_activations = std::vector<float>(layerTotalSize + 1);
	_vertices = new Tensor3d<Pool3dVertex*>(_layerSize[0], _layerSize[1], _layerSize[2]);
	_bias = new BiasVertex(&_activations[layerTotalSize], 0);
	for(unsigned i=0; i<_layerSize[0]; ++i)
		for(unsigned j=0; j<_layerSize[1]; ++j)
			for(unsigned k=0; k<_layerSize[2]; ++k)
			{
				int vIndex = i*_layerSize[1]*_layerSize[2] + j*_layerSize[2] + k;
				Tensor2d<UnweightedEdge*> *inputEdges = new Tensor2d<UnweightedEdge*>(_poolDim, _poolDim);
				MaxPool3dUnit *v = new MaxPool3dUnit(&_activations[vIndex], 0, inputEdges);
				for(unsigned h=0; h<_poolDim; ++h)
					for(unsigned w=0; w<_poolDim; ++w)
					{
						int gIndex = vIndex*_poolDim*_poolDim + h*_poolDim + w;
						UnweightedEdge *e = new UnweightedEdge((*prevVertices)[i+h][j+w][k]/*, v*/, &_outGrads[gIndex]);
						(*inputEdges)[h][w] = e;
						(*prevVertices)[i+h][j+w][k]->addOutputEdge(e);
					}				

				(*_vertices)[i][j][k] = v;
			}
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
		int i = vidx/(_layerWidth*_layerDepth);
		int jk = vidx - (i*_layerWidth*_layerDepth);
		int j = jk/_layerDepth;
		int k = jk - (j*_layerDepth);

		// Input vertex index should be the same as the output one? Yes, because unlike convolution, 
		// pooling is applied to each channel independently.
		int iidx = i*_inputWidth*_inputDepth + j*_inputDepth + k;
		float activation = _input[iidx];
		for(unsigned h=0; h<_poolDim; ++h)
			for(unsigned w=0; w<_poolDim; ++w)
			{	
				int ix = iidx + h*_inputWidth*_inputDepth + w*_inputDepth;
				if(_input[ix] > activation)
					activation = _input[ix];
			}	
			
		return activation;
	}

};

void Pool3dLayer::forwardProp(PropagationType p)
{
	//std::cout << "Pool3dLayer::forwardProp" << std::endl;

	int layerSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	int inputHeight = _prevLayer->layerSize()[0];
	int inputWidth = _prevLayer->layerSize()[1];
	int inputDepth = _prevLayer->layerSize()[2];

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.end());
	thrust::device_vector<float> activations(layerSize);

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(layerSize), 
		activations.begin(), Pool3dTransform(thrust::raw_pointer_cast(input.data()), _poolDim, //_poolDepth, 
			inputHeight, inputWidth, inputDepth, _layerSize[0], _layerSize[1], _layerSize[2]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vector as well. Also, make sure location of output vector is not changed.
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
	//std::cout << "Pool3dLayer::forwardProp" << std::endl;
}

struct Pool3dBack{
	float *_inGrad;
	float *_prevAct;
	unsigned _poolDim;
	unsigned _layerHeight;
	unsigned _layerWidth;
	unsigned _layerDepth;

	Pool3dBack(float *ig, float *pa, unsigned pd, unsigned lh, unsigned lw, unsigned ld)
		:_inGrad(ig), _prevAct(pa), _poolDim(pd), _layerHeight(lh), _layerWidth(lw), _layerDepth(ld)
	{}
	__host__ __device__ float operator()(size_t eidx)
	{
		// outGrad = gradPool(inGrad, prevAct)

		int vidx = eidx/(_poolDim*_poolDim);
		int i = vidx/(_layerWidth*_layerDepth);
		//int jk = vidx - i*_layerWidth*_layerDepth;
		//int j = jk/_layerDepth;
		//int k = jk - j*_layerDepth;

		int hwp = eidx - vidx*_poolDim*_poolDim;
		int hp = hwp/_poolDim;
		int wp = hwp - hp*_poolDim;
		
		int aidx = vidx + i*(_poolDim - 1)*_layerDepth;
		unsigned eix = aidx + hp*_poolDim*_layerDepth + wp*_layerDepth;
		float outGrad = _inGrad[vidx];
		for(unsigned h=0; h<_poolDim; ++h)
			for(unsigned w=0; w<_poolDim; ++w)
			{
				unsigned aix = aidx + h*_poolDim*_layerDepth + w*_layerDepth;
				if(_prevAct[aix] > _prevAct[eix])
					return 0;
			}

		return outGrad;
	}
};


void Pool3dLayer::backProp(unsigned expNum, const std::vector<double> &action, double delta)
{
	//std::cout << "Pool3dLayer::backProp" << std::endl;

	int layerTotalSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	std::vector<float> grad(layerTotalSize);
	for(unsigned i=0; i<_layerSize[0]; ++i)
		for(unsigned j=0; j<_layerSize[1]; ++j)
			for(unsigned k=0; k<_layerSize[2]; ++k)
			{
				int gidx = i*_layerSize[1]*_layerSize[2] + j*_layerSize[2] + k;
				std::vector<Edge*> outputEdges = (*_vertices)[i][j][k]->outputEdges();
				float inGrad = 0;
				for(int l=0; l<outputEdges.size(); ++l)
					inGrad += outputEdges[l]->outGrad();
		
				grad[gidx] = inGrad;
			}

	thrust::device_vector<float> inGrads(grad.begin(), grad.end());
	//thrust::device_vector<float> dotProducts(_dotProducts.begin(), _dotProducts.end());
	//thrust::device_vector<float> actGrad(_activations.size());

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> prevAct(act.begin(), act.end());
	//thrust::device_vector<float> weights(_weights.begin(), _weights.end());

	thrust::device_vector<float> outGrads(_outGrads.size());
	//thrust::device_vector<float> tdUpdates(_TDUpdates.size());

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(outGrads.size()), outGrads.begin(), 
		Pool3dBack(thrust::raw_pointer_cast(inGrads.data()), thrust::raw_pointer_cast(prevAct.data()), _poolDim, _layerSize[0], _layerSize[1], _layerSize[2]));
	cudaDeviceSynchronize();

	thrust::copy(outGrads.begin(), outGrads.end(), _outGrads.begin());		
}

unsigned Pool3dLayer::poolDim() const
{
	//std::cout << "Pool3dLayer::poolDim" << std::endl;	

	return _poolDim;
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
	std::cout << "DenseLayer::DenseLayer" << std::endl;

	std::vector<unsigned> prevLayerSize = prevLayer->layerSize();
	int prevTotalSize = 1;
	for(unsigned i=0; i<prevLayerSize.size(); ++i)
		prevTotalSize *= prevLayerSize[i];

	std::vector<unsigned> curLayerSize = layerSize();
	_weights = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
	_cachedWeights = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
	_TDUpdates = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
	_outGrads = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
	_dotProducts = std::vector<float>(_numHiddenUnits);
	_activations = std::vector<float>(_numHiddenUnits + 1);

	_vertices = new Tensor1d<Dense1dVertex*>(curLayerSize[0]);
	_bias = new BiasVertex(&_activations[_numHiddenUnits], 0);
	
	BiasVertex *prevBias = prevLayer->biasVertex();
	for(unsigned i=0; i<curLayerSize[0]; ++i)
	{
		Dense1dVertex *v;
		if(prevLayer->layerType() == NetworkLayer::FC)
		{
			Tensor1d<Dense1dVertex*> *prevVertices = ((DenseLayer*)prevLayer)->vertices();
            std::vector<unsigned> prevLayerSize = prevLayer->layerSize();
			Tensor1d<WeightedEdge*> *inputEdges = new Tensor1d<WeightedEdge*>(prevLayerSize[0] + 1);
			v = new Relu1dUnit(&_activations[i], 0, &_TDUpdates[i], inputEdges);
            for(unsigned j=0; j<prevLayerSize[0]; ++j)
			{
				int eIndex = i*(prevTotalSize + 1) + j;
				Vertex *u = (*prevVertices)[j];
				WeightedEdge *e = new WeightedEdge(u, /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
				(*inputEdges)[j] = e;
				u->addOutputEdge(e);
			}
		
			int eIndex = i*(prevTotalSize + 1) + prevTotalSize;
			WeightedEdge *e = new WeightedEdge(prevBias, /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
			(*inputEdges)[prevTotalSize] = e;
			prevBias->addOutputEdge(e);

		}
	    else
		{
			// Same issue as in Conv3dLayer::Conv3dLayer. Will have to decide at some point whether should prevVertices be used as structs or pointers to structs. In
			// case pointers are NOT used then consider replacing pointers to vertices structs with only structs in Conv3dLayer and DenseLayer.
	        Tensor3d<Vertex*> *prevVertices = nullptr;
	        if(prevLayer->layerType() == NetworkLayer::CONV)
	            //prevVertices = ((Conv3dLayer*)prevLayer)->vertices();
				prevVertices = new Tensor3d<Vertex*>(((Conv3dLayer*)prevLayer)->vertices());
	        else if(prevLayer->layerType() == NetworkLayer::MAX_POOL)
	        	//prevVertices = ((Pool3dLayer*)prevLayer)->vertices();
				prevVertices = new Tensor3d<Vertex*>(((Pool3dLayer*)prevLayer)->vertices());
        	else
		    {   
				std::cout << "Unexpected previous layer type. Previous layer should be CONV, POOL or FC." << std::endl;
				std::exit(1);
			}

			std::vector<unsigned> prevLayerSize = prevLayer->layerSize();
			Tensor1d<WeightedEdge*> *inputEdges = new Tensor1d<WeightedEdge*>(prevTotalSize + 1);
			v = new Relu1dUnit(&_activations[i], 0, &_dotProducts[i], inputEdges);
			for(unsigned h=0; h<prevLayerSize[0]; ++h)
				for(unsigned w=0; w<prevLayerSize[1]; ++w)
            		for(unsigned d=0; d<prevLayerSize[2]; ++d)
                	{
						int eIndex = i*(prevTotalSize + 1) + h*prevLayerSize[1]*prevLayerSize[2] + w*prevLayerSize[2] + d;
	           			WeightedEdge *e = new WeightedEdge((*prevVertices)[h][w][d], /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
						(*inputEdges)[h*prevLayerSize[1]*prevLayerSize[2] + w*prevLayerSize[2] + d] = e;
						(*prevVertices)[h][w][d]->addOutputEdge(e);
					}

			int eIndex = i*(prevTotalSize + 1) + prevTotalSize;
	        WeightedEdge *e = new WeightedEdge(prevBias, /*v,*/ &_outGrads[eIndex], &_weights[eIndex], &_TDUpdates[eIndex]);
			(*inputEdges)[prevTotalSize] = e;
			prevBias->addOutputEdge(e);
		}

		(*_vertices)[i] = v;
	}
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
	int _layerWidth;
	int _layerDepth;

	Dense1dTransform(float *i, float *w, int ih, int lh, int lw, int ld)
		:_input(i), _weights(w),
		_inputHeight(ih), 
		_layerHeight(lh), _layerWidth(lw), _layerDepth(ld){}
	__host__ __device__ thrust::tuple<float, float> operator()(size_t vidx)
	{
		// inputSize = inputHeight*inputWidth*inputDepth + bias
		int inputTotalSize = _inputHeight + 1;
		int widx = vidx*inputTotalSize;
		float dotProduct = 0.0f;
		float activation;
		for(unsigned h=0; h<inputTotalSize; ++h)
		{
			int wx = widx + h;
			dotProduct += _input[h]*_weights[wx];
		}
		
		activation = (dotProduct > 0)? dotProduct : 0;
		return thrust::make_tuple(dotProduct, activation);
	}
};

void DenseLayer::forwardProp(PropagationType p)
{
	//std::cout << "DenseLayer::forwardProp" << std::endl;

	int layerSize = _layerSize[0];
	int prevLayerSize;
	if(_prevLayer->layerType() == NetworkLayer::FC)
		prevLayerSize = _prevLayer->layerSize()[0];
	else
		prevLayerSize = _prevLayer->layerSize()[0]*_prevLayer->layerSize()[1]*_prevLayer->layerSize()[2];

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.end());
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
			prevLayerSize, _layerSize[0], _layerSize[1], _layerSize[2]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vectors as well. Also, make sure location of output vector is not changed.
	thrust::copy(dotProducts.begin(), dotProducts.end(), _dotProducts.begin());
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
	
	//std::cout << "DenseLayer::forwardProp" << std::endl;
}

struct Dense1dBack{
	float *_inGrad;
	float *_dotProducts;
	float *_prevAct;
	float *_weights;
	float _delta;
	int _prevLayerSize;

	Dense1dBack(float *ig, float *dp, float *pa, float *w, float d, int pls)
		:_inGrad(ig), _dotProducts(dp), _prevAct(pa), _weights(w), _delta(d), _prevLayerSize(pls)
	{}
	__host__ __device__ thrust::tuple<float, float> operator()(size_t widx)
	{
		// actGrad = gradRelu(inGrad, dotProduct)
		// outGrad = weight*actGrad = weight*gradRelu(inGrad, dotProduct)
		// weightGrad = prevAct*actGrad = prevAct*gradRelu(inGrad, dotProduct) 
		// tdUpdate = weightGrad*delta

		int aidx = widx/_prevLayerSize;

		// actGrad is the gradient of relu activation.
		float actGrad = (_dotProducts[aidx] >= 0)? _inGrad[aidx] : 0;
		float outGrad = _weights[widx]*actGrad;
		float tdUpdate = _prevAct[widx]*actGrad*_delta;

		return thrust::make_tuple(outGrad, tdUpdate);
	}
};


void DenseLayer::backProp(unsigned expNum, const std::vector<double> &action, double delta)
{
	//std::cout << "DenseLayer::backProp" << std::endl;

	std::vector<float> grad((*_vertices).size());
	for(int i=0; i<grad.size(); ++i)
	{
		std::vector<Edge*> outputEdges = (*_vertices)[i]->outputEdges();
		float inGrad = 0;
		for(int j=0; j<outputEdges.size(); ++j)
			inGrad += outputEdges[j]->outGrad();
		
		// In case layer is not output.
		if(outputEdges.size() > 0)
			grad[i] = inGrad;
	}

	thrust::device_vector<float> inGrads(grad.begin(), grad.end());
	thrust::device_vector<float> dotProducts(_dotProducts.begin(), _dotProducts.end());
	//thrust::device_vector<float> actGrad(_activations.size());

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> prevAct(act.begin(), act.end());
	thrust::device_vector<float> weights(_weights.begin(), _weights.end());

	thrust::device_vector<float> outGrads(_outGrads.size());
	thrust::device_vector<float> tdUpdates(_TDUpdates.size());

	// This works here because outGrads and tdUpdates are of same size.
	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(_weights.size()), 
		thrust::make_zip_iterator(thrust::make_tuple(outGrads.begin(), tdUpdates.begin())), 
		Dense1dBack(thrust::raw_pointer_cast(inGrads.data()), thrust::raw_pointer_cast(dotProducts.data()), 
			thrust::raw_pointer_cast(prevAct.data()), thrust::raw_pointer_cast(weights.data()), delta, prevAct.size()));
	cudaDeviceSynchronize();

	thrust::copy(outGrads.begin(), outGrads.end(), _outGrads.begin());
	thrust::copy(tdUpdates.begin(), tdUpdates.end(), _TDUpdates.begin());	

	// prevInGrad = sum over respective outGrad elements. I think this should be done at beginning of backProp for current layer.
	//thrust::device_vector<float> prevInGrad(prevAct.size());
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
	std::cout << "DenseLayer::weights" << std::endl;

	return _weights;
}

std::vector<float> DenseLayer::dotProducts() const
{
	std::cout << "DenseLayer::dotProducts" << std::endl;

	return _dotProducts;
}

Tensor1d<Dense1dVertex*>* DenseLayer::vertices() const
{
	//std::cout << "DenseLayer::vertices" << std::endl;
	return _vertices;
}

void DenseLayer::setWeights(const std::vector<float> &w)
{
	std::cout << "DenseLayer::setWeights" << std::endl;

	_weights.clear();
	std::copy(w.begin(), w.end(), std::back_inserter(_weights));
}


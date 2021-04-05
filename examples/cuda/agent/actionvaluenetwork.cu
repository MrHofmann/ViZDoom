#include "actionvaluenetwork.h"
#include <limits>
#include <thrust/device_vector.h>

NetworkLayer::NetworkLayer(std::string n, ActivationType a, const std::vector<unsigned> &s, NetworkLayer *pl)
	:_layerName(n), _activationType(a), _layerSize(s), _prevLayer(pl)
{
	std::cout << "NetworkLayer::NetworkLayer" << std::endl;
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
	:NetworkLayer(ln, at, ls, pl)
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

void InputLayer::forwardProp()
{
	std::cout << "InputLayer::forwardProp" << std::endl;

	unsigned length = _layerSize[0];
	unsigned width = _layerSize[1];
	unsigned depth = _layerSize[2];
	for(unsigned i=0; i<length; ++i)
		for(unsigned j=0; j<width; ++j)
			for(unsigned k=0; k<depth; ++k)
				(*_vertices)[i][j][k]->setActivation((*_state)[i*width*depth + j*depth + k]);
}

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
	_TDUpdates = std::vector<float>(filterTotalSize*_layerSize[2] + _layerSize[2]);
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
							WeightedEdge *e = new WeightedEdge((*prevVertices)[i+h][j+w][d], v, &_weights[eIndex], &_TDUpdates[eIndex]);
							//(*inputEdges)[h][w][d] = e;
							(*inputEdges)[h*_filterDim*_filterDepth + w*_filterDepth + d] = e;
						}

				int eIndex = k*(filterTotalSize + 1) + filterTotalSize;
				(*inputEdges)[filterTotalSize] = new WeightedEdge(biasVertex(), v, &_weights[eIndex], &_TDUpdates[eIndex]);
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
		_layerHeight(lh), _layerWidth(lw), _layerDepth(ld){}
	__host__ __device__ thrust::tuple<float, float> operator()(size_t vidx)
	{
		// vidx = i*outputWidth*outputDepth + j*outputDepth + k
		int i = vidx/(_layerWidth*_layerDepth);
		int jk = vidx - (i*_layerWidth*_layerDepth);
		int j = jk/_layerDepth;
		int k = jk - (j*_layerDepth);

		//weightsTotalSize = _weightsHeight*_weightsWidth*_weightsDepth + 1;
		//int eIndex = k*(filterTotalSize + 1) + h*_filterDim*_filterDepth + w*_filterDepth + d;
		int widx = k*(_filterDim*_filterDim*_filterDepth + 1);
		// Input vertex index should be the same as the output one.
		int iidx = i*_inputWidth*_inputDepth + j*_inputDepth + k;
		float dotProduct = 0.0f;
		float activation;
		for(unsigned h=0; h<_filterDim; ++h)
			for(unsigned w=0; w<_filterDim; ++w)
				for(unsigned d=0; d<_filterDepth; ++d)
				{
					int wx = widx + h*_filterDim*_filterDepth + w*_filterDepth + d;
					int ix = iidx + h*_inputWidth*_inputDepth + w*_inputDepth + d;
					dotProduct += _input[ix]*_weights[wx];
				}
		
		dotProduct += _weights[widx + _filterDim*_filterDim*_filterDepth];
		activation = (dotProduct > 0)? dotProduct : 0;
		return thrust::make_tuple(dotProduct, activation);
	}
};

void Conv3dLayer::forwardProp()
{
	std::cout << "Conv3dLayer::forwardProp" << std::endl;

	int layerSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	int inputHeight = _prevLayer->layerSize()[0];
	int inputWidth = _prevLayer->layerSize()[1];
	int inputDepth = _prevLayer->layerSize()[2];

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.end());
	thrust::device_vector<float> weights(_weights.begin(), _weights.end());
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

Tensor3d<Conv3dVertex*> *Conv3dLayer::vertices() const
{
	//std::cout << "Conv3dLayer::vertices" << std::endl;
	return _vertices;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------//
Pool3dLayer::Pool3dLayer(std::string ln, ActivationType at, std::vector<unsigned> ls, NetworkLayer *prevLayer, unsigned pdi, unsigned pde, unsigned ps)
    :NetworkLayer(ln, at, ls, prevLayer), _poolDim(pdi), _poolDepth(pde), _poolStride(ps)
{
	std::cout << "Pool3dLayer::Pool3dLayer" << std::endl;

	if(prevLayer->layerType() != NetworkLayer::CONV)
	{
		std::cout << "Unexpected previous layer type. Previous layer must be CONV." << std::endl;
		std::exit(1);
	}

	Tensor3d<Conv3dVertex*> *prevVertices = ((Conv3dLayer*)prevLayer)->vertices();
	int layerTotalSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	_activations = std::vector<float>(layerTotalSize + 1);
	_vertices = new Tensor3d<Pool3dVertex*>(_layerSize[0], _layerSize[1], _layerSize[2]);
	_bias = new BiasVertex(&_activations[layerTotalSize], 0);
	for(unsigned i=0; i<_layerSize[0]; ++i)
		for(unsigned j=0; j<_layerSize[1]; ++j)
			for(unsigned k=0; k<_layerSize[2]; ++k)
			{
				int vIndex = i*_layerSize[1]*_layerSize[2] + j*_layerSize[2] + k;
				Tensor3d<UnweightedEdge*> *inputEdges = new Tensor3d<UnweightedEdge*>(_poolDim, _poolDim, _poolDepth);
				MaxPool3dUnit *v = new MaxPool3dUnit(&_activations[vIndex], 0, inputEdges);
				for(unsigned h=0; h<_poolDim; ++h)
					for(unsigned w=0; w<_poolDim; ++w)
						for(unsigned d=0; d<_poolDepth; ++d)
						{
							UnweightedEdge *e = new UnweightedEdge((*prevVertices)[i+h][j+w][d], v);
							(*inputEdges)[h][w][d] = e;
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
	int _poolDepth;
	int _inputHeight;
	int _inputWidth;
	int _inputDepth;
	int _layerHeight;
	int _layerWidth;
	int _layerDepth;

	Pool3dTransform(float *i, int pdi, int pde, int ih, int iw, int id, int lh, int lw, int ld)
		:_input(i), _poolDim(pdi), _poolDepth(pde), 
		_inputHeight(ih), _inputWidth(iw), _inputDepth(id), 
		_layerHeight(lh), _layerWidth(lw), _layerDepth(ld){}
	__host__ __device__  float operator()(size_t vidx)
	{
		// vidx = i*outputWidth*outputDepth + j*outputDepth + k
		int i = vidx/(_layerWidth*_layerDepth);
		int jk = vidx - (i*_layerWidth*_layerDepth);
		int j = jk/_layerDepth;
		int k = jk - (j*_layerDepth);

		// Input vertex index should be the same as the output one.
		int iidx = i*_inputWidth*_inputDepth + j*_inputDepth + k;
		float activation = _input[0];
		for(unsigned h=0; h<_poolDim; ++h)
			for(unsigned w=0; w<_poolDim; ++w)
				for(unsigned d=0; d<_poolDepth; ++d)
				{
					int ix = iidx + h*_inputWidth*_inputDepth + w*_inputDepth + d;
					if(_input[ix] > activation)
						activation = _input[ix];
				}		

		return activation;
	}

};

void Pool3dLayer::forwardProp()
{
	std::cout << "Pool3dLayer::forwardProp" << std::endl;

	int layerSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	int inputHeight = _prevLayer->layerSize()[0];
	int inputWidth = _prevLayer->layerSize()[1];
	int inputDepth = _prevLayer->layerSize()[2];

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.end());
	thrust::device_vector<float> activations(layerSize);

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(layerSize), 
		activations.begin(), Pool3dTransform(thrust::raw_pointer_cast(input.data()), _poolDim, _poolDepth, 
			inputHeight, inputWidth, inputDepth, _layerSize[0], _layerSize[1], _layerSize[2]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vector as well. Also, make sure location of output vector is not changed.
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
}

unsigned Pool3dLayer::poolDim() const
{
	//std::cout << "Pool3dLayer::poolDim" << std::endl;	

	return _poolDim;
}

unsigned Pool3dLayer::poolDepth() const
{	
	//std::cout << "Pool3dLayer::poolDepth" << std::endl;	

	return _poolDepth;
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
	_TDUpdates = std::vector<float>((prevTotalSize + 1)*_numHiddenUnits);
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
					//	std::cout << eIndex << " = " << _weights.size() << " = " << &_weights[eIndex] << std::endl;
				Vertex *u = (*prevVertices)[j];
				WeightedEdge *e = new WeightedEdge(u, v, &_weights[eIndex], &_TDUpdates[eIndex]);
				(*inputEdges)[j] = e;
			}
		
			int eIndex = i*(prevTotalSize + 1) + prevTotalSize;
					//	std::cout << eIndex << " : " << _weights.size() << " : " << &_weights[eIndex] << std::endl;
			WeightedEdge *e = new WeightedEdge(prevBias, v, &_weights[eIndex], &_TDUpdates[eIndex]);
			(*inputEdges)[prevTotalSize] = e;

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
					//	std::cout << eIndex << " - " << _weights.size() << " - " << &_weights[eIndex] << std::endl;
	           			WeightedEdge *e = new WeightedEdge((*prevVertices)[h][w][d], v, &_weights[eIndex], &_TDUpdates[eIndex]);
						(*inputEdges)[h*prevLayerSize[1]*prevLayerSize[2] + w*prevLayerSize[2] + d] = e;
                	}

			int eIndex = i*(prevTotalSize + 1) + prevTotalSize;
						//std::cout << eIndex << " | " << _weights.size() << " | " << &_weights[eIndex] << std::endl;
	        WeightedEdge *e = new WeightedEdge(prevBias, v, &_weights[eIndex], &_TDUpdates[eIndex]);
			(*inputEdges)[prevTotalSize] = e;
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
	int _inputWidth;
	int _inputDepth;
	int _layerHeight;
	int _layerWidth;
	int _layerDepth;

	Dense1dTransform(float *i, float *w, int ih, int iw, int id, int lh, int lw, int ld)
		:_input(i), _weights(w),
		_inputHeight(ih), _inputWidth(iw), _inputDepth(id), 
		_layerHeight(lh), _layerWidth(lw), _layerDepth(ld){}
	__host__ __device__ thrust::tuple<float, float> operator()(size_t vidx)
	{
		// vidx = i*outputWidth*outputDepth + j*outputDepth + k
		//int i = vidx/(_layerWidth*_layerDepth);
		//int jk = vidx - (i*_layerWidth*_layerDepth);
		//int j = jk/_layerDepth;
		//int k = jk - (j*_layerDepth);

		// inputSize = inputHeight*inputWidth*inputDepth + bias
		int inputTotalSize = _inputHeight*_inputWidth*_inputDepth + 1;
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

void DenseLayer::forwardProp()
{
	std::cout << "DenseLayer::forwardProp" << std::endl;

	int layerSize = _layerSize[0]*_layerSize[1]*_layerSize[2];
	int inputHeight = _prevLayer->layerSize()[0];
	int inputWidth = _prevLayer->layerSize()[1];
	int inputDepth = _prevLayer->layerSize()[2];

	std::vector<float> act = _prevLayer->activations();
	thrust::device_vector<float> input(act.begin(), act.end());
	thrust::device_vector<float> weights(_weights.begin(), _weights.end());
	thrust::device_vector<float> dotProducts(layerSize);
	thrust::device_vector<float> activations(layerSize);

	thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(layerSize), 
		thrust::make_zip_iterator(thrust::make_tuple(dotProducts.begin(), activations.begin())), 
		Dense1dTransform(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(weights.data()), 
			inputHeight, inputWidth, inputDepth, _layerSize[0], _layerSize[1], _layerSize[2]));
	cudaDeviceSynchronize();

	// It looks like it works great with std::vector<float> as output vector of thrust::copy. 
	// Maybe try thrust::host_vector<float> as member vectors as well. Also, make sure location of output vector is not changed.
	thrust::copy(dotProducts.begin(), dotProducts.end(), _dotProducts.begin());
	thrust::copy(activations.begin(), activations.end(), _activations.begin());
}

unsigned DenseLayer::numHiddenUnits() const
{
	//std::cout << "DenseLayer::numHiddenUnits" << std::endl;	

	return _numHiddenUnits;
}

Tensor1d<Dense1dVertex*>* DenseLayer::vertices() const
{
	//std::cout << "DenseLayer::vertices" << std::endl;
	return _vertices;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------//

ActionValueNetwork::ActionValueNetwork(const NetworkConfig &conf)
	:_stateDim(conf.state_dim), _numActions(conf.num_actions)
{
	std::cout << "ActionValueNetwork::ActionValueNetwork" << std::endl;

	NetworkLayer *prevLayer = nullptr;
	std::vector<unsigned> prevLayerSize = {_stateDim[0], _stateDim[1], _stateDim[2]};
	NetworkLayer *layer = new InputLayer("input", RELU, prevLayerSize, prevLayer);
	_layers.push_back(layer);
	_layerSizes["input"] = prevLayerSize;

	prevLayer = layer;
	for(unsigned i=0; i<conf.num_filters.size(); ++i)
	{
		std::string layerName = "conv_" + std::to_string(i);
		unsigned numFilter = conf.num_filters[i];
		unsigned filterDim = conf.filter_dim[i];
		unsigned filterStride = conf.filter_strides[i];
		ActivationType activation = conf.activations[i];

		// Add padding later.
		unsigned layerHeight = (prevLayerSize[0] - conf.filter_dim[i])/conf.filter_strides[i] + 1;
		unsigned layerWidth = (prevLayerSize[1] - conf.filter_dim[i])/conf.filter_strides[i] + 1;
		unsigned layerDepth = conf.num_filters[i];

		layer = new Conv3dLayer(layerName, activation, std::vector<unsigned>({layerHeight, layerWidth, layerDepth}), prevLayer, filterDim, prevLayerSize[2], filterStride);
		_layers.push_back(layer);

		prevLayer = layer;
		prevLayerSize = {layerHeight, layerWidth, layerDepth};
		_layerSizes[layerName] = prevLayerSize;
		
		if(conf.pool_dim[i] > 0)
		{
		   	layerName = "max_pool_" + std::to_string(i);
			unsigned poolDim = conf.pool_dim[i];
			unsigned poolStride = conf.pool_strides[i];

			layerHeight = (prevLayerSize[0] - poolDim)/poolStride + 1;
			layerWidth = (prevLayerSize[1] - poolDim)/poolStride + 1;
			layerDepth = prevLayerSize[2];
			
			layer = new Pool3dLayer(layerName, activation, std::vector<unsigned>({layerHeight, layerWidth, layerDepth}), prevLayer, poolDim, prevLayerSize[2], poolStride);
			_layers.push_back(layer);
		
			prevLayer = layer;
			prevLayerSize = {layerHeight, layerWidth, layerDepth};
			_layerSizes[layerName] = prevLayerSize;
		}
	}

	int prevFcSize = prevLayerSize[0]*prevLayerSize[1]*prevLayerSize[2];
	unsigned i;
	for(i=0; i<conf.num_hidden_units.size(); ++i)
	{
		std::string layerName = "fc_" + std::to_string(i);
		unsigned numHiddenUnits = conf.num_hidden_units[i];

		layer = new DenseLayer(layerName, RELU, std::vector<unsigned>({numHiddenUnits}), prevLayer, numHiddenUnits);
		_layers.push_back(layer);
		_layerSizes[layerName] = {conf.num_hidden_units[i]};
  	
		prevLayer = layer;
		prevFcSize = conf.num_hidden_units[i];
	}
	
	// Output layer	
	std::string layerName = "fc_" + std::to_string(i);
	unsigned outputSize = 1 << conf.num_actions;
	unsigned numHiddenUnits = outputSize;

	layer = new DenseLayer(layerName, RELU, std::vector<unsigned>({numHiddenUnits}), prevLayer, numHiddenUnits);
	_layers.push_back(layer);
	_layerSizes[layerName] = {outputSize};
	
//	this->init_kaiming();

	std::string prevLayerName;
	for(auto it=this->_layers.begin(); it!=_layers.end(); it++)
	{
		std::string layerName = (*it)->layerName();
		std::cerr << layerName << " [ ";
		for(unsigned i=0; i<_layerSizes[layerName].size(); ++i)
			std::cerr << _layerSizes[layerName][i] << " ";
		std::cerr << "] [ ";
		if((*it)->layerType() == NetworkLayer::CONV)
			std::cerr << ((Conv3dLayer*)*it)->filterDim() << ", " << ((Conv3dLayer*)*it)->filterDim() << ", " << ((Conv3dLayer*)*it)->filterDepth();
		else if((*it)->layerType() == NetworkLayer::MAX_POOL)
			std::cerr << ((Pool3dLayer*)*it)->poolDim() << ", " << ((Pool3dLayer*)*it)->poolDim() << ", " << ((Pool3dLayer*)*it)->poolDepth();
		else if((*it)->layerType() == NetworkLayer::FC)
		{
			std::vector<unsigned> prevLayerSize = _layerSizes[prevLayerName];
			int s = 0;
			for(unsigned i=0; i<prevLayerSize.size(); ++i)
				s += prevLayerSize[i];
			std::cerr << ((DenseLayer*)*it)->numHiddenUnits()*s;
		}
		std::cerr << " ]" << std::endl;

		prevLayerName = layerName;
	}
}

void ActionValueNetwork::init_input(vizdoom::BufferPtr s)
{
	std::cout << "ActionValueNetwork::init_input" << std::endl;
	
	InputLayer *inputLayer = (InputLayer*)*_layers.begin();
	inputLayer->setState(s);
}

// state -> conv3d -> max_pool -> conv3d -> max_pool -> fully_connected -> fully_connected -> softmax
std::vector<float> ActionValueNetwork::get_action_values(vizdoom::BufferPtr s)
{
	std::cout << "ActionValueNetwork::get_action_values" << std::endl;
	
	// Init first layer.
	init_input(s);
	auto it =_layers.begin();
	for(; it!=_layers.end(); it++)
		// activation(prev_layer*W + b)
		(*it)->forwardProp();

	std::vector<float> actionValues = _layers.back()->activations();
	return std::vector<float>(actionValues.begin(), actionValues.end()-1);
}


/*void ActionValueNetwork::init_saxe(unsigned num_rows, unsigned num_cols)
{
	std::cout << "ActionValueNetwork::init_saxe" << std::endl;
}*/

// Kaiming He initialization. Appropriate for ReLU and leaky ReLU activations.
/*void ActionValueNetwork::init_kaiming()
{
	std::cout << "ActionValueNetwork::init_kaiming" << std::endl;

	// Try using member random generator.
	std::default_random_engine g;

	for(auto it=weights_conv.begin(); it!=weights_conv.end(); it++)
	{
		int n = this->filter_dim[it->first]*this->filter_dim[it->first]*this->num_filters[it->first];
		std::normal_distribution<double> normal(0, std::sqrt(2.0/n));
		for(unsigned i=0; i<it->second.size(); ++i)
			for(unsigned j=0; j<it->second[i].size(); ++j)
				for(unsigned k=0; k<it->second[i][j].size(); ++k)
					for(unsigned l=0; l<it->second[i][j][k].size(); ++l)
						it->second[i][j][k][l] = normal(g);
	}

	// Check whether n should be the size of input or output of coresponding layer. 
	// It should be input (fan-in) so check which one of it->second.size or it->second[i].size should be used as n.
	for(auto it=weights_fc.begin(); it!=weights_fc.end(); it++)
	{
		int n = it->second.size();
		std::normal_distribution<double> normal(0, std::sqrt(2.0/n));
		for(unsigned i=0; i<it->second.size(); ++i)
			for(unsigned j=0; j<it->second[i].size(); ++j)
				it->second[i][j] = normal(g);
	}

	//Bias should already be initialized to 0.
}*/

/*std::vector<std::vector<double>> ActionValueNetwork::get_td_update(vizdoom::BufferPtr s, const std::vector<double> &delta_mat)
{
	std::cout << "ActionValueNetwork::get_td_update" << std::endl;

	// Continue here tomorrow.
}*/

/*std::list<std::pair<LayerType, std::pair<std::string, vizdoom::BufferPtr>>> ActionValueNetwork::get_layers() const
{
	std::cout << "ActionValueNetwork::get_layers" << std::endl;

	return this->layers;
}*/

/*std::map<std::string, matrix4d> ActionValueNetwork::get_weights_conv() const
{
	std::cout << "ActionValueNetwork::get_weights" << std::endl;

	return weights_conv;		
}


std::map<std::string, matrix1d> ActionValueNetwork::get_bias_conv() const
{
	std::cout << "ActionValueNetwork::get_weights" << std::endl;

	return bias_conv;		
}

std::map<std::string, matrix2d> ActionValueNetwork::get_weights_fc() const
{
	std::cout << "ActionValueNetwork::get_weights" << std::endl;

	return weights_fc;		
}


std::map<std::string, double> ActionValueNetwork::get_bias_fc() const
{
	std::cout << "ActionValueNetwork::get_weights" << std::endl;

	return bias_fc;		
}


void ActionValueNetwork::set_weights(const std::map<std::string, matrix4d> &w_conv, const std::map<std::string, matrix1d> &b_conv, 
		const std::map<std::string, matrix2d> &w_fc, const std::map<std::string, double> &b_fc)
{
	std::cout << "ActionValueNetwork::set_weights" << std::endl;

	this->weights_conv = w_conv;
	this->bias_conv = b_conv;
	this->weights_fc = w_fc;
	this->bias_fc = b_fc;
}*/

#include "experiencereplaybuffer.h"
#include <experimental/algorithm>

ExperienceReplayBuffer::ExperienceReplayBuffer(unsigned size, unsigned minibatchSize, int seed)
	:_maxSize(size), _minibatchSize(minibatchSize), _randGenerator(std::mt19937(seed))
{
	//std::cout << "ExperienceReplayBuffer::ExperienceReplayBuffer" << std::endl;

	//self.buffer = []
	//self.rand_generator = np.random.RandomState(seed)
	//_randGenerator = std::mt19937{std::random_device{}()};
}

void ExperienceReplayBuffer::append(vizdoom::BufferPtr s, std::vector<double> a, double r, bool t, vizdoom::BufferPtr sp)
{
	//std::cout << "ExperienceReplayBuffer::append" << std::endl;

	if(_buffer.size() == _maxSize)
		_buffer.pop_front();

	_buffer.push_back({s, a, r, t, sp});
}

std::vector<ExperienceSample> ExperienceReplayBuffer::sample() const
{
	//idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
	//return [self.buffer[idx] for idx in idxs]

	// Try to seed member random generator in constructor, and feed it to sample function.
	std::mt19937 r = std::mt19937{std::random_device{}()};
	std::vector<ExperienceSample> randomChoice;
	std::experimental::sample(_buffer.begin(), _buffer.end(), std::back_inserter(randomChoice), _minibatchSize, r);
	return randomChoice;
}

unsigned ExperienceReplayBuffer::getBufferSize() const
{
	return _buffer.size();
}

unsigned ExperienceReplayBuffer::getMinibatchSize() const
{
	return _minibatchSize;
}


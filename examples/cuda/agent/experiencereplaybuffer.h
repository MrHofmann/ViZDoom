#ifndef EXPERIENCEREPLAYBUFFER_H
#define EXPERIENCEREPLAYBUFFER_H

#include "datastructures.h"
#include <random>

class ExperienceReplayBuffer{
private:
	std::list<ExperienceSample> _buffer;
	//self.rand_generator = np.random.RandomState(seed)
	std::mt19937 _randGenerator;
	unsigned _minibatchSize;
	unsigned _maxSize;

public:
	ExperienceReplayBuffer(){}
	ExperienceReplayBuffer(unsigned size, unsigned minibatchSize, int seed);
	void append(vizdoom::BufferPtr state, std::vector<double> action, double reward, bool terminal, vizdoom::BufferPtr nextState);
	std::vector<ExperienceSample> sample() const;
	unsigned getBufferSize() const;
	unsigned getMinibatchSize() const;
};

#endif // EXPERIENCEREPLAYBUFFER_H

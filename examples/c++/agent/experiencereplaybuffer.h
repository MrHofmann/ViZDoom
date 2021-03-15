#ifndef EXPERIENCEREPLAYBUFFER_H
#define EXPERIENCEREPLAYBUFFER_H

#include "datastructures.h"
#include <random>

class ExperienceReplayBuffer{
private:
	std::list<ExperienceSample> buffer;
  //self.rand_generator = np.random.RandomState(seed)
	std::mt19937 rand_generator;
  unsigned minibatch_size;
  unsigned max_size;

public:
  ExperienceReplayBuffer(){}
	ExperienceReplayBuffer(unsigned size, unsigned minibatch_size, int seed);
  void append(vizdoom::BufferPtr state, std::vector<double> action, double reward, bool terminal, vizdoom::BufferPtr next_state);
  std::vector<ExperienceSample> sample() const;
  unsigned get_buffer_size() const;
  unsigned get_minibatch_size() const;
};

#endif // EXPERIENCEREPLAYBUFFER_H

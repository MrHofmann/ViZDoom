#include "experiencereplaybuffer.h"
#include <experimental/algorithm>

ExperienceReplayBuffer::ExperienceReplayBuffer(unsigned size, unsigned minibatch_size, int seed)
	:minibatch_size(minibatch_size), max_size(size), rand_generator(std::mt19937(seed))
{
	//self.buffer = []
	//self.rand_generator = np.random.RandomState(seed)
	//this->rand_generator = std::mt19937{std::random_device{}()};
}

void ExperienceReplayBuffer::append(vizdoom::BufferPtr s, std::vector<double> a, double r, bool t, vizdoom::BufferPtr s_p)
{
	if(this->buffer.size() == this->max_size)
		this->buffer.pop_front();

	this->buffer.push_back({s, a, r, t, s_p});
}

std::vector<ExperienceSample> ExperienceReplayBuffer::sample() const
{
	//idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
	//return [self.buffer[idx] for idx in idxs]

	// Try to seed member random generator in constructor, and feed it to sample function.
	std::mt19937 r = std::mt19937{std::random_device{}()};
	std::vector<ExperienceSample> random_choice;
	std::experimental::sample(buffer.begin(), buffer.end(), std::back_inserter(random_choice), this->minibatch_size, r);
	return random_choice;
}

unsigned ExperienceReplayBuffer::get_buffer_size() const
{
	return this->buffer.size();
}

unsigned ExperienceReplayBuffer::get_minibatch_size() const
{
	return this->minibatch_size;
}


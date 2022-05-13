//
// Created by cleve on 5/13/2022.
//

#include <dataloader/dataloader.h>

#include <utils/random.h>

using namespace cppbp::dataloader;

cppbp::dataloader::DataLoader::DataLoader(cppbp::dataloader::IDataset& ds, size_t batch_size, bool shuffle)
	: ds_(&ds), batch_size_(batch_size), shuffle_(shuffle)
{
}

std::vector<DataPair> cppbp::dataloader::DataLoader::batch()
{
	if (shuffle_)
	{
		return next_rand();
	}
	return next_seq();
}

std::vector<DataPair> cppbp::dataloader::DataLoader::next_rand()
{
	std::vector<int> samples;
	while (samples.size() < batch_size_)
	{
		int b = utils::random::randint(0, ds_->size());
		if (std::find(std::begin(samples), std::end(samples), b) == std::end(samples))
		{
			samples.push_back(b);
		}
	}

	std::vector<DataPair> ret{};
	for (auto i : samples)
	{
		ret.push_back(ds_->get(i));
	}

	return ret;
}

std::vector<DataPair> cppbp::dataloader::DataLoader::next_seq()
{
	std::vector<DataPair> ret{};
	for (int i = 0; i < batch_size_; i++)
	{
		ret.push_back(ds_->get(next_));
		next_ = (next_ + 1) % ds_->size();
	}
	return ret;
}

//
// Created by cleve on 5/13/2022.
//

#include <dataloader/dataloader.h>

#include <utils/random.h>

using namespace cppbp::dataloader;

cppbp::dataloader::DataLoader::DataLoader(cppbp::dataloader::IDataset& ds,
	size_t batch_size,
	bool shuffle,
	double test_ratio)
	: ds_(&ds), batch_size_(batch_size), shuffle_(shuffle)
{
	size_t test_size = static_cast<int>(ds_->size() * test_ratio);
	while (test_.size() < test_size)
	{
		test_.emplace(utils::random::randint(0, ds_->size() - 1));
	}
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
	std::set<int> samples;
	while (samples.size() < batch_size_)
	{
		int b = utils::random::randint(0, ds_->size() - 1);
		if (test_.contains(b))
		{
			continue;
		}

		samples.emplace(b);
	}

	std::vector<DataPair> ret{};
	for (auto i : samples)
	{
		ret.push_back(ds_->get(i));
	}

	return ret;
}

std::vector<DataPair> DataLoader::test()
{
	std::vector<DataPair> ret{};
	for (auto i : test_)
	{
		ret.push_back(ds_->get(i));
	}

	return ret;
}

std::vector<DataPair> cppbp::dataloader::DataLoader::next_seq()
{
	std::vector<DataPair> ret{};
	while (ret.size() < batch_size_)
	{
		if (!test_.contains(next_))
		{
			ret.push_back(ds_->get(next_));
		}
		next_ = (next_ + 1) % ds_->size();
	}
	return ret;
}

//
// Created by cleve on 5/13/2022.
//

#pragma once

#include <dataloader/dataset.h>

#include <vector>
#include <utility>
#include <set>

namespace cppbp::dataloader
{
class DataLoader
{
 public:
	explicit DataLoader(IDataset& ds, size_t batch_size, bool shuffle, double test_ratio = 0.2);

	std::vector<DataPair> train_batch();
	std::vector<DataPair> eval_batch();
 private:
	std::vector<DataPair> next_rand();

	std::vector<DataPair> next_seq();

	IDataset* ds_{};
	size_t batch_size_{};
	bool shuffle_{};

	size_t next_{ 0 };

	std::set<size_t> test_;
};
}
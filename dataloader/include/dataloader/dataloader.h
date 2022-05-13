//
// Created by cleve on 5/13/2022.
//

#pragma once

#include <dataloader/dataset.h>

#include <vector>
#include <utility>

namespace cppbp::dataloader
{
class DataLoader
{
 public:
	explicit DataLoader(IDataset& ds, size_t batch_size, bool shuffle);

	std::vector<DataPair> batch();
 private:
	std::vector<DataPair> next_rand();

	std::vector<DataPair> next_seq();

	IDataset* ds_{};
	size_t batch_size_{};
	bool shuffle_{};

	size_t next_{0};
};
}
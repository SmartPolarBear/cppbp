//
// Created by cleve on 5/13/2022.
//
#pragma once

#include <vector>
#include <utility>

namespace cppbp::dataloader
{
using DataPair = std::pair<std::vector<double>, std::vector<double>>;

class IDataset
{
 public:
	virtual DataPair get(size_t index) const = 0;
	virtual size_t size() const = 0;
};
}
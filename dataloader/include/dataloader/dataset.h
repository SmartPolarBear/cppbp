//
// Created by cleve on 5/13/2022.
//
#pragma once

#include <base/base.h>

#include <utility>
#include <vector>

#include <Eigen/Eigen>

namespace cppbp::dataloader
{
using DataPair = std::pair<base::VectorType, base::VectorType >;

class IDataset
{
 public:
	using DatasetPair = std::pair<IDataset, IDataset>;

	virtual DataPair get(size_t index) const = 0;
	virtual size_t size() const = 0;
};
}// namespace cppbp::dataloader
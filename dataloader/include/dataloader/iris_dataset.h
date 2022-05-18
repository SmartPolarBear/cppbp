//
// Created by cleve on 5/13/2022.
//

#pragma once

#include "dataset.h"
#include <csv2/reader.hpp>

#include <string>
#include <tuple>
#include <unordered_map>

namespace cppbp::dataloader
{

class IrisDataset
	: public IDataset
{
 public:
	explicit IrisDataset(std::string pathname);

	DataPair get(size_t index) const override;
	size_t size() const override;

 private:

	static inline const std::unordered_map<std::string, int> LABEL_TO_ID{
		{ "Iris-setosa", 0 },
		{ "Iris-versicolor", 1 },
		{ "Iris-virginica", 2 }
	};

	std::string path_{};
	std::vector<DataPair> data_{};
};

}
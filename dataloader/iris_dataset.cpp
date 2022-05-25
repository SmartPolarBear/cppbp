//
// Created by cleve on 5/11/2022.
//
#include <dataloader/iris_dataset.h>

#include <algorithm>
#include <iostream>

using namespace cppbp::dataloader;

using namespace std;

using namespace Eigen;

cppbp::dataloader::IrisDataset::IrisDataset(std::string pathname, bool regularize)
	: IDataset(), path_(std::move(pathname)), regularize_(regularize)
{
	csv2::Reader<csv2::delimiter<','>,
				 csv2::quote_character<'"'>,
				 csv2::first_row_is_header<false>,
				 csv2::trim_policy::trim_whitespace>
		csv{};

	VectorXd sum = VectorXd::Zero(4);
	VectorXd squared = VectorXd::Zero(4);

	if (csv.mmap(path_))
	{
		for (const auto& row : csv)
		{
			VectorXd params = VectorXd::Zero(4);
			VectorXd label = VectorXd::Zero(3);

			for (int col = 0; auto cell : row)
			{
				std::string value{};
				cell.read_value(value);
				if (col < 4)
				{
					params[col] = std::stod(value);
				}
				else
				{
					auto one_hot = LABEL_TO_ID.at(value);
					label[one_hot] = 1;
				}

				col++;
			}

			data_.emplace_back(params, label);

			sum += params;
			squared += params.cwiseProduct(params);
		}
	}

	if (regularize_)
	{
		sum /= data_.size();
		squared /= data_.size();

		VectorXd variance((squared - sum.cwiseProduct(sum)).array().sqrt());

		for (auto& d : data_)
		{
			d.first -= sum;
			d.first = d.first.cwiseQuotient(variance);
		}
	}
}

DataPair cppbp::dataloader::IrisDataset::get(size_t index) const
{
	return data_.at(index);
}

size_t cppbp::dataloader::IrisDataset::size() const
{
	return data_.size();
}

//
// Created by cleve on 5/11/2022.
//
#include <dataloader/iris_dataset.h>

#include <algorithm>

cppbp::dataloader::IrisDataset::IrisDataset(std::string pathname)
	: path_(std::move(pathname))
{
	csv2::Reader<csv2::delimiter<','>,
				 csv2::quote_character<'"'>,
				 csv2::first_row_is_header<false>,
				 csv2::trim_policy::trim_whitespace> csv{};
	if (csv.mmap(path_))
	{
		for (const auto& row : csv)
		{
			std::vector<double> params{};
			std::vector<double> label{};

			for (int col = 0; auto cell : row)
			{
				std::string value{};
				cell.read_value(value);
				if (col < 4)
				{
					params.push_back(std::stod(value));
				}
				else
				{
					auto one_hot = LABEL_TO_ID.at(value);
					label.resize(3);
					label[one_hot] = 1;
				}
				col++;
			}

			data_.emplace_back(params, label);
		}
	}
}

std::pair<std::vector<double>, std::vector<double>> cppbp::dataloader::IrisDataset::get(size_t index) const
{
	return data_.at(index);
}

size_t cppbp::dataloader::IrisDataset::size() const
{
	return data_.size();
}



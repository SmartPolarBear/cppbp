//
// Created by cleve on 5/31/2022.
//

#pragma once

#include "dataset.h"
#include <csv2/reader.hpp>

#include <string>
#include <tuple>
#include <unordered_map>

namespace cppbp::dataloader
{

class MNISTDataset
        : public IDataset
{
public:
    explicit MNISTDataset(std::string label_path, std::string img_path, bool regularize);

    DataPair get(size_t index) const override;

    size_t size() const override;

private:
    void load_labels();
    void load_images();

    std::string label_path_{};
    std::string img_path_{};

    std::vector<base::VectorType> lbls_{};
    std::vector<base::VectorType> imgs_{};

    bool regularize_{};
};

}
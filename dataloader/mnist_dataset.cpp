//
// Created by cleve on 5/31/2022.
//
#include <dataloader/mnist_dataset.h>

#include <utils/utils.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <gsl/gsl>

using namespace cppbp::dataloader;
using namespace cppbp::utils;

using namespace std;
using namespace gsl;

using namespace Eigen;

MNISTDataset::MNISTDataset(std::string label_path, std::string img_path, bool regularize)
        : IDataset(), label_path_(std::move(label_path)), img_path_(std::move(label_path)), regularize_(regularize)
{
    load_labels();
    load_images();

    if (regularize_)
    {
        base::VectorType sum = base::VectorType::Zero(imgs_[0].size());
        base::VectorType squared = base::VectorType::Zero(imgs_[0].size());

        for (const auto& vec: imgs_)
        {
            sum += vec;
            squared += vec.cwiseProduct(vec);
        }

        sum /= imgs_.size();
        squared /= imgs_.size();

        VectorXd variance((squared - sum.cwiseProduct(sum)).array().sqrt());

        for (auto &d: imgs_)
        {
            d -= sum;
            d = d.cwiseQuotient(variance);
        }
    }
}

DataPair cppbp::dataloader::MNISTDataset::get(size_t index) const
{
    return make_pair(imgs_.at(index), lbls_.at(index));
}

size_t cppbp::dataloader::MNISTDataset::size() const
{
    return lbls_.size();
}

void MNISTDataset::load_labels()
{
    std::ifstream file(label_path_, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file");
    }
    auto _ = finally([&]()
                     {
                         file.close();
                     });

    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;

    file >> magic_number;
    file >> number_of_images;

    endian_swap(magic_number);
    endian_swap(number_of_images);

    for (int i = 0; i < number_of_images; ++i)
    {
        uint8_t num = 0;
        file >> num;
        base::VectorType vec(10);
        vec.coeffRef(num) = 1;
        lbls_.push_back(vec);
    }
}

void MNISTDataset::load_images()
{
    std::ifstream file(img_path_, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file");
    }
    auto _ = finally([&]()
                     {
                         file.close();
                     });

    uint32_t magic_number = 0;
    uint32_t img_cnt = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    file >> magic_number;
    file >> img_cnt;
    file >> rows;
    file >> cols;

    endian_swap(magic_number);
    endian_swap(img_cnt);
    endian_swap(rows);
    endian_swap(cols);

    for (int i = 0; i < img_cnt; ++i)
    {
        base::VectorType vec(rows * cols);
        size_t cnt = 0;
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                file >> vec.coeffRef(cnt++);
            }
        }
        imgs_.push_back(vec);
    }
}



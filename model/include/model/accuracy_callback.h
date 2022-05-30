//
// Created by cleve on 5/30/2022.
//

#pragma once

#include <model/callback.h>

#include <vector>
#include <unordered_map>

namespace cppbp::model
{
class AccuracyCallback
        : public IModelCallback
{
public:
    AccuracyCallback();

    explicit AccuracyCallback(std::initializer_list<int> k);

    std::string before_world() override;

    std::string after_world() override;

    std::string before_train(size_t step) override;

    std::string train_step(size_t step, const dataloader::DataPair &dp, Eigen::VectorXd predicts, double loss) override;

    std::string after_train(size_t epoch) override;

    std::string before_eval(size_t epoch) override;

    std::string eval_step(size_t step, const dataloader::DataPair &dp, Eigen::VectorXd predicts, double loss) override;

    std::string after_eval(size_t epoch) override;

private:
    size_t total_{};
    std::unordered_map<int, size_t> accurate_{};
    std::vector<int> k_{};
};
}
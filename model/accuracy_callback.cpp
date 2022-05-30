//
// Created by cleve on 5/30/2022.
//

#include <model/accuracy_callback.h>

#include <utils/utils.h>

#include <sstream>

#include <fmt/format.h>

using namespace std;

cppbp::model::AccuracyCallback::AccuracyCallback()
        : k_{1}
{

}

cppbp::model::AccuracyCallback::AccuracyCallback(std::initializer_list<int> k)
        : k_(k)
{

}

std::string cppbp::model::AccuracyCallback::before_world()
{
    return "";
}

std::string cppbp::model::AccuracyCallback::after_world()
{
    return "";
}

std::string cppbp::model::AccuracyCallback::before_train(size_t step)
{
    total_ = 0;
    accurate_.clear();
    for (auto k: k_)
    {
        accurate_[k] = 0;
    }
    return "";
}

std::string
cppbp::model::AccuracyCallback::train_step(size_t step, const cppbp::dataloader::DataPair &dp, Eigen::VectorXd predicts,
                                           double loss)
{
    total_++;
    const auto &[data, label] = dp;
    auto ids = utils::argsort(predicts);
    auto gt = utils::argmax(label);
    for (auto k: k_)
    {
        for (int i = 0; i < k; i++)
        {
            if (ids[i] == gt)
            {
                accurate_[k] += 1;
                break;
            }
        }
    }
    return "";
}

std::string cppbp::model::AccuracyCallback::after_train(size_t epoch)
{
    stringstream ss{};
    for (auto k: k_)
    {
        ss << fmt::format("Top {} accuracy: {} ", k, accurate_[k] / static_cast<double>(total_));
    }
    return ss.str();
}

std::string cppbp::model::AccuracyCallback::before_eval(size_t epoch)
{
    return before_train(epoch);
}

std::string
cppbp::model::AccuracyCallback::eval_step(size_t step, const cppbp::dataloader::DataPair &dp, Eigen::VectorXd predicts,
                                          double loss)
{
    return train_step(step, dp, predicts, loss);
}

std::string cppbp::model::AccuracyCallback::after_eval(size_t epoch)
{
    return after_train(epoch);
}



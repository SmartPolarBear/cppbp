//
// Created by cleve on 5/26/2022.
//

#pragma once

#include <base/serializable.h>

#include <layer/layer.h>
#include <layer/sigmoid.h>

#include <model/persist.h>

#include <utils/counter.h>

#include <Eigen/Eigen>

#include <cstdint>
#include <vector>

namespace cppbp::layer
{
class BatchNorm
        : public ILayer,
          public utils::Counter<BatchNorm>
{
public:
     BatchNorm();

    void backprop() override;

    void forward() override;

    ILayer *next() override;

    ILayer *prev() override;

    void set_prev(ILayer *prev) override;

    void set_next(ILayer *next) override;

    std::tuple<std::shared_ptr<char[]>, size_t> serialize() override;

    char *deserialize(char *data) override;

    std::string name() const override;

    std::string summary() const override;

    ILayer &connect(ILayer &next) override;

    void set(Eigen::VectorXd vec) override;

    void set_deltas(Eigen::VectorXd deltas) override;

    void set_errors(Eigen::VectorXd errors) override;

    Eigen::VectorXd get() const override;

    IActivationFunction &activation_function() override;

    ILayer &operator|(ILayer &next) override;

    void reshape(size_t input) override;

    void optimize(optimizer::IOptimizer &optimizer_1) override;

private:
    uint64_t id_{};

    size_t input_len_{};

    Sigmoid placeholder{};

    double sigma_{};

    Eigen::VectorXd input_;
    Eigen::VectorXd scaled_input_;
    Eigen::VectorXd normalized_input_;

    Eigen::VectorXd activations_;

    Eigen::VectorXd deltas_;
    Eigen::VectorXd errors_;

    Eigen::VectorXd gammas_;
    Eigen::VectorXd betas_;

    ILayer* next_{};
    ILayer* prev_{};
};
}// namespace cppbp::layer
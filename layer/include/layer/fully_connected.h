//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <base/serializable.h>
#include <base/magic.h>

#include <layer/layer.h>


#include <utils/counter.h>

#include <Eigen/Eigen>

#include <cstdint>
#include <vector>

namespace cppbp::layer
{
class FullyConnected
        : public ILayer,
          public utils::Counter<FullyConnected>,
          public base::IMagic<uint16_t>
{
public:
    uint16_t magic() const override;

    FullyConnected() = default;

    std::ostream &serialize(std::ostream &out) override;

    std::istream &deserialize(std::istream &input) override;

    ILayer *next() override;

    ILayer *prev() override;

    explicit FullyConnected(size_t len, IActivationFunction &af);

    void reshape(size_t input) override;

    void set_prev(ILayer *prev) override;

    void set_next(ILayer *next) override;

    ILayer &connect(ILayer &next) override;

    ILayer &operator|(ILayer &next) override;

    void set(Eigen::VectorXd vec) override;

    [[nodiscard]] Eigen::VectorXd get() const override;

    void set_deltas(Eigen::VectorXd dlts) override;

    void backprop() override;

    void forward() override;

    void set_errors(Eigen::VectorXd errors) override;

    void optimize(optimizer::IOptimizer &opt) override;

    [[nodiscard]] std::string name() const override;

    [[nodiscard]] std::string summary() const override;

    IActivationFunction &activation_function() override;

private:
    uint64_t id_{};

    IActivationFunction *act_func_{nullptr};

    Eigen::MatrixXd weights_{};

    Eigen::VectorXd input_;
    Eigen::VectorXd activations_;

    Eigen::VectorXd deltas_;
    Eigen::VectorXd errors_;

    size_t len_{};

    ILayer *next_{};
    ILayer *prev_{};

    // To work around the lifetime issues
    std::shared_ptr<IActivationFunction> restored_act_func_{};
};

}


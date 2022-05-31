//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <layer/layer.h>

#include <base/magic.h>
#include <base/serializable.h>

#include <optimizer/optimizer.h>
#include <optimizer/loss.h>

#include <model/callback.h>

#include <dataloader/dataloader.h>

#include <vector>
#include <optional>

#include <concepts>

namespace cppbp::model
{

template<typename T>
concept IterableData=
requires(T t)
{
    { t.begin() };
    { t.end() };
};

template<typename T>
concept SizedData=
requires(T t, int i)
{
    { t.size() }->std::convertible_to<size_t>;
    { t.get(i) };
};

class Model
        : public base::IForward,
          public base::IBackProp,
          public base::ISummary,
          public base::INamable,
          public base::ISerializable,
          public optimizer::IOptimizable,
          public base::IMagic<uint64_t>
{
public:
    explicit Model(layer::ILayer &layer, optimizer::ILossFunction &loss);

    explicit Model(std::vector<layer::ILayer> &layers, optimizer::ILossFunction &loss);

    Eigen::VectorXd operator()(std::vector<double> input);

    Eigen::VectorXd operator()(Eigen::VectorXd input);

    void fit(cppbp::dataloader::DataLoader &dl,
             size_t epoch,
             cppbp::optimizer::IOptimizer &opt,
             bool verbose,
             size_t callback_skip_epoch = 100,
             std::optional<std::vector<std::shared_ptr<IModelCallback>>> cbks = std::nullopt);

    void evaluate(cppbp::dataloader::DataLoader &dl,
                  bool verbose,
                  std::optional<std::vector<std::shared_ptr<IModelCallback>>> cbks = std::nullopt);

    template<IterableData T>
    std::vector<base::VectorType> predict(const T &dataset)
    {
        std::vector<base::VectorType> ret{};
        for (const auto &d: dataset)
        {
            const auto &[data, label] = d;
            ret.push_back((*this)(data));
        }
        return ret;
    }

    template<SizedData T>
    std::vector<base::VectorType> predict(const T &dataset)
    {
        std::vector<base::VectorType> ret{};
        for (int i = 0; i < dataset.size(); i++)
        {
            auto [data, label] = dataset.get(i);
            ret.push_back((*this)(data));
        }
        return ret;
    }

    [[nodiscard]] std::string name() const override;

    [[nodiscard]] std::string summary() const override;

    void optimize(optimizer::IOptimizer &iOptimizer) override;

    void save_state(const std::string &filename);

    void load_state(const std::string &filename);

    std::ostream &serialize(std::ostream &out) override;

    std::istream &deserialize(std::istream &input) override;

    uint64_t magic() const override;

private:
    Model() = default;

    void set(std::vector<double> values);

    void forward() override;

    void backprop() override;

    layer::ILayer *input_{}, *output_{};
    std::string name_{"Model"};
    optimizer::ILossFunction *loss_{};

    // To work around lifetime issues
    std::shared_ptr<optimizer::ILossFunction> restored_loss_{};
    std::vector<std::shared_ptr<layer::ILayer>> restored_layers_{};

};
}
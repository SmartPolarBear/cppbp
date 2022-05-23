//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <layer/layer.h>

#include <base/serializable.h>

#include <optimizer/optimizer.h>
#include <optimizer/loss.h>

#include <model/persist.h>

#include <dataloader/dataloader.h>

#include <vector>
#include <optional>

namespace cppbp::model
{
class Model
	: public base::IForward,
	  public base::IBackProp,
	  public base::ISummary,
	  public base::INamable,
	  public base::ISerializable,
	  public optimizer::IOptimizable
{
 public:
	explicit Model(layer::ILayer& layer, optimizer::ILossFunction& loss);

	explicit Model(std::vector<layer::ILayer>& layers, optimizer::ILossFunction& loss);

	Eigen::VectorXd operator()(std::vector<double> input);
	Eigen::VectorXd operator()(Eigen::VectorXd input);

	void fit(cppbp::dataloader::DataLoader& dl,
		size_t epoch,
		cppbp::optimizer::IOptimizer& opt,
		bool verbose);

	[[nodiscard]] std::string name() const override;

	[[nodiscard]] std::string summary() const override;

	void optimize(optimizer::IOptimizer& iOptimizer) override;

	void save(const std::string& filename);

	static inline std::optional<Model> from_file(const std::string& filename);

	std::tuple<std::shared_ptr<char[]>, size_t> serialize() override;

	char* deserialize(char* data) override;

 private:
	Model();

	void set(std::vector<double> values);

	void forward() override;

	void backprop() override;

	layer::ILayer* input_{}, * output_{};
	std::string name_{ "Model" };
	optimizer::ILossFunction* loss_{};

	// To work around lifetime issues
	std::shared_ptr<optimizer::ILossFunction> restored_loss_{};
	std::vector<std::shared_ptr<layer::ILayer>> restored_layers_{};

};
}
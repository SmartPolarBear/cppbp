//
// Created by cleve on 5/23/2022.
//

#pragma once

#include <model/callback.h>

namespace cppbp::model
{
class LossOutputCallback
	: public IModelCallback
{
 public:
	std::string before_world() override;
	std::string after_world() override;
	std::string before_train(size_t step) override;
	std::string train_step(size_t step, const dataloader::DataPair& dp, Eigen::VectorXd predicts, double loss) override;
	std::string after_train(size_t epoch) override;
	std::string before_eval(size_t epoch) override;
	std::string eval_step(size_t step, const dataloader::DataPair& dp, Eigen::VectorXd predicts, double loss) override;
	std::string after_eval(size_t epoch) override;
 private:
	std::vector<double> losses_;
};
}
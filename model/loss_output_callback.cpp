//
// Created by cleve on 5/23/2022.
//

#include <model/loss_output_callback.h>

#include <fmt/format.h>

std::string cppbp::model::LossOutputCallback::before_world()
{
	return "";
}

std::string cppbp::model::LossOutputCallback::after_world()
{
	return "";
}

std::string cppbp::model::LossOutputCallback::before_train(size_t step)
{
	losses_.clear();
	return "";
}

std::string cppbp::model::LossOutputCallback::train_step(size_t step,
	const dataloader::DataPair& dp,
	Eigen::VectorXd predicts,
	double loss)
{
	losses_.push_back(loss);
	return fmt::format("Loss:{}", loss);
}

std::string cppbp::model::LossOutputCallback::after_train(size_t epoch)
{
	double sum{ 0 };
	for (auto l : losses_)
	{
		sum += l;
	}
	double size = losses_.size();
	losses_.clear();
	return fmt::format("After training, average loss is {}", sum / size);
}

std::string cppbp::model::LossOutputCallback::before_eval(size_t epoch)
{
	losses_.clear();
	return "";
}

std::string cppbp::model::LossOutputCallback::eval_step(size_t step,
	const dataloader::DataPair& dp,
	Eigen::VectorXd predicts,
	double loss)
{
	losses_.push_back(loss);
	return fmt::format("Loss:{}", loss);
}

std::string cppbp::model::LossOutputCallback::after_eval(size_t epoch)
{
	double sum{ 0 };
	for (auto l : losses_)
	{
		sum += l;
	}
	double size = losses_.size();
	losses_.clear();
	return fmt::format("After eval, average loss is {}", sum / size);
}

//
// Created by cleve on 5/23/2022.
//

#pragma once

#include <dataloader/dataset.h>

#include <cstdint>
#include <string>
#include <memory>

namespace cppbp::model
{
class IModelCallback
{
 public:

	template<typename T, typename... Args>
	static inline std::shared_ptr<IModelCallback> make(Args&& ... args)
	{
		return std::make_shared<T>(std::forward<Args>(args)...);
	}

	virtual std::string before_world() = 0;
	virtual std::string after_world() = 0;

	virtual std::string before_train(size_t step) = 0;
	virtual std::string train_step(size_t step,
		const dataloader::DataPair& dp,
		Eigen::VectorXd predicts,
		double loss) = 0;
	virtual std::string after_train(size_t epoch) = 0;

	virtual std::string before_eval(size_t epoch) = 0;
	virtual std::string eval_step(size_t step,
		const dataloader::DataPair& dp,
		Eigen::VectorXd predicts,
		double loss) = 0;
	virtual std::string after_eval(size_t epoch) = 0;
};
}
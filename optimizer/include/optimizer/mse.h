//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <optimizer/loss.h>

#include <model/persist.h>


namespace cppbp::optimizer
{
class MSELoss
	: public ILossFunction
{
 public:
	double operator()(Eigen::VectorXd value, Eigen::VectorXd label) override;
	double eval(Eigen::VectorXd value, Eigen::VectorXd label) override;
	Eigen::VectorXd derive(Eigen::VectorXd value, Eigen::VectorXd label) override;

};


template<>
struct cppbp::model::persist::LossFunctionTypeId<MSELoss>
{
	static inline constexpr uint32_t value = 1;
};

}
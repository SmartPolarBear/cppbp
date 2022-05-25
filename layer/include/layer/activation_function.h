//
// Created by cleve on 5/11/2022.
//
#pragma once

#include <base/type_id.h>

#include <Eigen/Eigen>

#include <memory>

namespace cppbp::layer
{
class IActivationFunction
	: public base::ITypeId
{
 public:
	virtual double operator()(double x) = 0;

	virtual double eval(double x) = 0;
	virtual double derive(double y) = 0;

	virtual Eigen::VectorXd eval(Eigen::VectorXd x) = 0;
	virtual Eigen::MatrixXd derive(Eigen::VectorXd y) = 0;

};

class ActivationFunctionFactory final
{
 public:
	static std::shared_ptr<cppbp::layer::IActivationFunction> from_id(uint32_t id);

	template<typename Callback>
	static std::shared_ptr<cppbp::layer::IActivationFunction> from_id(uint32_t id, Callback cbk)
	{
		try
		{
			from_id(id);
		}
		catch (...)
		{
			return cbk(id);
		}
	}

};

}
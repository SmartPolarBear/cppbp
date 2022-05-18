//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <base/forward.h>
#include <base/backprop.h>
#include <base/summary.h>
#include <base/linked.h>

#include <optimizer/optimizer.h>

#include <layer/activation_function.h>

#include <Eigen/Eigen>

#include <memory>
#include <vector>

namespace cppbp::layer
{
class ILayer
	: public base::IForward,
	  public base::IBackProp,
	  public base::ISummary,
	  public base::INamable,
	  public optimizer::IOptimizable,
	  public base::ILinked<ILayer>
{
 public:
	virtual ILayer& connect(ILayer& next) = 0;

	virtual void set(Eigen::VectorXd vec) = 0;

	virtual void set_errors(Eigen::VectorXd error) = 0;

	virtual std::vector<double> get() const = 0;

	virtual IActivationFunction& activation_function() = 0;

	virtual ILayer& operator|(ILayer& next) = 0;

	virtual void reshape(size_t input) = 0;
};
}
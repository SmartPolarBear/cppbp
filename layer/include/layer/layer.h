//
// Created by cleve on 5/11/2022.
//

#pragma once

#include <base/forward.h>
#include <base/backprop.h>
#include <base/summary.h>
#include <base/linked.h>

#include <memory>
#include <vector>

namespace cppbp::layer
{
class ILayer
	: public base::IForward,
	  public base::IBackProp,
	  public base::ISummary,
	  public base::INamable,
	  public base::ILinked<ILayer>
{
 public:
	virtual ILayer& connect(ILayer& next) = 0;

	virtual void set(std::vector<double> values) = 0;

	virtual std::vector<double> get() const = 0;

	virtual ILayer& operator|(ILayer& next) = 0;
};
}
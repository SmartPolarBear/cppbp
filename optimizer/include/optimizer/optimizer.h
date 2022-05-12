//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <vector>

namespace cppbp::optimizer
{
class IOptimizer
{
 public:
	virtual void step() = 0;
	virtual double optimize(double prev, double grad) = 0;
};

class IOptimizable
{
 public:
	virtual void optimize(IOptimizer&) = 0;
};
}
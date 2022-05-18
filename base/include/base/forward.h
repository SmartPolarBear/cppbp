
//
// Created by cleve on 5/12/2022.
//

#pragma once

namespace cppbp::base
{
/// An interface for forward propagation
class IForward
{
 public:
	virtual void forward() = 0;
};
}
//
// Created by cleve on 5/12/2022.
//

#pragma once

namespace cppbp::base
{
/// An interface for back propagation
class IBackProp
{
 public:
	virtual void backprop() = 0;
};
}
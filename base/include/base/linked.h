//
// Created by cleve on 5/13/2022.
//

#pragma once

namespace cppbp::base
{
template<typename T>
class ILinked
{
 public:
	virtual T* next() = 0;
	virtual T* prev() = 0;
};
}
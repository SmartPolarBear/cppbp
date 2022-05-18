//
// Created by cleve on 5/13/2022.
//

#pragma once

namespace cppbp::base
{
/// Simple doubly-linked list
template<typename T>
class ILinked
{
 public:
	virtual T* next() = 0;
	virtual T* prev() = 0;


	virtual void set_prev(T* prev) = 0;

	virtual void set_next(T* next) = 0;
};
}
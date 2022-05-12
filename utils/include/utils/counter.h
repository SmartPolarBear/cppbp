//
// Created by cleve on 5/12/2022.
//

#pragma once

namespace cppbp::utils
{
template<typename T>
struct Counter
{
	static inline int objects_created = 0;
	static inline int objects_alive = 0;

	Counter()
	{
		++objects_created;
		++objects_alive;
	}

	Counter(const Counter&)
	{
		++objects_created;
		++objects_alive;
	}
 protected:
	~Counter() // objects should never be removed through pointers of this type
	{
		--objects_alive;
	}
};

}
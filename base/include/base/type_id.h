//
// Created by cleve on 5/23/2022.
//
#pragma once

#include <cstdint>

namespace cppbp::base
{
class ITypeId
{
 public:
	virtual uint32_t type_id() = 0;
};
}
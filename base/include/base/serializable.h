//
// Created by cleve on 5/23/2022.
//

#pragma once
#include <memory>
#include <tuple>

namespace cppbp::base
{
class ISerializable
{
 public:
	virtual std::tuple<std::shared_ptr<char[]>, size_t> serialize() = 0;
	virtual char* deserialize(char* data) = 0;
};
}
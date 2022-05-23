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
	virtual std::tuple<std::unique_ptr<char>, size_t> serialize() = 0;
	virtual std::unique_ptr<char> deserialize(std::unique_ptr<char> data) = 0;
};
}
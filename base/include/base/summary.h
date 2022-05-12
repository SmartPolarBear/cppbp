//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <string>

namespace cppbp::base
{
class INamable
{
 public:
	[[nodiscard]] virtual std::string name() const = 0;
};

class ISummary
{
 public:
	[[nodiscard]] virtual std::string summary() const = 0;
};
}
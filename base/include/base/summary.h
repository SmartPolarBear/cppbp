//
// Created by cleve on 5/12/2022.
//

#pragma once

#include <string>

namespace cppbp::base
{
/// An interface for objects that have a name property
class INamable
{
 public:
	[[nodiscard]] virtual std::string name() const = 0;
};

// An interface for pretty-print the details of the object
class ISummary
{
 public:
	[[nodiscard]] virtual std::string summary() const = 0;
};
}
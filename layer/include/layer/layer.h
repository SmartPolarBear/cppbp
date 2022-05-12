//
// Created by cleve on 5/11/2022.
//

#include <base/forward.h>
#include <base/backprop.h>
#include <base/summary.h>

#include <memory>

namespace cppbp::layer
{
class ILayer
	: public base::IForward,
	  public base::IBackProp,
	  public base::ISummary,
	  public base::INamable
{
};
}
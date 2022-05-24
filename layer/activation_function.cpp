//
// Created by cleve on 5/23/2022.
//

#include <layer/activation_function.h>
#include <layer/sigmoid.h>
#include <layer/relu.h>

std::shared_ptr<cppbp::layer::IActivationFunction> cppbp::layer::ActivationFunctionFactory::from_id(uint32_t id)
{
	switch (id)
	{
	case 1:
		return std::make_shared<Sigmoid>();
	case 2:
		return std::make_shared<Relu>();
	default:
		throw;
		// TODO: exception
	}
}

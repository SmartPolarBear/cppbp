//
// Created by cleve on 5/11/2022.
//
#pragma once

namespace cppbp::layer
{
class IActivationFunction
{
 public:
	virtual double operator()(double x) = 0;

	virtual double eval(double x) = 0;
	virtual double derive(double y) = 0;
};
}
//
// Created by cleve on 5/11/2022.
//

namespace cppbp::layer
{
class IActivationFunction
{
 public:
	virtual double operator()(double x) = 0;

	virtual double eval(double x) = 0;
	virtual double derive(double x) = 0;
};
}
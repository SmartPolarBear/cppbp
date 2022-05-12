//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>
#include <layer/sigmoid.h>

#include <iostream>

using namespace cppbp;
using namespace cppbp::layer;

int main()
{
	Sigmoid sigmoid{};
	FullyConnected fc1{ 5, sigmoid };
	FullyConnected fc2{ 15, sigmoid };
	FullyConnected fc3{ 4, sigmoid };

	fc1.connect(fc2).connect(fc3);

	std::cout << fc1.summary();

	return 0;
}
//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>
#include <layer/sigmoid.h>

#include <iostream>
#include <vector>

using namespace cppbp;
using namespace cppbp::layer;

using namespace std;

int main()
{
	Sigmoid sigmoid{};

	FullyConnected fc1{ 3, sigmoid };
	FullyConnected fc2{ 15, sigmoid };
	FullyConnected fc3{ 4, sigmoid };

	fc1.connect(fc2).connect(fc3);

	std::cout << fc1.summary() << endl;

	vector<double> input{ 0, 0, 0 };
	fc1.set(input);
	fc1.forward();
	auto ret = fc3.get();

	for (auto r : ret)
	{
		cout << r << " ";
	}

	return 0;
}
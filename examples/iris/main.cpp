//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>
#include <layer/sigmoid.h>

#include <model/model.h>

#include <optimizer/fixed_step_optimizer.h>
#include <optimizer/mse.h>

#include <dataloader/iris_dataset.h>

#include <iostream>
#include <vector>

using namespace cppbp;
using namespace cppbp::layer;
using namespace cppbp::model;
using namespace cppbp::optimizer;
using namespace cppbp::dataloader;

using namespace std;

int main()
{
	Sigmoid sigmoid{};

	FullyConnected fc1{ 3, sigmoid };
	FullyConnected fc2{ 15, sigmoid };
	FullyConnected fc3{ 4, sigmoid };

	MSELoss loss{};

	Model model{ fc1.connect(fc2).connect(fc3), loss };

	std::cout << model.summary() << endl;

	IrisDataset iris{ "data/iris.data" };

	auto d = iris.get(15);

	vector<double> input{ 0, 0, 0 };

	auto ret = model(input);

	for (auto r : ret)
	{
		cout << r << " ";
	}

	return 0;
}
//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>
#include <layer/sigmoid.h>

#include <model/model.h>

#include <optimizer/fixed_step_optimizer.h>
#include <optimizer/mse.h>

#include <dataloader/iris_dataset.h>
#include <dataloader/dataloader.h>

#include <iostream>
#include <vector>

using namespace cppbp;
using namespace cppbp::layer;
using namespace cppbp::model;
using namespace cppbp::optimizer;
using namespace cppbp::dataloader;

using namespace std;

int argmax(const std::vector<double>& vals)
{
	int ret = 0;
	for (int i = 1; i < vals.size(); i++)
	{
		if (vals[i] > vals[ret])
		{
			ret = i;
		}
	}
	return ret;
}

int main()
{
	Sigmoid sigmoid{};

	FullyConnected fc1{ 4, sigmoid };
	FullyConnected fc2{ 8, sigmoid };
	FullyConnected fc3{ 3, sigmoid };

	MSELoss loss{};
	Model model{ fc1.connect(fc2).connect(fc3), loss };

	std::cout << model.summary() << endl;

	IrisDataset iris{ "data/iris.data" };
	DataLoader dl{ iris, 16, true };

	FixedStepOptimizer optimizer{ 0.2 };
	model.fit(dl, 100, optimizer, true);

	for (int i = 0; i < iris.size(); i++)
	{
		auto [data, label] = iris.get(i);
		auto ret = model(data);
		cout << argmax(label) << "," << argmax(ret) << endl;
	}

	return 0;
}
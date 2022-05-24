//
// Created by cleve on 5/11/2022.
//

#include <layer/fully_connected.h>
#include <layer/input.h>

#include <model/model.h>

#include <optimizer/fixed_step_optimizer.h>
#include <optimizer/mse.h>
#include <optimizer/sgd_optimizer.h>

#include <dataloader/iris_dataset.h>
#include <dataloader/dataloader.h>

#include <fmt/format.h>

#include <iostream>
#include <vector>

using namespace cppbp;
using namespace cppbp::layer;
using namespace cppbp::model;
using namespace cppbp::optimizer;
using namespace cppbp::dataloader;

using namespace std;


int argmax(const Eigen::VectorXd& vals)
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
int acc = 0;
void res(Eigen::VectorXd& label,Eigen::VectorXd& ret)
{
	int a ,b;
	a = argmax(label);
	b = argmax(ret);
	if(a == b){
		acc++;
	}
	cout << fmt::format("Ground Truth:{}, Predict:{}", a, b) << endl;
}


int main()
{
	Sigmoid sigmoid{};

	Input in{ 4 };
	FullyConnected fc1{ 5, sigmoid };
	FullyConnected fc2{ 8, sigmoid };
	FullyConnected fc3{ 12, sigmoid };
	FullyConnected out{ 3, sigmoid };

	MSELoss loss{};
	Model model{ in | fc1 | fc2 | fc3 | out, loss };

	std::cout << model.summary() << endl;

	IrisDataset iris{ "data/iris.data" };
	DataLoader dl{ iris, 16, false };

	//FixedStepOptimizer optimizer{ 0.2 };
	Sgd_optimizer optimizer{0.3};
	model.fit(dl, 1800, optimizer, false);
	for (int i = 0; i < iris.size(); i++)
	{
		auto [data, label] = iris.get(i);
		auto ret = model(data);
		res(label,ret);

	}
	cout << acc << endl;
	return 0;
}
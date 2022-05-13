//
// Created by cleve on 5/12/2022.
//

#include <utils/random.h>

#include <random>

using namespace std;

double cppbp::utils::random::uniform(double l, double r)
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(l, r);
	return dist(mt);
}

int cppbp::utils::random::randint(int l, int r)
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<int> dist(l, r);
	return dist(mt);
}



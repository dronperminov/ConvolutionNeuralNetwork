#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>

class GaussRandom {
public:
	GaussRandom(int seed = 0);
	double Next(double stdDev, double mean = 0) const;
};

GaussRandom::GaussRandom(int seed) {
	srand(seed);
}

double GaussRandom::Next(double stdDev, double mean) const {
	double u1 = 1 - rand() / (RAND_MAX + 1.0);
	double u2 = 1 - rand() / (RAND_MAX + 1.0);
	double randStdNormal = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);

	return mean + stdDev * randStdNormal;
}
#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>

#include "NetworkLayer.hpp"

class GaussNoiseLayer : public NetworkLayer {
	double stddev;
	int total;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution;

public:
	GaussNoiseLayer(VolumeSize size, double stddev);

	void ForwardOutput(const std::vector<Volume> &X); // прямое распространение
	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

GaussNoiseLayer::GaussNoiseLayer(VolumeSize size, double stddev) : NetworkLayer(size), distribution(0, stddev) {
	this->stddev = stddev;
	this->total = size.width * size.height * size.deep;

	name = "gauss noise";
	info = "stddev: " + std::to_string(stddev);
}

// прямое распространение
void GaussNoiseLayer::ForwardOutput(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			output[batchIndex][i] = X[batchIndex][i];
			dX[batchIndex][i] = 1;
		}
	}
}

// прямое распространение
void GaussNoiseLayer::Forward(const std::vector<Volume> &X) {
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			output[batchIndex][i] = X[batchIndex][i] + distribution(generator);
			dX[batchIndex][i] = 1;
		}
	}
}

// обратное распространение
void GaussNoiseLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void GaussNoiseLayer::Save(std::ofstream &f) const {
	f << "gaussnoise " << inputSize << " " << stddev << std::endl;
}
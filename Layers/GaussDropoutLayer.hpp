#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>

#include "NetworkLayer.hpp"

class GaussDropoutLayer : public NetworkLayer {
	double p;
	double stddev;
	int total;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution;

public:
	GaussDropoutLayer(VolumeSize size, double p);

	void ForwardOutput(const std::vector<Volume> &X); // прямое распространение
	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

GaussDropoutLayer::GaussDropoutLayer(VolumeSize size, double p) : NetworkLayer(size), distribution(1, sqrt(p / (1 - p))) {
	this->p = p;
	this->stddev = sqrt(p / (1 - p));
	this->total = size.width * size.height * size.deep;

	name = "gauss dropout";
	info = "p: " + std::to_string(p) + ", stddev: " + std::to_string(stddev);
}

// прямое распространение
void GaussDropoutLayer::ForwardOutput(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			output[batchIndex][i] = X[batchIndex][i];
			dX[batchIndex][i] = 1;
		}
	}
}

// прямое распространение
void GaussDropoutLayer::Forward(const std::vector<Volume> &X) {
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			double noise = distribution(generator);

			output[batchIndex][i] = X[batchIndex][i] * noise;
			dX[batchIndex][i] = noise;
		}
	}
}

// обратное распространение
void GaussDropoutLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void GaussDropoutLayer::Save(std::ofstream &f) const {
	f << "gaussdropout " << inputSize << " " << p << std::endl;
}
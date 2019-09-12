#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class LeakyReLULayer : public NetworkLayer {
	int total;
	double alpha;

public:
	LeakyReLULayer(VolumeSize size, double alpha);

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

LeakyReLULayer::LeakyReLULayer(VolumeSize size, double alpha) : NetworkLayer(size) {
	total = size.width * size.height * size.deep;
	this->alpha = alpha;

	name = "leaky relu";
	info = "alpha: " + std::to_string(alpha);
}

// прямое распространение
void LeakyReLULayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			if (X[batchIndex][i] > 0) {
				output[batchIndex][i] = X[batchIndex][i];
				dX[batchIndex][i] = 1;
			}
			else {
				output[batchIndex][i] = alpha * X[batchIndex][i];
				dX[batchIndex][i] = alpha;
			}
		}
	}
}

// обратное распространение
void LeakyReLULayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void LeakyReLULayer::Save(std::ofstream &f) const {
	f << "leakyrelu " << inputSize << " " << alpha << std::endl;
}
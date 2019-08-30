#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class ELULayer : public NetworkLayer {
	int total;
	double alpha;

public:
	ELULayer(VolumeSize size, double alpha);

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

ELULayer::ELULayer(VolumeSize size, double alpha) : NetworkLayer(size) {
	this->alpha = alpha;
	total = size.width * size.height * size.deep;

	name = "elu";
	info = "alpha: " + std::to_string(alpha);
}

// прямое распространение
void ELULayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			if (X[batchIndex][i] > 0) {
				output[batchIndex][i] = X[batchIndex][i];
				dX[batchIndex][i] = 1;
			}
			else {
				output[batchIndex][i] = alpha * (exp(X[batchIndex][i]) - 1);
				dX[batchIndex][i] = alpha * exp(X[batchIndex][i]);
			}
		}
	}
}

// обратное распространение
void ELULayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void ELULayer::Save(std::ofstream &f) const {
	f << "elu " << inputSize << " " << alpha << std::endl;
}
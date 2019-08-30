#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class ReLULayer : public NetworkLayer {
	int total;

public:
	ReLULayer(VolumeSize size);

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

ReLULayer::ReLULayer(VolumeSize size) : NetworkLayer(size) {
	total = size.width * size.height * size.deep;

	name = "relu";
	info = "";
}

// прямое распространение
void ReLULayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			if (X[batchIndex][i] > 0) {
				output[batchIndex][i] = X[batchIndex][i];
				dX[batchIndex][i] = 1;
			}
			else {
				output[batchIndex][i] = 0;
				dX[batchIndex][i] = 0;
			}
		}
	}
}

// обратное распространение
void ReLULayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void ReLULayer::Save(std::ofstream &f) const {
	f << "relu " << inputSize << std::endl;
}
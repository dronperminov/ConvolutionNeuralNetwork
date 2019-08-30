#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class LogSigmoidLayer : public NetworkLayer {
	int total;

public:
	LogSigmoidLayer(VolumeSize size);

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

LogSigmoidLayer::LogSigmoidLayer(VolumeSize size) : NetworkLayer(size) {
	total = size.width * size.height * size.deep;

	name = "logsigmoid";
	info = "";
}

// прямое распространение
void LogSigmoidLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			output[batchIndex][i] = log(1.0 / (1 + exp(-X[batchIndex][i])));
			dX[batchIndex][i] = 1.0 / (exp(X[batchIndex][i]) + 1);
		}
	}
}

// обратное распространение
void LogSigmoidLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void LogSigmoidLayer::Save(std::ofstream &f) const {
	f << "logsigmoid " << inputSize << std::endl;
}
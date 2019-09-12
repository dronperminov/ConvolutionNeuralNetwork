#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class ReshapeLayer : public NetworkLayer {
	int total;

public:
	ReshapeLayer(VolumeSize size, int width, int height, int deep);

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

ReshapeLayer::ReshapeLayer(VolumeSize size, int width, int height, int deep) : NetworkLayer(size, width, height, deep) {
	if (size.width * size.height * size.deep != width * height * deep)
		throw std::runtime_error("Unable to reshape");

	total = size.width * size.height * size.deep;

	name = "reshape";
	info = "";
}

// прямое распространение
void ReshapeLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			output[batchIndex][i] = X[batchIndex][i];
}

// обратное распространение
void ReshapeLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] = dout[batchIndex][i];
}

// сохранение слоя в файл
void ReshapeLayer::Save(std::ofstream &f) const {
	f << "reshape " << inputSize << std::endl;
}
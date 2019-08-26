#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class SoftsignLayer : public NetworkLayer {
	int total;

public:
	SoftsignLayer(VolumeSize size);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

SoftsignLayer::SoftsignLayer(VolumeSize size) : NetworkLayer(size) {
	total = size.width * size.height * size.deep;

	name = "softsign";
	info = "";
}

// получение количества обучаемых параметров
int SoftsignLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void SoftsignLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			output[batchIndex][i] = X[batchIndex][i] / (1 + fabs(X[batchIndex][i]));
			dX[batchIndex][i] = 1.0 / pow(1 + fabs(X[batchIndex][i]), 2);
		}
	}
}

// обратное распространение
void SoftsignLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void SoftsignLayer::Save(std::ofstream &f) const {
	f << "softsign " << inputSize << std::endl;
}
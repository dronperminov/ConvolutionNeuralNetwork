#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class TanhLayer : public NetworkLayer {
	int total;

public:
	TanhLayer(VolumeSize size);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

TanhLayer::TanhLayer(VolumeSize size) : NetworkLayer(size) {
	total = size.width * size.height * size.deep;

	name = "tanh";
	info = "";
}

// получение количества обучаемых параметров
int TanhLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void TanhLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			double value = tanh(X[batchIndex][i]);

			output[batchIndex][i] = value;
			dX[batchIndex][i] = 1 - value * value;
		}
	}
}

// обратное распространение
void TanhLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void TanhLayer::Save(std::ofstream &f) const {
	f << "tanh " << inputSize << std::endl;
}
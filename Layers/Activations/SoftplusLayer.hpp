#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class SoftplusLayer : public NetworkLayer {
	int total;

public:
	SoftplusLayer(int width, int height, int deep);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

SoftplusLayer::SoftplusLayer(int width, int height, int deep) : NetworkLayer(width, height, deep, width, height, deep) {
	total = width * height * deep;
}

// вывод конфигурации
void SoftplusLayer::PrintConfig() const {
	std::cout << "| softplus       | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << "           0 | ";
	std::cout << std::endl;
}

// получение количества обучаемых параметров
int SoftplusLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void SoftplusLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			output[batchIndex][i] = log(1 + exp(X[batchIndex][i]));
			dX[batchIndex][i] = 1.0 / (1 + exp(-X[batchIndex][i]));
		}
	}
}

// обратное распространение
void SoftplusLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void SoftplusLayer::Save(std::ofstream &f) const {
	f << "softplus " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << std::endl;
}
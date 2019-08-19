#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class SwishLayer : public NetworkLayer {
	int total;

public:
	SwishLayer(int width, int height, int deep);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

SwishLayer::SwishLayer(int width, int height, int deep) : NetworkLayer(width, height, deep, width, height, deep) {
	total = width * height * deep;
}

// вывод конфигурации
void SwishLayer::PrintConfig() const {
	std::cout << "| swish          | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << "           0 | ";
	std::cout << std::endl;
}

// получение количества обучаемых параметров
int SwishLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void SwishLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			double sigmoid = 1.0 / (1 + exp(-X[batchIndex][i]));
			double value = X[batchIndex][i] * sigmoid;

			output[batchIndex][i] = value;
			dX[batchIndex][i] = value + sigmoid * (1 - value);
		}
	}
}

// обратное распространение
void SwishLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void SwishLayer::Save(std::ofstream &f) const {
	f << "swish " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << std::endl;
}
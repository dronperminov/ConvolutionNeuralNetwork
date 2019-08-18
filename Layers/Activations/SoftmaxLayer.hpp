#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class SoftmaxLayer : public NetworkLayer {
	int total;

public:
	SoftmaxLayer(int width, int height, int deep);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

SoftmaxLayer::SoftmaxLayer(int width, int height, int deep) : NetworkLayer(width, height, deep, width, height, deep) {
	total = width * height * deep;
}

// вывод конфигурации
void SoftmaxLayer::PrintConfig() const {
	std::cout << "| softmax        | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << "           0 | ";
	std::cout << std::endl;
}

// получение количества обучаемых параметров
int SoftmaxLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void SoftmaxLayer::Forward(const std::vector<Volume> &X) {
	output = std::vector<Volume>(X.size(), Volume(outputSize));

	#pragma omp parallel for
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		double sum = 0;

		for (int i = 0; i < total; i++) {
			output[batchIndex][i] = exp(X[batchIndex][i]);
			sum += output[batchIndex][i];
		}

		for (int i = 0; i < total; i++)
			output[batchIndex][i] /= sum;
	}
}

// обратное распространение
void SoftmaxLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	dX = std::vector<Volume>(dout.size(), Volume(inputSize));

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			double sum = 0;

			for (int j = 0; j < total; j++)
				sum += dout[batchIndex][j] * output[batchIndex][i] * ((i == j) - output[batchIndex][j]);

			dX[batchIndex][i] = sum;
		}
	}
}

// сохранение слоя в файл
void SoftmaxLayer::Save(std::ofstream &f) const {
	f << "softmax " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << std::endl;
}
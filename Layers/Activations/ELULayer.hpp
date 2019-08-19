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
	ELULayer(int width, int height, int deep, double alpha);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

ELULayer::ELULayer(int width, int height, int deep, double alpha) : NetworkLayer(width, height, deep, width, height, deep) {
	this->alpha = alpha;
	total = width * height * deep;
}

// вывод конфигурации
void ELULayer::PrintConfig() const {
	std::cout << "| elu            | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << "           0 | ";
	std::cout << "alpha: " << alpha;
	std::cout << std::endl;
}

// получение количества обучаемых параметров
int ELULayer::GetTrainableParams() const {
	return 0;
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
	f << "elu " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " " << alpha << std::endl;
}
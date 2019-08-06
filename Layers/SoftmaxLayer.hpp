#pragma once

#include <iostream>
#include <iomanip>
#include "NetworkLayer.hpp"

class SoftmaxLayer : public NetworkLayer {
	int total;

public:
	SoftmaxLayer(int width, int height, int deep);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количество обучаемых параметров

	void Forward(const Volume& input); // прямое распространение
	void Backward(Volume& prevDeltas); // обратное распространение
	void Save(std::ostream &f); // сохранение слоя в файл
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

// получение количество обучаемых параметров
int SoftmaxLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void SoftmaxLayer::Forward(const Volume& input) {
	double sum = 0;

	for (int i = 0; i < total; i++) {
		output[i] = exp(input[i]);
		sum += output[i];
	}

	#pragma omp parallel for
	for (int i = 0; i < total; i++) {
		output[i] /= sum;
		deltas[i] = 1;
	}
}

// обратное распространение
void SoftmaxLayer::Backward(Volume& prevDeltas) {
	#pragma omp parallel for
	for (int i = 0; i < total; i++) {
		double sum = 0;

		for (int j = 0; j < total; j++)
			sum += deltas[j] * output[i] * ((i == j) - output[j]);

		prevDeltas[i] *= sum;
	}
}

// сохранение слоя в файл
void SoftmaxLayer::Save(std::ostream &f) {
	f << "softmax " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << std::endl;
}
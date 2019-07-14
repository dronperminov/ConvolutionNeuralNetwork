#pragma once

#include <iostream>
#include <iomanip>
#include <random>
#include "NetworkLayer.hpp"

class DropoutLayer : public NetworkLayer {
	double p; // вероятность исключения нейронов

	std::default_random_engine generator;
	std::binomial_distribution<int> distribution;

public:
	DropoutLayer(int width, int height, int deep, double p);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количество обучаемых параметров

	void ForwardOutput(const Volume& input); // прямое распространение
	void Forward(const Volume& input); // прямое распространение
	void Backward(Volume& prevDeltas); // обратное распространение
	void Save(std::ostream &f); // сохранение слоя в файл
};

DropoutLayer::DropoutLayer(int width, int height, int deep, double p) : NetworkLayer(width, height, deep, width, height, deep), distribution(1, p) {
	this->p = p;
}

// вывод конфигурации
void DropoutLayer::PrintConfig() const {
	std::cout << "|      dropout layer       | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << "           0 | ";
	std::cout << "p:" << p;
	std::cout << std::endl;
}

// получение количество обучаемых параметров
int DropoutLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void DropoutLayer::ForwardOutput(const Volume& input) {
	//#pragma omp parallel for collapse(3)
	for (int d = 0; d < inputSize.deep; d++) {
		for (int i = 0; i < inputSize.height; i++) {
			for (int j = 0; j < inputSize.width; j++) {
				output(d, i, j) = input(d, i, j);
				deltas(d, i, j) = 1;
			}
		}
	}
}

// прямое распространение
void DropoutLayer::Forward(const Volume& input) {
	#pragma omp parallel for collapse(3)
	for (int d = 0; d < inputSize.deep; d++) {
		for (int i = 0; i < inputSize.height; i++) {
			for (int j = 0; j < inputSize.width; j++) {
				if (distribution(generator)) {
					output(d, i, j) = input(d, i, j) / p;
					deltas(d, i, j) = 1;
				}
				else {
					output(d, i, j) = 0;
					deltas(d, i, j) = 0;
				}
			}
		}
	}
}

// обратное распространение
void DropoutLayer::Backward(Volume& prevDeltas) {
	//#pragma omp parallel for collapse(3)
	for (int d = 0; d < inputSize.deep; d++)
		for (int i = 0; i < inputSize.height; i++)
			for (int j = 0; j < inputSize.width; j++)
				prevDeltas(d, i, j) *= deltas(d, i, j);
}

// сохранение слоя в файл
void DropoutLayer::Save(std::ostream &f) {
	f << "dropout " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " " << p << std::endl;
}
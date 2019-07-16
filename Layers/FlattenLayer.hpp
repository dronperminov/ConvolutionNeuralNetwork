#pragma once

#include <iostream>
#include <iomanip>
#include "NetworkLayer.hpp"

class FlattenLayer : public NetworkLayer {
public:
	FlattenLayer(int inputWidth, int inputHeight, int inputDeep);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количество обучаемых параметров

	void Forward(const Volume& input); // прямое распространение
	void Backward(Volume& prevDeltas); // обратное распространение
	void Save(std::ostream &f); // сохранение слоя в файл
};

FlattenLayer::FlattenLayer(int width, int height, int deep) : NetworkLayer(width, height, deep, 1, 1, width * height * deep) {

}

// вывод конфигурации
void FlattenLayer::PrintConfig() const {
	std::cout << "|      flatten layer       | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << "           0 | ";
	std::cout << std::endl;
}

// получение количество обучаемых параметров
int FlattenLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void FlattenLayer::Forward(const Volume& input) {
	int index = 0;

	for (int i = 0; i < inputSize.height; i++) {
		for (int j = 0; j < inputSize.width; j++) {
			for (int d = 0; d < inputSize.deep; d++) {
				deltas(index, 0, 0) = 1;
				output(index++, 0, 0) = input(d, i, j);
			}
		}
	}
}

// обратное распространение
void FlattenLayer::Backward(Volume& prevDeltas) {
	int index = 0;

	for (int i = 0; i < inputSize.height; i++)
		for (int j = 0; j < inputSize.width; j++)
			for (int d = 0; d < inputSize.deep; d++)
				prevDeltas(d, i, j) *= deltas(index++, 0, 0);
}

// сохранение слоя в файл
void FlattenLayer::Save(std::ostream &f) {
	f << "flatten " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << std::endl;
}
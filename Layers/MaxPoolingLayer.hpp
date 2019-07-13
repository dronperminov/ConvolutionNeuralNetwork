#pragma once

#include <iostream>
#include <iomanip>
#include "NetworkLayer.hpp"

class MaxPoolingLayer : public NetworkLayer {
	int scale; // коэффициент пулинга
	Volume maxIndexes; // маска максимальных индексов

public:
	MaxPoolingLayer(int width, int height, int deep, int scale = 2);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количество обучаемых параметров

	void Forward(const Volume& input); // прямое распространение
	void Backward(Volume& prevDeltas); // обратное распространение
	void Save(std::ostream &f); // сохранение слоя в файл
};

MaxPoolingLayer::MaxPoolingLayer(int width, int height, int deep, int scale) :
	NetworkLayer(width, height, deep, width / scale, height / scale, deep),
	maxIndexes(width, height, deep)
{
	if (width % scale != 0 || height % scale != 0)
		throw std::runtime_error("Unable creating maxpool layer with this scale");

	this->scale = scale;
}

// вывод конфигурации
void MaxPoolingLayer::PrintConfig() const {
	std::cout << "|     max pooling layer    | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << "           0 | ";
	std::cout << "scale: " << scale << std::endl; 
}

// получение количество обучаемых параметров
int MaxPoolingLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void MaxPoolingLayer::Forward(const Volume& input) {
	#pragma omp parallel for
	for (int d = 0; d < inputSize.deep; d++) {
		for (int i = 0, i1 = 0; i < inputSize.height; i += scale, i1++) {
			for (int j = 0, j1 = 0; j < inputSize.width; j += scale, j1++) {
				int imax = i;
				int jmax = j;
				double max = input(d, i, j);

				int dx = j + scale;
				int dy = i + scale;

				for (int x = j; x < dx; x++) {
					for (int y = i; y < dy; y++) {
						maxIndexes(d, y, x) = 0;
						this->input(d, y, x) = input(d, y, x);

						if (input(d, y, x) > max) {
							max = input(d, y, x);
							imax = y;
							jmax = x;
						}
					}
				}

				output(d, i1, j1) = max;
				deltas(d, i1, j1) = 1;
				maxIndexes(d, imax, jmax) = 1;
			}
		}
	}
}

// обратное распространение
void MaxPoolingLayer::Backward(Volume& prevDeltas) {
	#pragma omp parallel for collapse(3)
	for (int d = 0; d < inputSize.deep; d++)
		for (int i = 0; i < inputSize.height; i++)
			for (int j = 0; j < inputSize.width; j++)
				prevDeltas(d, i, j) *= maxIndexes(d, i, j) * deltas(d, i / scale, j / scale);
}

// сохранение слоя в файл
void MaxPoolingLayer::Save(std::ostream &f) {
	f << "maxpool " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " " << scale << std::endl;
}
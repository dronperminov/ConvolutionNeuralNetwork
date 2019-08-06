#pragma once

#include <iostream>
#include <iomanip>
#include "NetworkLayer.hpp"

class MaxPoolingLayer : public NetworkLayer {
	int scale; // коэффициент пулинга
	Volume maxIndexes; // маска максимальных индексов
	std::vector<int> di;
	std::vector<int> dj;

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
	maxIndexes(width, height, deep), di(height), dj(width)
{
	if (width % scale != 0 || height % scale != 0)
		throw std::runtime_error("Unable creating maxpool layer with this scale");

	this->scale = scale;

	for (int i = 0; i < height; i++)
		di[i] = i / scale;

	for (int i = 0; i < width; i++)
		dj[i] = i / scale;
}

// вывод конфигурации
void MaxPoolingLayer::PrintConfig() const {
	std::cout << "| max pooling    | ";
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
	#pragma omp parallel for collapse(3)
	for (int d = 0; d < inputSize.deep; d++) {
		for (int i = 0; i < inputSize.height; i += scale) {
			for (int j = 0; j < inputSize.width; j += scale) {
				int imax = i;
				int jmax = j;
				double max = input(d, i, j);

				int dx = j + scale;
				int dy = i + scale;

				for (int y = i; y < dy; y++) {
					for (int x = j; x < dx; x++) {
						double value = input(d, y, x);
						maxIndexes(d, y, x) = 0;

						if (value > max) {
							max = value;
							imax = y;
							jmax = x;
						}
					}
				}

				int i1 = di[i];
				int j1 = dj[j];

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
				prevDeltas(d, i, j) *= maxIndexes(d, i, j) * deltas(d, di[i], dj[j]);
}

// сохранение слоя в файл
void MaxPoolingLayer::Save(std::ostream &f) {
	f << "maxpool " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " " << scale << std::endl;
}
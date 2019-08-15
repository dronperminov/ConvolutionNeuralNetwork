#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class MaxPoolingLayer : public NetworkLayer {
	int scale;

	std::vector<int> di;
	std::vector<int> dj;

public:
	MaxPoolingLayer(int width, int height, int deep, int scale = 2);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количество обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

MaxPoolingLayer::MaxPoolingLayer(int width, int height, int deep, int scale) : NetworkLayer(width, height, deep, width / scale, height / scale, deep) {
	if (width % scale != 0 || height % scale != 0)
		throw std::runtime_error("Unable creating maxpool layer with this scale");

	this->scale = scale;

	di = std::vector<int>(height);
	dj = std::vector<int>(width);

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
void MaxPoolingLayer::Forward(const std::vector<Volume> &X) {
	output = std::vector<Volume>(X.size(), Volume(outputSize));
	dX = std::vector<Volume>(X.size(), Volume(inputSize));

	#pragma omp parallel for collapse(4)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int d = 0; d < inputSize.deep; d++) {
			for (int i = 0; i < inputSize.height; i += scale) {
				for (int j = 0; j < inputSize.width; j += scale) {
					int imax = i;
					int jmax = j;
					double max = X[batchIndex](d, i, j);

					for (int y = i; y < i + scale; y++) {
						for (int x = j; x < j + scale; x++) {
							double value = X[batchIndex](d, y, x);
							dX[batchIndex](d, y, x) = 0;

							if (value > max) {
								max = value;
								imax = y;
								jmax = x;
							}
						}
					}

					output[batchIndex](d, di[i], dj[j]) = max;
					dX[batchIndex](d, imax, jmax) = 1;
				}
			}
		}
	}
}

// обратное распространение
void MaxPoolingLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(4)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int d = 0; d < inputSize.deep; d++)
			for (int i = 0; i < inputSize.height; i++)
				for (int j = 0; j < inputSize.width; j++)
					dX[batchIndex](d, i, j) *= dout[batchIndex](d, di[i], dj[j]);
}

// сохранение слоя в файл
void MaxPoolingLayer::Save(std::ofstream &f) const {
	f << "maxpool " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " " << scale << std::endl;
}
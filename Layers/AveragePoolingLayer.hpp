#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class AveragePoolingLayer : public NetworkLayer {
	int scale;

	std::vector<int> di;
	std::vector<int> dj;

public:
	AveragePoolingLayer(VolumeSize size, int scale = 2);

	int GetTrainableParams() const; // получение количество обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

AveragePoolingLayer::AveragePoolingLayer(VolumeSize size, int scale) : NetworkLayer(size, size.width / scale, size.height / scale, size.deep) {
	if (size.width % scale != 0 || size.height % scale != 0)
		throw std::runtime_error("Unable creating maxpool layer with this scale");

	name = "avg pooling";
	info = "scale: " + std::to_string(scale);

	this->scale = scale;

	di = std::vector<int>(size.height);
	dj = std::vector<int>(size.width);

	for (int i = 0; i < size.height; i++)
		di[i] = i / scale;

	for (int i = 0; i < size.width; i++)
		dj[i] = i / scale;
}

// получение количество обучаемых параметров
int AveragePoolingLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void AveragePoolingLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(4)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int d = 0; d < inputSize.deep; d++) {
			for (int i = 0; i < inputSize.height; i += scale) {
				for (int j = 0; j < inputSize.width; j += scale) {
					double sum = 0;

					for (int y = i; y < i + scale; y++)
						for (int x = j; x < j + scale; x++)
							sum += X[batchIndex](d, y, x);

					output[batchIndex](d, di[i], dj[j]) = sum / (scale * scale);
				}
			}
		}
	}
}

// обратное распространение
void AveragePoolingLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(4)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int d = 0; d < inputSize.deep; d++)
			for (int i = 0; i < inputSize.height; i++)
				for (int j = 0; j < inputSize.width; j++)
					dX[batchIndex](d, i, j) = dout[batchIndex](d, di[i], dj[j]) / (scale * scale);
}

// сохранение слоя в файл
void AveragePoolingLayer::Save(std::ofstream &f) const {
	f << "avgpool " << inputSize << " " << scale << std::endl;
}
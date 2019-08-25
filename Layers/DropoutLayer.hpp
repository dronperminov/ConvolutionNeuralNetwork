#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>

#include "NetworkLayer.hpp"

class DropoutLayer : public NetworkLayer {
	double p;
	double q;
	int total;

	std::default_random_engine generator;
	std::binomial_distribution<int> distribution;

public:
	DropoutLayer(VolumeSize size, double p);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void ForwardOutput(const std::vector<Volume> &X); // прямое распространение
	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

DropoutLayer::DropoutLayer(VolumeSize size, double p) : NetworkLayer(size.width, size.height, size.deep, size.width, size.height, size.deep), distribution(1, 1 - p) {
	this->p = p;
	this->q = 1 - p;
	this->total = size.width * size.height * size.deep;

	name = "dropout";
	info = "p: " + std::to_string(p);
}

// получение количество обучаемых параметров
int DropoutLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение
void DropoutLayer::ForwardOutput(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			output[batchIndex][i] = X[batchIndex][i];
			dX[batchIndex][i] = 1;
		}
	}
}

// прямое распространение
void DropoutLayer::Forward(const std::vector<Volume> &X) {
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			if (distribution(generator)) { 
				output[batchIndex][i] = X[batchIndex][i] / q;
				dX[batchIndex][i] = 1;
			}
			else {
				output[batchIndex][i] = 0;
				dX[batchIndex][i] = 0;
			}
		}
	}
}

// обратное распространение
void DropoutLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// сохранение слоя в файл
void DropoutLayer::Save(std::ofstream &f) const {
	f << "dropout " << inputSize << " " << p << std::endl;
}
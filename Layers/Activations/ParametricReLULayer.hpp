#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../NetworkLayer.hpp"

class ParametricReLULayer : public NetworkLayer {
	int total;
	Volume alpha;
	Volume dalpha;
	std::vector<Volume> paramsalpha;

	void InitParams(); // инициализация параметров для обучения
	void InitWeights(); // инициализация весовых коэффициентов
	void LoadWeights(std::ifstream &f); // считывание весовых коэффициентов из файла

public:
	ParametricReLULayer(VolumeSize size);
	ParametricReLULayer(VolumeSize size, std::ifstream &f);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение
	void UpdateWeights(const Optimizer &optimizer); // обновление весовых коэффициентов

	void ResetCache(); // сброс параметров
	void Save(std::ofstream &f) const; // сохранение слоя в файл

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index) const; // получение градиента веса по индексу
	void ZeroGradient(int index); // обнуление градиента веса по индексу
};

ParametricReLULayer::ParametricReLULayer(VolumeSize size) : NetworkLayer(size.width, size.height, size.deep, size.width, size.height, size.deep), alpha(1, 1, size.width * size.height * size.deep), dalpha(1, 1, size.width * size.height * size.deep) {
	total = size.width * size.height * size.deep;

	InitParams();
	InitWeights();

	name = "param. relu";
	info = "";
}

ParametricReLULayer::ParametricReLULayer(VolumeSize size, std::ifstream &f) : NetworkLayer(size.width, size.height, size.deep, size.width, size.height, size.deep), alpha(1, 1, size.width * size.height * size.deep), dalpha(1, 1, size.width * size.height * size.deep) {
	total = size.width * size.height * size.deep;

	InitParams();
	LoadWeights(f);

	name = "param. relu";
	info = "";
}

// инициализация параметров для обучения
void ParametricReLULayer::InitParams() {
	for (int i = 0; i < OPTIMIZER_PARAMS_COUNT; i++)
		paramsalpha.push_back(Volume(1, 1, total));
}

// инициализация весовых коэффициентов
void ParametricReLULayer::InitWeights() {
	for (int i = 0; i < total; i++)
		alpha[i] = random.Next(0.01, 0);
}

// считывание весовых коэффициентов из файла
void ParametricReLULayer::LoadWeights(std::ifstream &f) {
	for (int i = 0; i < total; i++)
		f >> alpha[i];
}

// получение количества обучаемых параметров
int ParametricReLULayer::GetTrainableParams() const {
	return total;
}

// прямое распространение
void ParametricReLULayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			if (X[batchIndex][i] > 0) {
				output[batchIndex][i] = X[batchIndex][i];
				dX[batchIndex][i] = 1;
			}
			else {
				output[batchIndex][i] = alpha[i] * X[batchIndex][i];
				dX[batchIndex][i] = alpha[i];
			}
		}
	}
}

// обратное распространение
void ParametricReLULayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			if (X[batchIndex][i] <= 0)
				dalpha[i] += dout[batchIndex][i] * X[batchIndex][i];

	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] *= dout[batchIndex][i];
}

// обновление весовых коэффициентов
void ParametricReLULayer::UpdateWeights(const Optimizer &optimizer) {
	int batchSize = output.size();

	for (int i = 0; i < total; i++) {
		optimizer.Update(dalpha[i] / batchSize, paramsalpha[0][i], paramsalpha[1][i], paramsalpha[2][i], alpha[i]);
		dalpha[i] = 0;
	}
}

// сброс параметров
void ParametricReLULayer::ResetCache() {
	for (int index = 0; index < OPTIMIZER_PARAMS_COUNT; index++)
		for (int i = 0; i < total; i++)
			paramsalpha[index][i] = 0;
}

// сохранение слоя в файл
void ParametricReLULayer::Save(std::ofstream &f) const {
	f << "prelu " << inputSize << std::endl;

	for (int i = 0; i < total; i++)
		f << std::setprecision(15) << alpha[i] << " ";

	f << std::endl;
}

// установка веса по индексу
void ParametricReLULayer::SetParam(int index, double weight) {
	alpha[index] = weight;
}

// получение веса по индексу
double ParametricReLULayer::GetParam(int index) const {
	return alpha[index];
}

// получение градиента веса по индексу
double ParametricReLULayer::GetGradient(int index) const {
	return dalpha[index];
}

// обнуление градиента веса по индексу
void ParametricReLULayer::ZeroGradient(int index) {
	dalpha[index] = 0;
}
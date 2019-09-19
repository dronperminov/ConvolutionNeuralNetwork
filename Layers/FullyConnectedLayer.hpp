#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <random>

#include "NetworkLayer.hpp"
#include "../Entities/Matrix.hpp"

class FullyConnectedLayer : public NetworkLayer {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution;

	// тип активационной функции
	enum class ActivationType {
		None,
		Sigmoid,
		Tanh,
		ReLU,
		LeakyReLU,
		ELU
	};

	int inputs;
	int outputs;

	ActivationType activationType; // тип активационной функции

	Matrix W; // матрица весовых коэффициентов
	Matrix dW;
	std::vector<Matrix> paramsW;

	std::vector<double> b; // смещения
	std::vector<double> db;
	std::vector<std::vector<double>> paramsb;

	std::vector<Volume> df;

	void InitParams(); // инициализация параметров для обучения
	void InitWeights(); // инициализация весовых коэффициентов
	void LoadWeights(std::ifstream &f); // считывание весовых коэффициентов из файла

	ActivationType GetActivationType(const std::string& type) const; // получение типа активационной функции по строке
	std::string GetActivationType() const; // получение строки для активационной функции
	void Activate(int batchIndex, int i, double value); // применение активационной функции

public:
	FullyConnectedLayer(VolumeSize size, int outputs, const std::string& type = "none");
	FullyConnectedLayer(VolumeSize size, int outputs, const std::string& type, std::ifstream &f);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение
	void UpdateWeights(const Optimizer &optimizer, bool trainable); // обновление весовых коэффициентов

	void ResetCache(); // сброс параметров
	void Save(std::ofstream &f) const; // сохранение слоя в файл
	void SetBatchSize(int batchSize); // установка размера батча

	void SetWeight(int i, int j, double weight);
	void SetBias(int i, double bias);

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index) const; // получение градиента веса по индексу
	void ZeroGradient(int index); // обнуление градиента веса по индексу
};

FullyConnectedLayer::FullyConnectedLayer(VolumeSize size, int outputs, const std::string& type) : NetworkLayer(size, 1, 1, outputs), W(outputs, size.height * size.width * size.deep), dW(outputs, size.height * size.width * size.deep), b(outputs), db(outputs), distribution(0.0, sqrt(2.0 / (size.height * size.width * size.deep))) {
	this->inputs = size.height * size.width * size.deep;
	this->outputs = outputs;

	activationType = GetActivationType(type);

	name = "fc";
	info = std::to_string(outputs) + " neurons";

	if (activationType != ActivationType::None)
		info += ", f: " + GetActivationType();

	InitParams();
	InitWeights();
}

FullyConnectedLayer::FullyConnectedLayer(VolumeSize size, int outputs, const std::string& type, std::ifstream &f) : NetworkLayer(size, 1, 1, outputs), W(outputs, size.height * size.width * size.deep), dW(outputs, size.height * size.width * size.deep), b(outputs), db(outputs), distribution(0.0, sqrt(2.0 / (size.height * size.width * size.deep))) {
	this->inputs = size.height * size.width * size.deep;
	this->outputs = outputs;

	activationType = GetActivationType(type);

	name = "fc";
	info = std::to_string(outputs) + " neurons";

	if (activationType != ActivationType::None)
		info += ", f: " + GetActivationType();
	
	InitParams();
	LoadWeights(f);
}

// инициализация параметров для обучения
void FullyConnectedLayer::InitParams() {
	for (int i = 0; i < OPTIMIZER_PARAMS_COUNT; i++) {
		paramsW.push_back(Matrix(outputs, inputs));
		paramsb.push_back(std::vector<double>(outputs));
	}
}

// инициализация весовых коэффициентов
void FullyConnectedLayer::InitWeights() {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			W(i, j) = distribution(generator);

		b[i] = 0.01;
	}
}

// считывание весовых коэффициентов из файла
void FullyConnectedLayer::LoadWeights(std::ifstream &f) {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			f >> W(i, j);

		f >> b[i];
	}
}

// получение типа активационной функции по строке
FullyConnectedLayer::ActivationType FullyConnectedLayer::GetActivationType(const std::string& type) const {
	if (type == "sigmoid")
		return ActivationType::Sigmoid;

	if (type == "tanh")
		return ActivationType::Tanh;

	if (type == "relu")
		return ActivationType::ReLU;

	if (type == "leakyrelu")
		return ActivationType::LeakyReLU;

	if (type == "elu")
		return ActivationType::ELU;

	if (type == "none" || type == "")
		return ActivationType::None;

	throw std::runtime_error("Invalid activation function");
}

// получение строки для активационной функции
std::string FullyConnectedLayer::GetActivationType() const {
	if (activationType == ActivationType::Sigmoid)
		return "sigmoid";

	if (activationType == ActivationType::Tanh)
		return "tanh";

	if (activationType == ActivationType::ReLU)
		return "relu";

	if (activationType == ActivationType::LeakyReLU)
		return "leakyrelu";

	if (activationType == ActivationType::ELU)
		return "elu";

	if (activationType == ActivationType::None)
		return "none";

	throw std::runtime_error("Invalid activation function");
}

// применение активационной функции
void FullyConnectedLayer::Activate(int batchIndex, int i, double value) {
	if (activationType == ActivationType::None) {
		output[batchIndex][i] = value;
		df[batchIndex][i] = 1;
	}
	else if (activationType == ActivationType::Sigmoid) {
		value = 1 / (1 + exp(-value));
		output[batchIndex][i] = value;
		df[batchIndex][i] = value * (1 - value);
	}
	else if (activationType == ActivationType::Tanh) {
		value = tanh(value);
		output[batchIndex][i] = value;
		df[batchIndex][i] = 1 - value * value;
	}
	else if (activationType == ActivationType::ReLU) {
		if (value > 0) {
			output[batchIndex][i] = value;
			df[batchIndex][i] = 1;
		}
		else {
			output[batchIndex][i] = 0;
			df[batchIndex][i] = 0;
		}
	}
	else if (activationType == ActivationType::LeakyReLU) {
		if (value > 0) {
			output[batchIndex][i] = value;
			df[batchIndex][i] = 1;
		}
		else {
			output[batchIndex][i] = 0.01 * value;
			df[batchIndex][i] = 0.01;
		}
	}
	else if (activationType == ActivationType::ELU) {
		if (value > 0) {
			output[batchIndex][i] = value;
			df[batchIndex][i] = 1;
		}
		else {
			output[batchIndex][i] = exp(value) - 1;
			df[batchIndex][i] = exp(value);
		}
	}
}

// получение количество обучаемых параметров
int FullyConnectedLayer::GetTrainableParams() const {
	return outputs * (inputs + 1);
}

// прямое распространение
void FullyConnectedLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < outputs; i++) {
			double sum = b[i];

			for (int j = 0; j < inputs; j++)
				sum += W(i, j) * X[batchIndex][j];

			Activate(batchIndex, i, sum);
		}
	}
}

// обратное распространение
void FullyConnectedLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (calc_dX) {
		#pragma omp parallel for collapse(2)
		for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
			for (int j = 0; j < inputs; j++) {
				double sum = 0;

				for (int i = 0; i < outputs; i++)
					sum += W(i, j) * dout[batchIndex][i] * df[batchIndex][i];

				dX[batchIndex][j] = sum;
			}
		}
	}

	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int i = 0; i < outputs; i++) {
			double delta = dout[batchIndex][i] * df[batchIndex][i];

			for (int j = 0; j < inputs; j++)
				dW(i, j) += delta * X[batchIndex][j];

			db[i] += delta;
		}
	}
}

// обновление весовых коэффициентов
void FullyConnectedLayer::UpdateWeights(const Optimizer &optimizer, bool trainable) {
	int batchSize = output.size();

	#pragma omp parallel for
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++) {
			if (trainable)
				optimizer.Update(dW(i, j) / batchSize, paramsW[0](i, j), paramsW[1](i, j), paramsW[2](i, j), W(i, j));

			dW(i, j) = 0;
		}

		if (trainable)
			optimizer.Update(db[i] / batchSize, paramsb[0][i], paramsb[1][i], paramsb[2][i], b[i]);

		db[i] = 0;
	}
}

// сброс параметров
void FullyConnectedLayer::ResetCache() {
	for (int index = 0; index < OPTIMIZER_PARAMS_COUNT; index++) {
		for (int i = 0; i < outputs; i++) {
			for (int j = 0; j < inputs; j++)
				paramsW[index](i, j) = 0;

			paramsb[index][i] = 0;
		}
	}
}

// сохранение слоя в файл
void FullyConnectedLayer::Save(std::ofstream &f) const {
	f << "fc " << inputSize << " " << outputs << " " << GetActivationType() << std::endl;

	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			f << std::setprecision(15) << W(i, j) << " ";

		f << std::setprecision(15) << b[i] << std::endl;
	}
}

// установка размера батча
void FullyConnectedLayer::SetBatchSize(int batchSize) {
	output = std::vector<Volume>(batchSize, Volume(outputSize));
	df = std::vector<Volume>(batchSize, Volume(outputSize));
	dX = std::vector<Volume>(batchSize, Volume(inputSize));
}

void FullyConnectedLayer::SetWeight(int i, int j, double weight) {
	W(i, j) = weight;
}

void FullyConnectedLayer::SetBias(int i, double bias) {
	b[i] = bias;
}

// установка веса по индексу
void FullyConnectedLayer::SetParam(int index, double weight) {
	int i = index / (inputs + 1);
	int j = index % (inputs + 1);

	if (j < inputs) {
		W(i, j) = weight;
	}
	else {
		b[i] = weight;
	}
}

// получение веса по индексу
double FullyConnectedLayer::GetParam(int index) const {
	int i = index / (inputs + 1);
	int j = index % (inputs + 1);

	return j < inputs ? W(i, j) : b[i];
}

// получение градиента веса по индексу
double FullyConnectedLayer::GetGradient(int index) const {
	int i = index / (inputs + 1);
	int j = index % (inputs + 1);

	return j < inputs ? dW(i, j) : db[i];
}

// обнуление градиента веса по индексу
void FullyConnectedLayer::ZeroGradient(int index) {
	int i = index / (inputs + 1);
	int j = index % (inputs + 1);

	if (j < inputs)
		dW(i, j) = 0;
	else
		db[i] = 0;
}
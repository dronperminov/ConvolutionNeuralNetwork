#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

#include "../Entities/Matrix.hpp"
#include "NetworkLayer.hpp"

class FullConnectedLayer : public NetworkLayer {
	// тип активационной функции
	enum class ActivationType {
		None,
		Sigmoid,
		Tanh,
		ReLU
	};

	int inputs; // число входов
	int outputs; // число выходов
	ActivationType activationType; // тип активационной функции

	Matrix weights; // веса
	Matrix dw; // изменение весов
	Matrix dww; // дполнительное изменение весов
	Matrix gradWeights; // градиенты весовых коэффициентов

	std::vector<double> biases; // смещения
	std::vector<double> db; // изменение смещений
	std::vector<double> dbb; // дополнительное изменение смещений
	std::vector<double> gradBiases; // градиенты смещений

	ActivationType GetActivationType(const std::string& type) const; // получение типа активационной функции по строке
	std::string GetActivationType() const; // получение строки для активационной функции

	void ActivateReLU(int i, double value);
	void ActivateSigmoid(int i, double value);
	void ActivateTanh(int i, double value);
	void ActivateNone(int i, double value);

	void (FullConnectedLayer::*Activate)(int, double);
	void SetActivationFunction();

public:
	FullConnectedLayer(int inputs, int outputs, const std::string& type = "sigmoid");
	FullConnectedLayer(int inputs, int outputs, const std::string& type, std::ifstream &f); // загрузка слоя из файла

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количество обучаемых параметров

	void Forward(const Volume& input); // прямое распространение
	void Backward(Volume& prevDeltas); // обратное распространение
	void UpdateWeights(const Optimizer& optimizer, const Volume& input); // обновление весовых коэффициентов
	
	void CalculateGradients(const Volume &input); // вычисление градиентов
	void UpdateWeights(const Optimizer& optimizer, int batchSize); // обновление весовых коэффициентов
	
	void ResetCache(); // сброс параметров
	void Save(std::ostream &f); // сохранение слоя в файл

	void SetWeight(int i, int j, double weight); // установка веса
	void SetBias(int i, double bias); // установка смещения

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index, const Volume &input) const; // получение градиента веса по индексу
};

void FullConnectedLayer::ActivateNone(int i, double value) {
	output[i] = value;
	deltas[i] = 1;
}

void FullConnectedLayer::ActivateReLU(int i, double value) {
	if (value > 0) {
		output[i] = value;
		deltas[i] = 1;
	}
	else {
		output[i] = 0;
		deltas[i] = 0;	
	}
}

void FullConnectedLayer::ActivateSigmoid(int i, double value) {
	value = 1 / (1 + exp(-value));
	output[i] = value;
	deltas[i] = value * (1 - value);
}

void FullConnectedLayer::ActivateTanh(int i, double value) {
	value = tanh(value);
	output[i] = value;
	deltas[i] = 1 - value * value;
}

void FullConnectedLayer::SetActivationFunction() {
	if (activationType == ActivationType::None) {
		Activate = ActivateNone;
	}
	else if (activationType == ActivationType::ReLU) {
		Activate = ActivateReLU;
	}
	else if (activationType == ActivationType::Sigmoid) {
		Activate = ActivateSigmoid;
	}
	else if (activationType == ActivationType::Tanh) {
		Activate = ActivateTanh;
	}
}

// получение типа активационной функции по строке
FullConnectedLayer::ActivationType FullConnectedLayer::GetActivationType(const std::string& type) const {
	if (type == "sigmoid")
		return ActivationType::Sigmoid;
	
	if (type == "tanh")
		return ActivationType::Tanh;
	
	if (type == "relu")
		return ActivationType::ReLU;

	if (type == "none" || type == "")
		return ActivationType::None;

	throw std::runtime_error("Invalid activation function");
}

// получение строки для активационной функции
std::string FullConnectedLayer::GetActivationType() const {
	if (activationType == ActivationType::Sigmoid)
		return "sigmoid";
	
	if (activationType == ActivationType::Tanh)
		return "tanh";
	
	if (activationType == ActivationType::ReLU)
		return "relu";

	if (activationType == ActivationType::None)
		return "none";
	
	throw std::runtime_error("Invalid activation function");
}

FullConnectedLayer::FullConnectedLayer(int inputs, int outputs, const std::string& type) :
	NetworkLayer(1, 1, inputs, 1, 1, outputs),
	weights(outputs, inputs), dw(outputs, inputs), dww(outputs, inputs), gradWeights(outputs, inputs),
	biases(outputs), db(outputs), dbb(outputs), gradBiases(outputs) {
	
	this->inputs = inputs;
	this->outputs = outputs;

	activationType = GetActivationType(type);
	SetActivationFunction();

	weights.FillRandom(random, sqrt(2.0 / inputs));

	for (int i = 0; i < outputs; i++)
		biases[i] = 0.01;
}

// загрузка слоя из файла
FullConnectedLayer::FullConnectedLayer(int inputs, int outputs, const std::string& type, std::ifstream &f) :
	NetworkLayer(1, 1, inputs, 1, 1, outputs),
	weights(outputs, inputs), dw(outputs, inputs), dww(outputs, inputs), gradWeights(outputs, inputs),
	biases(outputs), db(outputs), dbb(outputs), gradBiases(outputs) {

	this->inputs = inputs;
	this->outputs = outputs;

	activationType = GetActivationType(type);
	SetActivationFunction();

	for (int i = 0; i < outputs; i++)
		for (int j = 0; j < inputs; j++)
			f >> weights(i, j);

	std::string b;
	f >> b;

	if (b != "biases")
		throw std::runtime_error("Invalid fc description");

	for (int i = 0; i < outputs; i++)
		f >> biases[i];
}

// вывод конфигурации
void FullConnectedLayer::PrintConfig() const {
	std::cout << "| full connected | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << std::setw(12) << GetTrainableParams() << " | ";
	std::cout << outputs << " neurons";

	if (activationType != ActivationType::None)
		std::cout << ", f: " << GetActivationType();

	std::cout << std::endl;
}

// получение количество обучаемых параметров
int FullConnectedLayer::GetTrainableParams() const {
	return outputs * (inputs + 1);
}

// прямое распространение
void FullConnectedLayer::Forward(const Volume& input) {
	#pragma omp parallel for
	for (int i = 0; i < outputs; i++) {
		double sum = biases[i];

		for (int j = 0; j < inputs; j++)
			sum += weights(i, j) * input[j];

		(this->*Activate)(i, sum);
	}
}

// обратное распространение
void FullConnectedLayer::Backward(Volume& prevDeltas) {
	#pragma omp parallel for
	for (int i = 0; i < inputs; i++) {
		double sum = 0;

		for (int j = 0; j < outputs; j++)
			sum += weights(j, i) * deltas[j];

		prevDeltas[i] *= sum;
	}
}

// обновление весовых коэффициентов
void FullConnectedLayer::UpdateWeights(const Optimizer& optimizer, const Volume& input) {
	#pragma omp parallel for
	for (int i = 0; i < outputs; i++) {
		double delta = deltas[i];

		for (int j = 0; j < inputs; j++)
			optimizer.Update(delta * input[j], dw(i, j), dww(i, j), weights(i, j));
	
		optimizer.Update(delta, db[i], dbb[i], biases[i]);
	}
}


// вычисление градиентов
void FullConnectedLayer::CalculateGradients(const Volume &input) {
	#pragma omp parallel for
	for (int i = 0; i < outputs; i++) {
		double delta = deltas[i];

		for (int j = 0; j < inputs; j++)
			gradWeights(i, j) += delta * input[j];
		
		gradBiases[i] += delta;
	}
}

// обновление весовых коэффициентов
void FullConnectedLayer::UpdateWeights(const Optimizer& optimizer, int batchSize) {
	#pragma omp parallel for
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++) {
			optimizer.Update(gradWeights(i, j) / batchSize, dw(i, j), dww(i, j), weights(i, j));
			gradWeights(i, j) = 0;
		}
		
		optimizer.Update(gradBiases[i] / batchSize, db[i], dbb[i], biases[i]);
		gradBiases[i] = 0;
	}
}

// сброс параметров
void FullConnectedLayer::ResetCache() {
	#pragma omp parallel for
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++) {
			dw(i, j) = 0;
			dww(i, j) = 0;
		}

		db[i] = 0;
		dbb[i] = 0;
	}
}

// сохранение слоя в файл
void FullConnectedLayer::Save(std::ostream &f) {
	f << "fc " << inputs << " " << outputs << " " << GetActivationType() << std::endl;

	for (int i = 0; i < outputs; i++) {
		f << weights(i, 0);

		for (int j = 1; j < inputs; j++)
			f << " " << std::setprecision(15) << weights(i, j);

		f << std::endl;
	}

	f << "biases";

	for (int i = 0; i < outputs; i++)
		f << " " << std::setprecision(15) << biases[i];

	f << std::endl;
}

// установка веса
void FullConnectedLayer::SetWeight(int i, int j, double weight) {
	weights(i, j) = weight;
}

// установка мещения
void FullConnectedLayer::SetBias(int i, double bias) {
	biases[i] = bias;
}

// установка веса по индексу
void FullConnectedLayer::SetParam(int index, double weight) {
	int i = index / (inputs + 1);
	int j = index % (inputs + 1);

	if (j < inputs) {
		weights(i, j) = weight;
	}
	else {
		biases[i] = weight;
	}
}

// получение веса по индексу
double FullConnectedLayer::GetParam(int index) const {
	int i = index / (inputs + 1);
	int j = index % (inputs + 1);

	return j < inputs ? weights(i, j) : biases[i];
}

// получение градиента веса по индексу
double FullConnectedLayer::GetGradient(int index, const Volume &input) const {
	int i = index / (inputs + 1);
	int j = index % (inputs + 1);

	return deltas[i] * (j < inputs ? input[j] : 1);
}
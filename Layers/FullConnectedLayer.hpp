#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>

#include "../Entities/Matrix.hpp"
#include "NetworkLayer.hpp"

// тип активационной функции
enum class ActivationType {
	Sigmoid,
	Tanh,
	ReLU
};

class FullConnectedLayer : public NetworkLayer {
	int inputs; // число входов
	int outputs; // число выходов
	ActivationType activationType; // тип активационной функции

	Matrix weights; // веса
	Matrix dw; // изменение весов
	std::vector<double> biases; // смещения
	std::vector<double> db; // изменение смещений

	double (*activation)(double); // активационная функция
	double (*derivative)(double); // производная активационной функции

	ActivationType GetActivationType(const std::string& type) const; // получение типа активационной функции по строке
	std::string GetActivationType() const; // получение строки для активационной функции

	static double Sigmoid(double x);
	static double SigmoidDerivative(double y);

	static double Tangent(double x);
	static double TangentDerivative(double y);

	static double ReLU(double x);
	static double ReLUDerivative(double y);

public:
	FullConnectedLayer(int inputs, int outputs, const std::string& type = "sigmoid");
	FullConnectedLayer(int inputs, int outputs, const std::string& type, std::ifstream &f); // загрузка слоя из файла

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количество обучаемых параметров

	void Forward(const Volume& input); // прямое распространение
	void Backward(Volume& prevDeltas); // обратное распространение
	void UpdateWeights(const Optimizer& optimizer, const Volume& input); // обновление весовых коэффициентов
	
	void ResetCache(); // сброс параметров
	void Save(std::ostream &f); // сохранение слоя в файл

	void SetWeight(int i, int j, double weight); // установка веса
	void SetBias(int i, double bias); // установка смещения
};

// получение типа активационной функции по строке
ActivationType FullConnectedLayer::GetActivationType(const std::string& type) const {
	if (type == "sigmoid")
		return ActivationType::Sigmoid;
	
	if (type == "tanh")
		return ActivationType::Tanh;
	
	if (type == "relu")
		return ActivationType::ReLU;
	
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
	
	throw std::runtime_error("Invalid activation function");
}

double FullConnectedLayer::Sigmoid(double x) {
	return 1.0 / (1 + exp(-x));
}

double FullConnectedLayer::SigmoidDerivative(double y) {
	return y * (1 - y);
}

double FullConnectedLayer::Tangent(double x) {
	return tanh(x);
}

double FullConnectedLayer::TangentDerivative(double y) {
	return 1 - y * y;
}

double FullConnectedLayer::ReLU(double x) {
	return x > 0 ? x : 0;
}

double FullConnectedLayer::ReLUDerivative(double y) {
	return y > 0 ? 1 : 0;
}

FullConnectedLayer::FullConnectedLayer(int inputs, int outputs, const std::string& type) :
	NetworkLayer(1, 1, inputs, 1, 1, outputs),
	weights(outputs, inputs), dw(outputs, inputs),
	biases(outputs), db(outputs) {
	
	this->inputs = inputs;
	this->outputs = outputs;

	activationType = GetActivationType(type);

	if (activationType == ActivationType::Sigmoid) {
		activation = Sigmoid;
		derivative = SigmoidDerivative;
	}
	else if (activationType == ActivationType::Tanh) {
		activation = Tangent;
		derivative = TangentDerivative;
	}
	else if (activationType == ActivationType::ReLU) {
		activation = ReLU;
		derivative = ReLUDerivative;
	}

	weights.FillRandom(random, sqrt(2.0 / inputs));

	for (int i = 0; i < outputs; i++)
		biases[i] = 0.01;
}

// загрузка слоя из файла
FullConnectedLayer::FullConnectedLayer(int inputs, int outputs, const std::string& type, std::ifstream &f) : NetworkLayer(1, 1, inputs, 1, 1, outputs),
	weights(outputs, inputs), dw(outputs, inputs),
	biases(outputs), db(outputs) {

	this->inputs = inputs;
	this->outputs = outputs;

	activationType = GetActivationType(type);
	
	if (activationType == ActivationType::Sigmoid) {
		activation = Sigmoid;
		derivative = SigmoidDerivative;
	}
	else if (activationType == ActivationType::Tanh) {
		activation = Tangent;
		derivative = TangentDerivative;
	}
	else if (activationType == ActivationType::ReLU) {
		activation = ReLU;
		derivative = ReLUDerivative;
	}

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
	std::cout << "|   full connected layer   | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << std::setw(12) << (outputs * (inputs + 1)) << " | ";
	std::cout << outputs << " neurons, f: " << GetActivationType() << std::endl;
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
			sum += weights(i, j) * input(j, 0, 0);

		sum = activation(sum);

		deltas(i, 0, 0) = derivative(sum);
		output(i, 0, 0) = sum;
	}
}

// обратное распространение
void FullConnectedLayer::Backward(Volume& prevDeltas) {
	#pragma omp parallel for
	for (int i = 0; i < inputs; i++) {
		double sum = 0;

		for (int j = 0; j < outputs; j++)
			sum += weights(j, i) * deltas(j, 0, 0);

		prevDeltas(i, 0, 0) *= sum;
	}
}

// обновление весовых коэффициентов
void FullConnectedLayer::UpdateWeights(const Optimizer& optimizer, const Volume& input) {
	#pragma omp parallel for
	for (int i = 0; i < outputs; i++) {
		double delta = deltas(i, 0, 0);

		for (int j = 0; j < inputs; j++)
			optimizer.Update(delta * input(j, 0, 0), dw(i, j), weights(i, j));
	
		optimizer.Update(delta, db[i], biases[i]);
	}
}

// сброс параметров
void FullConnectedLayer::ResetCache() {
	#pragma omp parallel for
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			dw(i, j) = 0;

		db[i] = 0;
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
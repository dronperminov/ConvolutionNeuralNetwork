#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class BatchNormalizationLayer : public NetworkLayer {
	int total;
	double momentum;

	Volume gamma;
	Volume dgamma;
	std::vector<Volume> paramsgamma;

	Volume beta;
	Volume dbeta;
	std::vector<Volume> paramsbeta;

	std::vector<Volume> X_norm;

	Volume mu, var;
	Volume running_mu, running_var;

	void InitParams(); // инициализация параметров для обучения
	void InitWeights(); // инициализация весовых коэффициентов
	void LoadWeights(std::ifstream &f); // считывание весовых коэффициентов из файла

public:
	BatchNormalizationLayer(int width, int height, int deep, double momentum);
	BatchNormalizationLayer(int width, int height, int deep, double momentum, std::ifstream &f);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количества обучаемых параметров

	void ForwardOutput(const std::vector<Volume> &X); // прямое распространение
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

BatchNormalizationLayer::BatchNormalizationLayer(int width, int height, int deep, double momentum) : NetworkLayer(width, height, deep, width, height, deep),
	gamma(width, height, deep), dgamma(width, height, deep), beta(width, height, deep), dbeta(width, height, deep), 
	mu(width, height, deep), var(width, height, deep), running_mu(width, height, deep), running_var(width, height, deep) {

	this->momentum = momentum;
	total = width * height * deep;

	InitParams();
	InitWeights();
}

BatchNormalizationLayer::BatchNormalizationLayer(int width, int height, int deep, double momentum, std::ifstream &f) : NetworkLayer(width, height, deep, width, height, deep),
	gamma(width, height, deep), dgamma(width, height, deep), beta(width, height, deep), dbeta(width, height, deep), 
	mu(width, height, deep), var(width, height, deep), running_mu(width, height, deep), running_var(width, height, deep) {

	this->momentum = momentum;
	total = width * height * deep;

	InitParams();
	LoadWeights(f);
}

// инициализация параметров для обучения
void BatchNormalizationLayer::InitParams() {
	for (int i = 0; i < OPTIMIZER_PARAMS_COUNT; i++) {
		paramsgamma.push_back(Volume(outputSize));
		paramsbeta.push_back(Volume(outputSize));
	}
}

// инициализация весовых коэффициентов
void BatchNormalizationLayer::InitWeights() {
	for (int i = 0; i < total; i++) {
		gamma[i] = 1;
		beta[i] = 0;

		running_mu[i] = 0;
		running_var[i] = 0;
	}
}

// считывание весовых коэффициентов из файла
void BatchNormalizationLayer::LoadWeights(std::ifstream &f) {
	for (int i = 0; i < total; i++)
		f >> gamma[i];

	for (int i = 0; i < total; i++)
		f >> beta[i];

	for (int i = 0; i < total; i++)
		f >> running_mu[i];

	for (int i = 0; i < total; i++)
		f >> running_var[i];
}

// вывод конфигурации
void BatchNormalizationLayer::PrintConfig() const {
	std::cout << "| batch norm.    | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << std::setw(12) << GetTrainableParams() << " | ";
	std::cout << "moment: " << momentum;
	std::cout << std::endl;
}

// получение количество обучаемых параметров
int BatchNormalizationLayer::GetTrainableParams() const {
	return 2 * total;
}

// прямое распространение
void BatchNormalizationLayer::ForwardOutput(const std::vector<Volume> &X) {
	output = std::vector<Volume>(X.size(), Volume(outputSize));

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			output[batchIndex][i] = gamma[i] * (X[batchIndex][i] - running_mu[i]) / sqrt(running_var[i] + 1e-8) + beta[i];
}

// прямое распространение
void BatchNormalizationLayer::Forward(const std::vector<Volume> &X) {
	output = std::vector<Volume>(X.size(), Volume(outputSize));
	X_norm = std::vector<Volume>(X.size(), Volume(inputSize));

	#pragma omp parallel for
	for (int i = 0; i < total; i++) {
		double sum = 0;

		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
			sum += X[batchIndex][i];

		mu[i] = sum / X.size();
		sum = 0;

		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
			sum += pow(X[batchIndex][i] - mu[i], 2);

		var[i] = sum / X.size();

		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
			X_norm[batchIndex][i] = (X[batchIndex][i] - mu[i]) / sqrt(var[i] + 1e-8);
			output[batchIndex][i] = gamma[i] * X_norm[batchIndex][i] + beta[i];
		}

		running_mu[i] = momentum * running_mu[i] + (1 - momentum) * mu[i];
		running_var[i] = momentum * running_var[i] + (1 - momentum) * var[i];
	}
}

// обратное распространение
void BatchNormalizationLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			double delta = dout[batchIndex][i];

			dgamma[i] += delta * X_norm[batchIndex][i];
			dbeta[i] += delta;
		}
	}

	if (calc_dX) {
		size_t N = dout.size();

		std::vector<Volume> dX_norm(N, Volume(outputSize));
		Volume d1(outputSize);
		Volume d2(outputSize);

		for (size_t batchIndex = 0; batchIndex < N; batchIndex++) {
			for (int i = 0; i < total; i++) {
				dX_norm[batchIndex][i] = dout[batchIndex][i] * gamma[i];

				d1[i] += dX_norm[batchIndex][i];
				d2[i] += dX_norm[batchIndex][i] * X_norm[batchIndex][i];
			}
		}

		dX = std::vector<Volume>(N, Volume(outputSize));

		#pragma omp parallel for collapse(2)
		for (int i = 0; i < total; i++)
			for (size_t batchIndex = 0; batchIndex < N; batchIndex++)
				dX[batchIndex][i] = (dX_norm[batchIndex][i] - (d1[i] + X_norm[batchIndex][i] * d2[i]) / N) / (sqrt(var[i] + 1e-8));
	}
}

// обновление весовых коэффициентов
void BatchNormalizationLayer::UpdateWeights(const Optimizer &optimizer) {
	#pragma omp parallel for
	for (int i = 0; i < total; i++) {
		optimizer.Update(dbeta[i], paramsbeta[0][i], paramsbeta[1][i], beta[i]);
		optimizer.Update(dgamma[i], paramsgamma[0][i], paramsgamma[1][i], gamma[i]);

		dbeta[i] = 0;
		dgamma[i] = 0;
	}
}

// сброс параметров
void BatchNormalizationLayer::ResetCache() {
	for (int i = 0; i < total; i++) {
		running_mu[i] = 0;
		running_var[i] = 0;

		for (int j = 0; j < OPTIMIZER_PARAMS_COUNT; j++) {
			paramsgamma[j][i] = 0;
			paramsbeta[j][i] = 0;
		}
	}
}

// сохранение слоя в файл
void BatchNormalizationLayer::Save(std::ofstream &f) const {
	f << "batchnormalization " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " " << momentum << std::endl;

	for (int i = 0; i < total; i++)
		f << std::setprecision(15) << gamma[i] << " ";

	f << std::endl;

	for (int i = 0; i < total; i++)
		f << std::setprecision(15) << beta[i] << " ";

	f << std::endl;

	for (int i = 0; i < total; i++)
		f << std::setprecision(15) << running_mu[i] << " ";

	f << std::endl;

	for (int i = 0; i < total; i++)
		f << std::setprecision(15) << running_var[i] << " ";

	f << std::endl;
}

// установка веса по индексу
void BatchNormalizationLayer::SetParam(int index, double weight) {
	if (index / total == 0) {
		gamma[index] = weight;
	}
	else {
		beta[index % total] = weight;
	}
}

// получение веса по индексу
double BatchNormalizationLayer::GetParam(int index) const {
	if (index / total == 0) {
		return gamma[index];
	}
	else {
		return beta[index % total];
	}
}

// получение градиента веса по индексу
double BatchNormalizationLayer::GetGradient(int index) const {
	if (index / total == 0) {
		return dgamma[index];
	}
	else {
		return dbeta[index % total];
	}
}

// обнуление градиента веса по индексу
void BatchNormalizationLayer::ZeroGradient(int index) {
	if (index / total == 0) {
		dgamma[index] = 0;
	}
	else {
		dbeta[index % total] = 0;
	}
}
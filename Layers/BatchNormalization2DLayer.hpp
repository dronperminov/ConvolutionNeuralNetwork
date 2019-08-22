#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class BatchNormalization2DLayer : public NetworkLayer {
	int wh;
	double momentum;

	Volume gamma;
	Volume dgamma;
	std::vector<Volume> paramsgamma;

	Volume beta;
	Volume dbeta;
	std::vector<Volume> paramsbeta;

	std::vector<Volume> X_norm;
	std::vector<Volume> dX_norm;

	Volume mu, var;
	Volume running_mu, running_var;

	void InitParams(); // инициализация параметров для обучения
	void InitWeights(); // инициализация весовых коэффициентов
	void LoadWeights(std::ifstream &f); // считывание весовых коэффициентов из файла

public:
	BatchNormalization2DLayer(int width, int height, int deep, double momentum);
	BatchNormalization2DLayer(int width, int height, int deep, double momentum, std::ifstream &f);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количества обучаемых параметров

	void ForwardOutput(const std::vector<Volume> &X); // прямое распространение
	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение
	void UpdateWeights(const Optimizer &optimizer); // обновление весовых коэффициентов

	void ResetCache(); // сброс параметров
	void Save(std::ofstream &f) const; // сохранение слоя в файл
	void SetBatchSize(int batchSize); // установка размера батча

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index) const; // получение градиента веса по индексу
	void ZeroGradient(int index); // обнуление градиента веса по индексу
};

BatchNormalization2DLayer::BatchNormalization2DLayer(int width, int height, int deep, double momentum) : NetworkLayer(width, height, deep, width, height, deep),
	gamma(1, 1, deep), dgamma(1, 1, deep), beta(1, 1, deep), dbeta(1, 1, deep), 
	mu(1, 1, deep), var(1, 1, deep), running_mu(1, 1, deep), running_var(1, 1, deep) {

	this->momentum = momentum;
	wh = width * height;

	InitParams();
	InitWeights();
}

BatchNormalization2DLayer::BatchNormalization2DLayer(int width, int height, int deep, double momentum, std::ifstream &f) : NetworkLayer(width, height, deep, width, height, deep),
	gamma(width, height, deep), dgamma(width, height, deep), beta(width, height, deep), dbeta(width, height, deep), 
	mu(width, height, deep), var(width, height, deep), running_mu(width, height, deep), running_var(width, height, deep) {

	this->momentum = momentum;
	wh = width * height;

	InitParams();
	LoadWeights(f);
}

// инициализация параметров для обучения
void BatchNormalization2DLayer::InitParams() {
	for (int i = 0; i < OPTIMIZER_PARAMS_COUNT; i++) {
		paramsgamma.push_back(Volume(1, 1, outputSize.deep));
		paramsbeta.push_back(Volume(1, 1, outputSize.deep));
	}
}

// инициализация весовых коэффициентов
void BatchNormalization2DLayer::InitWeights() {
	for (int i = 0; i < outputSize.deep; i++) {
		gamma[i] = 1;
		beta[i] = 0;

		running_mu[i] = 0;
		running_var[i] = 0;
	}
}

// считывание весовых коэффициентов из файла
void BatchNormalization2DLayer::LoadWeights(std::ifstream &f) {
	for (int i = 0; i < outputSize.deep; i++)
		f >> gamma[i];

	for (int i = 0; i < outputSize.deep; i++)
		f >> beta[i];

	for (int i = 0; i < outputSize.deep; i++)
		f >> running_mu[i];

	for (int i = 0; i < outputSize.deep; i++)
		f >> running_var[i];
}

// вывод конфигурации
void BatchNormalization2DLayer::PrintConfig() const {
	std::cout << "| batch norm. 2D | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << std::setw(12) << GetTrainableParams() << " | ";
	std::cout << "moment: " << momentum;
	std::cout << std::endl;
}

// получение количество обучаемых параметров
int BatchNormalization2DLayer::GetTrainableParams() const {
	return 2 * outputSize.deep;
}

// прямое распространение
void BatchNormalization2DLayer::ForwardOutput(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(4)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
		for (int d = 0; d < outputSize.deep; d++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					output[batchIndex](d, i, j) = gamma[d] * (X[batchIndex](d, i, j) - running_mu[d]) / sqrt(running_var[d] + 1e-8) + beta[d];	
}

// прямое распространение
void BatchNormalization2DLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for
	for (int d = 0; d < outputSize.deep; d++) {
		double sum = 0;

		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					sum += X[batchIndex](d, i, j);

		mu[d] = sum / (X.size() * wh);
		sum = 0;

		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					sum += pow(X[batchIndex](d, i, j) - mu[d], 2);

		var[d] = sum / (X.size() * wh);

		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
			for (int i = 0; i < outputSize.height; i++) {
				for (int j = 0; j < outputSize.width; j++) {
					X_norm[batchIndex](d, i, j) = (X[batchIndex](d, i, j) - mu[d]) / sqrt(var[d] + 1e-8);
					output[batchIndex](d, i, j) = gamma[d] * X_norm[batchIndex](d, i, j) + beta[d];
				}
			}
		}

		running_mu[d] = momentum * running_mu[d] + (1 - momentum) * mu[d];
		running_var[d] = momentum * running_var[d] + (1 - momentum) * var[d];
	}
}

// обратное распространение
void BatchNormalization2DLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int d = 0; d < outputSize.deep; d++) {
			for (int i = 0; i < outputSize.height; i++) {
				for (int j = 0; j < outputSize.width; j++) {
					double delta = dout[batchIndex](d, i, j);

					dgamma[d] += delta * X_norm[batchIndex](d, i, j);
					dbeta[d] += delta;
				}
			}
		}
	}

	if (calc_dX) {
		size_t N = dout.size();

		Volume d1(1, 1, outputSize.deep);
		Volume d2(1, 1, outputSize.deep);

		for (size_t batchIndex = 0; batchIndex < N; batchIndex++) {
			for (int d = 0; d < outputSize.deep; d++) {
				for (int i = 0; i < outputSize.height; i++) {
					for (int j = 0; j < outputSize.width; j++) {
						dX_norm[batchIndex](d, i, j) = dout[batchIndex](d, i, j) * gamma[d];

						d1[d] += dX_norm[batchIndex](d, i, j);
						d2[d] += dX_norm[batchIndex](d, i, j) * X_norm[batchIndex](d, i, j);
					}
				}
			}
		}

		#pragma omp parallel for collapse(4)
		for (int d = 0; d < outputSize.deep; d++)
				for (int i = 0; i < outputSize.height; i++)
					for (int j = 0; j < outputSize.width; j++)
						for (size_t batchIndex = 0; batchIndex < N; batchIndex++)
							dX[batchIndex](d, i, j) = (dX_norm[batchIndex](d, i, j) - (d1[d] + X_norm[batchIndex](d, i, j) * d2[d]) / (N * wh)) / (sqrt(var[d] + 1e-8));
	}
}

// обновление весовых коэффициентов
void BatchNormalization2DLayer::UpdateWeights(const Optimizer &optimizer) {
	#pragma omp parallel for
	for (int i = 0; i < outputSize.deep; i++) {
		optimizer.Update(dbeta[i], paramsbeta[0][i], paramsbeta[1][i], paramsbeta[2][i], beta[i]);
		optimizer.Update(dgamma[i], paramsgamma[0][i], paramsgamma[1][i], paramsgamma[2][i], gamma[i]);

		dbeta[i] = 0;
		dgamma[i] = 0;
	}
}

// сброс параметров
void BatchNormalization2DLayer::ResetCache() {
	for (int i = 0; i < outputSize.deep; i++) {
		running_mu[i] = 0;
		running_var[i] = 0;

		for (int j = 0; j < OPTIMIZER_PARAMS_COUNT; j++) {
			paramsgamma[j][i] = 0;
			paramsbeta[j][i] = 0;
		}
	}
}

// сохранение слоя в файл
void BatchNormalization2DLayer::Save(std::ofstream &f) const {
	f << "batchnormalization2D " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " " << momentum << std::endl;

	for (int i = 0; i < outputSize.deep; i++)
		f << std::setprecision(15) << gamma[i] << " ";

	f << std::endl;

	for (int i = 0; i < outputSize.deep; i++)
		f << std::setprecision(15) << beta[i] << " ";

	f << std::endl;

	for (int i = 0; i < outputSize.deep; i++)
		f << std::setprecision(15) << running_mu[i] << " ";

	f << std::endl;

	for (int i = 0; i < outputSize.deep; i++)
		f << std::setprecision(15) << running_var[i] << " ";

	f << std::endl;
}

// установка размера батча
void BatchNormalization2DLayer::SetBatchSize(int batchSize) {
	output = std::vector<Volume>(batchSize, Volume(outputSize));
	dX = std::vector<Volume>(batchSize, Volume(inputSize));

	X_norm = std::vector<Volume>(batchSize, Volume(inputSize));
	dX_norm = std::vector<Volume>(batchSize, Volume(inputSize));
}

// установка веса по индексу
void BatchNormalization2DLayer::SetParam(int index, double weight) {
	if (index / outputSize.deep == 0) {
		gamma[index] = weight;
	}
	else {
		beta[index % outputSize.deep] = weight;
	}
}

// получение веса по индексу
double BatchNormalization2DLayer::GetParam(int index) const {
	if (index / outputSize.deep == 0) {
		return gamma[index];
	}
	else {
		return beta[index % outputSize.deep];
	}
}

// получение градиента веса по индексу
double BatchNormalization2DLayer::GetGradient(int index) const {
	if (index / outputSize.deep == 0) {
		return dgamma[index];
	}
	else {
		return dbeta[index % outputSize.deep];
	}
}

// обнуление градиента веса по индексу
void BatchNormalization2DLayer::ZeroGradient(int index) {
	if (index / outputSize.deep == 0) {
		dgamma[index] = 0;
	}
	else {
		dbeta[index % outputSize.deep] = 0;
	}
}
#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <vector>

#include "NetworkLayer.hpp"

class SamplerLayer : public NetworkLayer {
	int total;
	int outputs;

	double kl;

	NetworkLayer *muLayer;
	NetworkLayer *stdLayer;

	std::vector<Volume> deltas;
	std::vector<Volume> dL_mu;
	std::vector<Volume> dL_std;

	std::default_random_engine generator;
	std::normal_distribution<double> distribution;

public:
	SamplerLayer(VolumeSize size, int outputs, double kl);
	SamplerLayer(VolumeSize size, int outputs, double kl, std::ifstream &f);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение
	void UpdateWeights(const Optimizer &optimizer, bool trainable); // обновление весовых коэффициентов

	void ResetCache();
	void SetBatchSize(int batchSize); // установка размера батча
	void Save(std::ofstream &f) const; // сохранение слоя в файл

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index) const; // получение градиента веса по индексу
	void ZeroGradient(int index); // обнуление градиента веса по индексу
};

SamplerLayer::SamplerLayer(VolumeSize size, int outputs, double kl) : NetworkLayer(size, 1, 1, outputs), distribution(0, 1) {
	this->outputs = outputs;
	this->total = size.width * size.height * size.deep;
	this->kl = kl;

	muLayer = new FullyConnectedLayer(size, outputs);
	stdLayer = new FullyConnectedLayer(size, outputs);

	name = "sampler";
	info = "outputs: " + std::to_string(outputs);

	if (kl > 0)
		info += ", kl: " + std::to_string(kl);
}

SamplerLayer::SamplerLayer(VolumeSize size, int outputs, double kl, std::ifstream &f) : NetworkLayer(size, 1, 1, outputs), distribution(0, 1) {
	this->outputs = outputs;
	this->total = size.width * size.height * size.deep;
	this->kl = kl;

	std::string layerType;
	VolumeSize tmpSize;
	f >> layerType >> tmpSize;
	muLayer = LoadLayer(tmpSize, layerType, f);
	f >> layerType >> tmpSize;
	stdLayer = LoadLayer(tmpSize, layerType, f);

	name = "sampler";
	info = "outputs: " + std::to_string(outputs);

	if (kl > 0)
		info += ", kl: " + std::to_string(kl);
}

// получение количества обучаемых параметров
int SamplerLayer::GetTrainableParams() const {
	return muLayer->GetTrainableParams() + stdLayer->GetTrainableParams();
}

// прямое распространение
void SamplerLayer::Forward(const std::vector<Volume> &X) {
	muLayer->Forward(X);
	stdLayer->Forward(X);

	std::vector<Volume> &mu = muLayer->GetOutput();
	std::vector<Volume> &std = stdLayer->GetOutput();

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int i = 0; i < outputs; i++) {
			double e = exp(0.5 * std[batchIndex][i]) * distribution(generator);

			output[batchIndex][i] = e + mu[batchIndex][i];
			deltas[batchIndex][i] = 0.5 * e;

			// loss += 1 + std[batchIndex][i] - mu[batchIndex][i] * mu[batchIndex][i] - exp(std[batchIndex][i]);
			dL_mu[batchIndex][i] = kl * mu[batchIndex][i];
			dL_std[batchIndex][i] = kl * (-0.5 + 0.5 * exp(std[batchIndex][i]));
		}
	}
}

// обратное распространение
void SamplerLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int i = 0; i < outputs; i++) {
			dL_mu[batchIndex][i] += dout[batchIndex][i];
			dL_std[batchIndex][i] += dout[batchIndex][i] * deltas[batchIndex][i];
		}
	}

	muLayer->Backward(dL_mu, X, true);
	stdLayer->Backward(dL_std, X, true);

	if (!calc_dX)
		return;

	std::vector<Volume> &dmu = muLayer->GetDeltas();
	std::vector<Volume> &dstd = stdLayer->GetDeltas();

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
		for (int i = 0; i < total; i++)
			dX[batchIndex][i] = dmu[batchIndex][i] + dstd[batchIndex][i];
}

// обновление весовых коэффициентов
void SamplerLayer::UpdateWeights(const Optimizer &optimizer, bool trainable) {
	muLayer->UpdateWeights(optimizer, trainable);
	stdLayer->UpdateWeights(optimizer, trainable);
}

void SamplerLayer::ResetCache() {
	muLayer->ResetCache();
	stdLayer->ResetCache();
}

// сохранение слоя в файл
void SamplerLayer::Save(std::ofstream &f) const {
	f << "sampler " << inputSize << " " << outputs << " " << kl << std::endl;
	muLayer->Save(f);
	stdLayer->Save(f);
}

// установка размера батча
void SamplerLayer::SetBatchSize(int batchSize) {
	muLayer->SetBatchSize(batchSize);
	stdLayer->SetBatchSize(batchSize);

	output = std::vector<Volume>(batchSize, Volume(outputSize));
	dX = std::vector<Volume>(batchSize, Volume(inputSize));
	
	deltas = std::vector<Volume>(batchSize, Volume(outputSize));
	dL_mu = std::vector<Volume>(batchSize, Volume(outputSize));
	dL_std = std::vector<Volume>(batchSize, Volume(outputSize));
}

// установка веса по индексу
void SamplerLayer::SetParam(int index, double weight) {
	int n = muLayer->GetTrainableParams();

	if (index < n)
		muLayer->SetParam(index, weight);
	else
		stdLayer->SetParam(index - n, weight);
}

// получение веса по индексу
double SamplerLayer::GetParam(int index) const {
	int n = muLayer->GetTrainableParams();

	if (index < n)
		return muLayer->GetParam(index);
	
	return stdLayer->GetParam(index - n);
}

// получение градиента веса по индексу
double SamplerLayer::GetGradient(int index) const {
	int n = muLayer->GetTrainableParams();

	if (index < n)
		return muLayer->GetGradient(index);
	
	return stdLayer->GetGradient(index - n);
}

// обнуление градиента веса по индексу
void SamplerLayer::ZeroGradient(int index) {
	int n = muLayer->GetTrainableParams();

	if (index < n)
		muLayer->ZeroGradient(index);
	else
		stdLayer->ZeroGradient(index - n);
}
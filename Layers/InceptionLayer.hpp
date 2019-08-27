#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"
#include "BatchNormalization2DLayer.hpp"
#include "ConvLayer.hpp"

class InceptionLayer : public NetworkLayer {
	std::vector<NetworkLayer*> convs;
	int totalOutput;
	int totalInput;
	size_t last;

public:
	InceptionLayer(VolumeSize size, int features);
	InceptionLayer(VolumeSize size, int features, std::ifstream &f);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение
	void UpdateWeights(const Optimizer &optimizer); // обновление весовых коэффициентов

	void ResetCache();
	void Save(std::ofstream &f) const; // сохранение слоя в файл
	void SetBatchSize(int batchSize); // установка размера батча

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index) const; // получение градиента веса по индексу
	void ZeroGradient(int index); // обнуление градиента веса по индексу
};

InceptionLayer::InceptionLayer(VolumeSize size, int features) : NetworkLayer(size, size.width, size.height, 3*features) {
	convs.push_back(new ConvLayer(size, features, 1, 0, 1));
	convs.push_back(new ConvLayer(size, features, 3, 1, 1));
	convs.push_back(new ConvLayer(size, features, 5, 2, 1));

	totalInput = size.width * size.height * size.deep;
	totalOutput = size.width * size.height * convs.size() * features;

	last = convs.size() - 1;

	name = "inception";
	info = "features: " + std::to_string(features);
}

InceptionLayer::InceptionLayer(VolumeSize size, int features, std::ifstream &f) : NetworkLayer(size, size.width, size.height, 3*features) {
	for (int i = 0; i < 3; i++) {
		std::string name;
		VolumeSize blockSize;

		f >> name >> blockSize;

		std::cout << name;

		if (name != "conv")
			throw std::runtime_error("Unknown layer name '" + name + "' for inception layer");
		
		int fc, fs, P, S;
		f >> fs >> fc >> P >> S;

		convs.push_back(new ConvLayer(blockSize, fc, fs, P, S, f));
	}

	totalInput = size.width * size.height * size.deep;
	totalOutput = size.width * size.height * 3 * features;

	last = convs.size() - 1;

	name = "inception";
	info = "features: " + std::to_string(features);
}

// получение количества обучаемых параметров
int InceptionLayer::GetTrainableParams() const {
	int count = 0;

	for (size_t i = 0; i < convs.size(); i++)
		count += convs[i]->GetTrainableParams();

	return count;
}

// прямое распространение
void InceptionLayer::Forward(const std::vector<Volume> &X) {
	int deep = outputSize.deep / convs.size();

	for (size_t index = 0; index < convs.size(); index++) {
		convs[index]->Forward(X);

		std::vector<Volume> &convOutput = convs[index]->GetOutput();

		#pragma omp parallel for collapse(4)
		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
			for (int i = 0; i < outputSize.height; i++)
					for (int j = 0; j < outputSize.width; j++)
						for (int k = 0; k < deep; k++)
							output[batchIndex](index * deep + k, i, j) = convOutput[batchIndex](k, i, j);
	}
}

// обратное распространение
void InceptionLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	int deep = outputSize.deep / convs.size();

	for (size_t index = 0; index < convs.size(); index++) {
		std::vector<Volume> douts(dout.size(), Volume(outputSize.width, outputSize.height, deep));

		#pragma omp parallel for collapse(4)
		for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					for (int k = 0; k < deep; k++)
						douts[batchIndex](k, i, j) = dout[batchIndex](index * deep + k, i, j);

		convs[index]->Backward(douts, X, calc_dX);
	}

	if (!calc_dX)
		return;
	
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int i = 0; i < totalInput; i++) {
			double sum = 0;

			for (size_t index = 0; index < convs.size(); index++)
				sum += convs[index]->GetDeltas()[batchIndex][i];

			dX[batchIndex][i] = sum;
		}
	}
}

// обновление весовых коэффициентов
void InceptionLayer::UpdateWeights(const Optimizer &optimizer) {
	for (size_t i = 0; i < convs.size(); i++)
		convs[i]->UpdateWeights(optimizer);
}

void InceptionLayer::ResetCache() {
	for (size_t i = 0; i < convs.size(); i++)
		convs[i]->ResetCache();
}

// сохранение слоя в файл
void InceptionLayer::Save(std::ofstream &f) const {
	f << "inception " << inputSize << " " << outputSize.deep << " " << convs.size() << std::endl;

	for (size_t i = 0; i < convs.size(); i++)
		convs[i]->Save(f);
}

// установка размера батча
void InceptionLayer::SetBatchSize(int batchSize) {
	output = std::vector<Volume>(batchSize, Volume(outputSize));
	dX = std::vector<Volume>(batchSize, Volume(inputSize));

	for (size_t i = 0; i < convs.size(); i++)
		convs[i]->SetBatchSize(batchSize);
}

// установка веса по индексу
void InceptionLayer::SetParam(int index, double weight) {
	int count = 0;

	for (size_t i = 0; i < convs.size(); i++) {
		int params = convs[i]->GetTrainableParams();
		
		if (index < params) {
			convs[i]->SetParam(index % params, weight);
			return;
		}		

		index -= params;
	}
}

// получение веса по индексу
double InceptionLayer::GetParam(int index) const {
	int count = 0;

	for (size_t i = 0; i < convs.size(); i++) {
		int params = convs[i]->GetTrainableParams();

		if (index < params)
			return convs[i]->GetParam(index % params);

		index -= params;
	}

	throw std::runtime_error("Error");
}

// получение градиента веса по индексу
double InceptionLayer::GetGradient(int index) const {
	int count = 0;

	for (size_t i = 0; i < convs.size(); i++) {
		int params = convs[i]->GetTrainableParams();
		
		if (index < params)
			return convs[i]->GetGradient(index % params);

		index -= params;
	}

	throw std::runtime_error("Error");
}

// обнуление градиента веса по индексу
void InceptionLayer::ZeroGradient(int index) {
	int count = 0;

	for (size_t i = 0; i < convs.size(); i++) {
		int params = convs[i]->GetTrainableParams();
		
		if (index < params) {
			convs[i]->ZeroGradient(index % params);
			return;
		}		

		index -= params;
	}
}
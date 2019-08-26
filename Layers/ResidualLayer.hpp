#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"
#include "BatchNormalization2DLayer.hpp"
#include "ConvLayer.hpp"

class ResidualLayer : public NetworkLayer {
	std::vector<NetworkLayer*> convBlock;
	NetworkLayer *skipBlock;
	int totalOutput;
	int totalInput;
	size_t last;

public:
	ResidualLayer(VolumeSize size, int featureMapsOut);
	ResidualLayer(VolumeSize size, int featureMapsOut, std::ifstream &f);

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

ResidualLayer::ResidualLayer(VolumeSize size, int featureMapsOut) : NetworkLayer(size, size.width, size.height, featureMapsOut) {
	convBlock.push_back(new ConvLayer(size, featureMapsOut, 3, 1, 1));
	convBlock.push_back(new BatchNormalization2DLayer(outputSize, 0.9));
	convBlock.push_back(new ConvLayer(outputSize, featureMapsOut, 3, 1, 1));
	convBlock.push_back(new BatchNormalization2DLayer(outputSize, 0.9));

	if (size.deep != featureMapsOut) {
		skipBlock = new ConvLayer(size, featureMapsOut, 1, 0, 1);
	}
	else {
		skipBlock = nullptr;
	}

	totalInput = size.width * size.height * size.deep;
	totalOutput = size.width * size.height * featureMapsOut;

	last = convBlock.size() - 1;

	name = "residual";
	info = "features: " + std::to_string(featureMapsOut);
}

ResidualLayer::ResidualLayer(VolumeSize size, int featureMapsOut, std::ifstream &f) : NetworkLayer(size, size.width, size.height, featureMapsOut) {
	int layers;
	f >> layers;

	for (int i = 0; i < layers; i++) {
		std::string name;
		VolumeSize blockSize;

		f >> name >> blockSize;

		std::cout << name;

		if (name == "conv") {
			int fc, fs, P, S;
			f >> fs >> fc >> P >> S;		
			convBlock.push_back(new ConvLayer(blockSize, fc, fs, P, S, f));
		}
		else if (name == "batchnormalization2D") {
			double momentum;
			f >> momentum;
			convBlock.push_back(new BatchNormalization2DLayer(blockSize, momentum, f));
		}
		else {
			throw std::runtime_error("Unknown layer name '" + name + "' for residual layer");
		}
	}

	if (size.deep != featureMapsOut) {
		int tmp;
		VolumeSize blockSize;

		f >> name >> blockSize;

		if (name != "conv")
			throw std::runtime_error("Invalid skip layer config");

		f >> tmp >> tmp >> tmp >> tmp;
		skipBlock = new ConvLayer(blockSize, featureMapsOut, 1, 0, 1, f);
	}
	else {
		skipBlock = nullptr;
	}

	totalInput = size.width * size.height * size.deep;
	totalOutput = size.width * size.height * featureMapsOut;

	last = convBlock.size() - 1;

	name = "residual";
	info = "features: " + std::to_string(featureMapsOut);
}

// получение количества обучаемых параметров
int ResidualLayer::GetTrainableParams() const {
	int count = 0;

	for (size_t i = 0; i < convBlock.size(); i++)
		count += convBlock[i]->GetTrainableParams();

	if (skipBlock)
		count += skipBlock->GetTrainableParams();

	return count;
}

// прямое распространение
void ResidualLayer::Forward(const std::vector<Volume> &X) {
	convBlock[0]->Forward(X);

	for (size_t i = 1; i < convBlock.size(); i++)
		convBlock[i]->Forward(convBlock[i - 1]->GetOutput());

	std::vector<Volume> &convOutput = convBlock[last]->GetOutput();

	if (skipBlock) {
		skipBlock->Forward(X);

		std::vector<Volume> &skipOutput = skipBlock->GetOutput();

		#pragma omp parallel for collapse(2)
		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
			for (int i = 0; i < totalOutput; i++)
				output[batchIndex][i] = skipOutput[batchIndex][i] + convOutput[batchIndex][i];
	}
	else {
		#pragma omp parallel for collapse(2)
		for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
			for (int i = 0; i < totalOutput; i++)
				output[batchIndex][i] = X[batchIndex][i] + convOutput[batchIndex][i];
	}
}

// обратное распространение
void ResidualLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (last == 0) {
		convBlock[0]->Backward(dout, X, calc_dX);
	}
	else {
		convBlock[last]->Backward(dout, convBlock[last - 1]->GetOutput(), true);

		for (size_t i = last - 1; i > 0; i--)
			convBlock[i]->Backward(convBlock[i + 1]->GetDeltas(), convBlock[i - 1]->GetOutput(), true);

		convBlock[0]->Backward(convBlock[1]->GetDeltas(), X, calc_dX);
	}

	if (!calc_dX)
		return;

	std::vector<Volume> &deltas = convBlock[0]->GetDeltas();

	if (skipBlock) {
		skipBlock->Backward(dout, X, true);
		
		std::vector<Volume> &skipDeltas = skipBlock->GetDeltas();

		#pragma omp parallel for collapse(2)
		for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
			for (int i = 0; i < totalInput; i++)
				dX[batchIndex][i] = skipDeltas[batchIndex][i] + deltas[batchIndex][i];
	}
	else {
		#pragma omp parallel for collapse(2)
		for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
			for (int i = 0; i < totalInput; i++)
				dX[batchIndex][i] = dout[batchIndex][i] + deltas[batchIndex][i];
	}
}

// обновление весовых коэффициентов
void ResidualLayer::UpdateWeights(const Optimizer &optimizer) {
	for (size_t i = 0; i < convBlock.size(); i++)
		convBlock[i]->UpdateWeights(optimizer);

	if (skipBlock)
		skipBlock->UpdateWeights(optimizer);
}

void ResidualLayer::ResetCache() {
	for (size_t i = 0; i < convBlock.size(); i++)
		convBlock[i]->ResetCache();

	if (skipBlock)
		skipBlock->ResetCache();
}

// сохранение слоя в файл
void ResidualLayer::Save(std::ofstream &f) const {
	f << "residual " << inputSize << " " << outputSize.deep << " " << convBlock.size() << std::endl;

	for (size_t i = 0; i < convBlock.size(); i++)
		convBlock[i]->Save(f);

	if (skipBlock)
		skipBlock->Save(f);
}

// установка размера батча
void ResidualLayer::SetBatchSize(int batchSize) {
	output = std::vector<Volume>(batchSize, Volume(outputSize));
	dX = std::vector<Volume>(batchSize, Volume(inputSize));

	for (size_t i = 0; i < convBlock.size(); i++)
		convBlock[i]->SetBatchSize(batchSize);

	if (skipBlock)
		skipBlock->SetBatchSize(batchSize);
}

// установка веса по индексу
void ResidualLayer::SetParam(int index, double weight) {
	int count = 0;

	for (size_t i = 0; i < convBlock.size(); i++) {
		int params = convBlock[i]->GetTrainableParams();
		
		if (index < params) {
			convBlock[i]->SetParam(index % params, weight);
			return;
		}		

		index -= params;
	}

	skipBlock->SetParam(index, weight);
}

// получение веса по индексу
double ResidualLayer::GetParam(int index) const {
	int count = 0;

	for (size_t i = 0; i < convBlock.size(); i++) {
		int params = convBlock[i]->GetTrainableParams();

		if (index < params)
			return convBlock[i]->GetParam(index % params);

		index -= params;
	}

	return skipBlock->GetParam(index);
}

// получение градиента веса по индексу
double ResidualLayer::GetGradient(int index) const {
	int count = 0;

	for (size_t i = 0; i < convBlock.size(); i++) {
		int params = convBlock[i]->GetTrainableParams();
		
		if (index < params)
			return convBlock[i]->GetGradient(index % params);

		index -= params;
	}

	return skipBlock->GetGradient(index);
}

// обнуление градиента веса по индексу
void ResidualLayer::ZeroGradient(int index) {
	int count = 0;

	for (size_t i = 0; i < convBlock.size(); i++) {
		int params = convBlock[i]->GetTrainableParams();
		
		if (index < params) {
			convBlock[i]->ZeroGradient(index % params);
			return;
		}		

		index -= params;
	}

	skipBlock->ZeroGradient(index);
}
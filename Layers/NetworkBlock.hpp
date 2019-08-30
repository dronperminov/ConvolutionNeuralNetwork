#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "NetworkLayer.hpp"
#include "ConvLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "AveragePoolingLayer.hpp"
#include "FullyConnectedLayer.hpp"

#include "ResidualLayer.hpp"
#include "InceptionLayer.hpp"

#include "DropoutLayer.hpp"
#include "BatchNormalizationLayer.hpp"
#include "BatchNormalization2DLayer.hpp"

#include "Activations/SigmoidLayer.hpp"
#include "Activations/LogSigmoidLayer.hpp"
#include "Activations/TanhLayer.hpp"
#include "Activations/ReLULayer.hpp"
#include "Activations/ELULayer.hpp"
#include "Activations/ParametricReLULayer.hpp"
#include "Activations/SwishLayer.hpp"
#include "Activations/SoftsignLayer.hpp"
#include "Activations/SoftplusLayer.hpp"
#include "Activations/SoftmaxLayer.hpp"

class NetworkBlock : public NetworkLayer {
	enum class MergeType {
		Sum,
		Stack
	};

	std::vector<std::vector<NetworkLayer *>> blocks;
	MergeType type;

	void MergeSum();
	void MergeStack();

	MergeType GetMergeType(const std::string& type);
	std::string GetMergeType() const;

public:
	NetworkBlock(VolumeSize size, const std::string& type);
	NetworkBlock(VolumeSize size, const std::string&, std::ifstream &f);

	void AddLayer(int index, NetworkLayer* layer);
	void AddBlock();
	void Compile();

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void ForwardOutput(const std::vector<Volume> &X); // прямое распространение
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

NetworkBlock::NetworkBlock(VolumeSize size, const std::string& type) : NetworkLayer(size) {
	this->type = GetMergeType(type);

	name = "block";
	info = std::string("merge: ") + GetMergeType();
}

NetworkBlock::NetworkBlock(VolumeSize size, const std::string& type, std::ifstream &f) : NetworkLayer(size) {
	this->type = GetMergeType(type);

	name = "block";
	info = std::string("merge: ") + GetMergeType();

	int totalBlocks;
	f >> totalBlocks;

	for (int i = 0; i < totalBlocks; i++) {
		AddBlock();

		int totalLayers;
		f >> totalLayers;

		for (int j = 0; j < totalLayers; j++) {
			std::string layerType;
			VolumeSize size;

			f >> layerType >> size;

			AddLayer(i, LoadLayer(size, layerType, f));
		}
	}

	Compile();
}

void NetworkBlock::MergeSum() {
	int total = outputSize.width * outputSize.height * outputSize.deep;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < output.size(); batchIndex++) {
		for (int i = 0; i < total; i++) {
			double sum = 0;

			for (size_t j = 0; j < blocks.size(); j++)
				sum += blocks[j][blocks[j].size() - 1]->GetOutput()[batchIndex][i];

			output[batchIndex][i] = sum;
		}
	}
}

void NetworkBlock::MergeStack() {
	int total = outputSize.width * outputSize.height * outputSize.deep;
	int current = 0;

	for (size_t index = 0; index < blocks.size(); index++) {
		NetworkLayer *layer = blocks[index][blocks[index].size() - 1];

		int deep = layer->GetOutputSize().deep;
		std::vector<Volume> &out = layer->GetOutput();

		#pragma omp parallel for collapse(4)
		for (size_t batchIndex = 0; batchIndex < output.size(); batchIndex++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					for (int k = 0; k < deep; k++)
						output[batchIndex](current + k, i, j) = out[batchIndex](k, i, j);

		current += deep;
	}
}

NetworkBlock::MergeType NetworkBlock::GetMergeType(const std::string& type) {
	if (type == "sum")
		return MergeType::Sum;

	if (type == "stack")
		return MergeType::Stack;
	
	throw std::runtime_error("Unknown merge type");
}

std::string NetworkBlock::GetMergeType() const {
	if (type == MergeType::Sum)
		return "sum";

	if (type == MergeType::Stack)
		return "stack";

	throw std::runtime_error("Unknown merge type");
}

void NetworkBlock::AddLayer(int index, NetworkLayer* layer) {
	if (index < 0 || index >= blocks.size())
		throw std::runtime_error("Invalid index for inserting layer");

	blocks[index].push_back(layer);
}

void NetworkBlock::AddBlock() {
	blocks.push_back(std::vector<NetworkLayer*>());
}

void NetworkBlock::Compile() {
	if (blocks.size() == 0)
		throw std::runtime_error("Blocks are empty");

	VolumeSize size = blocks[0][blocks[0].size() - 1]->GetOutputSize();

	for (size_t i = 0; i < blocks.size(); i++) {
		if (blocks[i].size() == 0)
			throw std::runtime_error("Block is empty");

		VolumeSize size2 = blocks[i][blocks[i].size() - 1]->GetOutputSize();

		if (size.width != size2.width || size.height != size2.height)
			throw std::runtime_error("Output layers have different sizes");

		if (type == MergeType::Sum && size.deep != size2.deep)
			throw std::runtime_error("Unable to make sum merge: output layers have different depth");
	}

	outputSize.height = size.height;
	outputSize.width = size.width;
	
	if (type == MergeType::Sum) {
		outputSize.deep = size.deep;
	}
	else {
		outputSize.deep = 0;

		for (size_t i = 0; i < blocks.size(); i++)
			outputSize.deep += blocks[i][blocks[i].size() - 1]->GetOutputSize().deep;
	}
}

// получение количества обучаемых параметров
int NetworkBlock::GetTrainableParams() const {
	int count = 0;

	for (size_t i = 0; i < blocks.size(); i++)
		for (size_t j = 0; j < blocks[i].size(); j++)
			count += blocks[i][j]->GetTrainableParams();

	return count;
}

// прямое распространение
void NetworkBlock::ForwardOutput(const std::vector<Volume> &X) {
	for (size_t i = 0; i < blocks.size(); i++) {
		blocks[i][0]->ForwardOutput(X);

		for (size_t j = 1; j < blocks[i].size(); j++)
			blocks[i][j]->ForwardOutput(blocks[i][j - 1]->GetOutput());
	}

	if (type == MergeType::Sum) {
		MergeSum();
	}
	else if (type == MergeType::Stack) {
		MergeStack();
	}
}

// прямое распространение
void NetworkBlock::Forward(const std::vector<Volume> &X) {
	for (size_t i = 0; i < blocks.size(); i++) {
		blocks[i][0]->Forward(X);

		for (size_t j = 1; j < blocks[i].size(); j++)
			blocks[i][j]->Forward(blocks[i][j - 1]->GetOutput());
	}

	if (type == MergeType::Sum) {
		MergeSum();
	}
	else if (type == MergeType::Stack) {
		MergeStack();
	}
}

// обратное распространение
void NetworkBlock::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (type == MergeType::Sum) {
		for (size_t i = 0; i < blocks.size(); i++) {
			int last = blocks[i].size() - 1;

			if (last == 0) {
				blocks[i][0]->Backward(dout, X, true);
			}
			else {
				blocks[i][last]->Backward(dout, blocks[i][last - 1]->GetOutput(), true);

				for (int j = last - 1; j > 0; j--)
					blocks[i][j]->Backward(blocks[i][j + 1]->GetDeltas(), blocks[i][j - 1]->GetOutput(), true);

				blocks[i][0]->Backward(blocks[i][1]->GetDeltas(), X, true);
			}
		}
	}
	else if (type == MergeType::Stack) {
		int current = 0;

		for (size_t index = 0; index < blocks.size(); index++) {
			NetworkLayer *layer = blocks[index][blocks[index].size() - 1];

			int deep = layer->GetOutputSize().deep;
			std::vector<Volume> douts(dout.size(), Volume(outputSize.width, outputSize.height, deep));

			#pragma omp parallel for collapse(4)
			for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
				for (int i = 0; i < outputSize.height; i++)
					for (int j = 0; j < outputSize.width; j++)
						for (int k = 0; k < deep; k++)
							douts[batchIndex](k, i, j) = dout[batchIndex](current + k, i, j);

			int last = blocks[index].size() - 1;

			if (last == 0) {
				blocks[index][0]->Backward(douts, X, true);
			}
			else {
				blocks[index][last]->Backward(douts, blocks[index][last - 1]->GetOutput(), true);

				for (int i = last - 1; i > 0; i--)
					blocks[index][i]->Backward(blocks[index][i + 1]->GetDeltas(), blocks[index][i - 1]->GetOutput(), true);

				blocks[index][0]->Backward(blocks[index][1]->GetDeltas(), X, true);
			}

			current += deep;
		}
	}

	if (!calc_dX)
		return;

	int totalInput = inputSize.height * inputSize.width * inputSize.deep;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int i = 0; i < totalInput; i++) {
			double sum = 0;

			for (size_t index = 0; index < blocks.size(); index++)
				sum += blocks[index][0]->GetDeltas()[batchIndex][i];

			dX[batchIndex][i] = sum;
		}
	}
}

// обновление весовых коэффициентов
void NetworkBlock::UpdateWeights(const Optimizer &optimizer) {
	for (size_t i = 0; i < blocks.size(); i++)
		for (size_t j = 0; j < blocks[i].size(); j++)
			blocks[i][j]->UpdateWeights(optimizer);
}

void NetworkBlock::ResetCache() {
	for (size_t i = 0; i < blocks.size(); i++)
		for (size_t j = 0; j < blocks[i].size(); j++)
			blocks[i][j]->ResetCache();
}

// сохранение слоя в файл
void NetworkBlock::Save(std::ofstream &f) const {
	f << "block " << inputSize << " " << GetMergeType() << " " << blocks.size() << std::endl;

	for (size_t i = 0; i < blocks.size(); i++) {
		f << blocks[i].size() << std::endl;

		for (size_t j = 0; j < blocks[i].size(); j++)
			blocks[i][j]->Save(f);
	}
}

// установка размера батча
void NetworkBlock::SetBatchSize(int batchSize) {
	output = std::vector<Volume>(batchSize, Volume(outputSize));
	dX = std::vector<Volume>(batchSize, Volume(inputSize));

	for (size_t i = 0; i < blocks.size(); i++)
		for (size_t j = 0; j < blocks[i].size(); j++)
			blocks[i][j]->SetBatchSize(batchSize);
}

// установка веса по индексу
void NetworkBlock::SetParam(int index, double weight) {
	int count = 0;

	for (size_t i = 0; i < blocks.size(); i++) {
		for (size_t j = 0; j < blocks[i].size(); j++) {
			int params = blocks[i][j]->GetTrainableParams();
			
			if (index < params) {
				blocks[i][j]->SetParam(index % params, weight);
				return;
			}		

			index -= params;
		}
	}
}

// получение веса по индексу
double NetworkBlock::GetParam(int index) const {
	int count = 0;

	for (size_t i = 0; i < blocks.size(); i++) {
		for (size_t j = 0; j < blocks[i].size(); j++) {
			int params = blocks[i][j]->GetTrainableParams();

			if (index < params)
				return blocks[i][j]->GetParam(index % params);

			index -= params;
		}
	}

	throw std::runtime_error("Error");
}

// получение градиента веса по индексу
double NetworkBlock::GetGradient(int index) const {
	int count = 0;

	for (size_t i = 0; i < blocks.size(); i++) {
		for (size_t j = 0; j < blocks[i].size(); j++) {
			int params = blocks[i][j]->GetTrainableParams();
			
			if (index < params)
				return blocks[i][j]->GetGradient(index % params);

			index -= params;
		}
	}

	throw std::runtime_error("Error");
}

// обнуление градиента веса по индексу
void NetworkBlock::ZeroGradient(int index) {
	int count = 0;

	for (size_t i = 0; i < blocks.size(); i++) {
		for (size_t j = 0; j < blocks[i].size(); j++) {
			int params = blocks[i][j]->GetTrainableParams();
			
			if (index < params) {
				blocks[i][j]->ZeroGradient(index % params);
				return;
			}		

			index -= params;
		}
	}
}
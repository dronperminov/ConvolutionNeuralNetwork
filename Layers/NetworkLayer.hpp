#pragma once

#include <iostream>
#include <iomanip>
#include <vector>

#include "../Entities/Volume.hpp"
#include "../Entities/Optimizers.hpp"

class NetworkLayer {
protected:
	GaussRandom random; // генератор случайных чисел с нормальным распределением

	VolumeSize inputSize;
	VolumeSize outputSize;

	std::string name; // имя слоя
	std::string info; // информация о слое

	std::vector<Volume> output;
	std::vector<Volume> dX;

public:
	NetworkLayer(int inputWidth, int inputHeight, int inputDeep, int outputWidth, int outputHeight, int outputDeep);

	VolumeSize GetOutputSize() const; // получение размера выхода слоя

	std::vector<Volume>& GetOutput();
	std::vector<Volume>& GetDeltas();

	void PrintConfig() const;
	
	virtual int GetTrainableParams() const = 0; // получение количества обучаемых параметров

	virtual void ForwardOutput(const std::vector<Volume> &X) { Forward(X); }; // прямое распространение
	virtual void Forward(const std::vector<Volume> &X) = 0; // прямое распространение
	virtual void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) = 0; // обратное распространение
	virtual void UpdateWeights(const Optimizer &optimizer) {} // обновление весовых коэффициентов
	
	virtual void ResetCache() {}
	virtual void Save(std::ofstream &f) const = 0; // сохранение слоя в файл
	virtual void SetBatchSize(int batchSize); // установка размера батча

	virtual void SetParam(int index, double weight) { throw std::runtime_error("Layer has no trainable parameters"); } // установка веса по индексу
	virtual double GetParam(int index) const { throw std::runtime_error("Layer has no trainable parameters"); } // получение веса по индексу
	virtual double GetGradient(int index) const { throw std::runtime_error("Layer has no trainable parameters"); } // получение градиента веса по индексу
	virtual void ZeroGradient(int index) { throw std::runtime_error("Layer has no trainable parameters"); } // обнуление градиента веса по индексу
};

NetworkLayer::NetworkLayer(int inputWidth, int inputHeight, int inputDeep, int outputWidth, int outputHeight, int outputDeep) {
	inputSize.width = inputWidth;
	inputSize.height = inputHeight;
	inputSize.deep = inputDeep;

	outputSize.width = outputWidth;
	outputSize.height = outputHeight;
	outputSize.deep = outputDeep;
}

// получение размера выхода слоя
VolumeSize NetworkLayer::GetOutputSize() const {
	return outputSize;
}

std::vector<Volume>& NetworkLayer::GetOutput() {
	return output;
}

std::vector<Volume>& NetworkLayer::GetDeltas() {
	return dX;
}

void NetworkLayer::PrintConfig() const {
	std::cout << "| " << std::left << std::setw(14) << name << " | ";
	std::cout << std::right << std::setw(12) << inputSize.ToString() << " | ";
	std::cout << std::setw(13) << outputSize.ToString() << " | ";
	std::cout << std::setw(12) << GetTrainableParams() << " | ";
	std::cout << info << std::endl; 
}

// установка размера батча
void NetworkLayer::SetBatchSize(int batchSize) {
	output = std::vector<Volume>(batchSize, Volume(outputSize));
	dX = std::vector<Volume>(batchSize, Volume(inputSize));
}
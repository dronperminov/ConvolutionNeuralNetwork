#pragma once

#include <iostream>
#include "../Entities/Volume.hpp"
#include "../Entities/GaussRandom.hpp"
#include "../Entities/Optimizers.hpp"

class NetworkLayer {
protected:
	VolumeSize inputSize; // размер входного объёма
	VolumeSize outputSize; // размер выходного объёма

	Volume output; // значение на выходе слоя
	Volume deltas; // дельты для обратного распространения ошибки
	GaussRandom random; // генератор случайных чисел с нормальным распределением

public:
	NetworkLayer(int inputWidth, int inputHeight, int inputDeep, int outputWidth, int outputHeight, int outputDeep);

	VolumeSize GetInputSize() const; // получение размера входного объёма
	VolumeSize GetOutputSize() const; // получение размера выходного объёма

	Volume& GetOutput(); // получение выхода слоя
	Volume& GetDeltas(); // получение дельт

	virtual void PrintConfig() const = 0;
	virtual int GetTrainableParams() const = 0; // получение количество обучаемых параметров

	virtual void ForwardOutput(const Volume &input) { Forward(input); } // прямое распространение для выхода
	virtual void Forward(const Volume& input) = 0; // прямое распространение
	virtual void Backward(Volume& prevDeltas) = 0; // обратное распространение
	virtual void UpdateWeights(const Optimizer& optimizer, const Volume& input) {} // обновление весовых коэффициентов
	
	virtual void ResetCache() {} // сброс параметров
	virtual void Save(std::ostream &f) = 0; // сохранение слоя в файл
};

NetworkLayer::NetworkLayer(int inputWidth, int inputHeight, int inputDeep, int outputWidth, int outputHeight, int outputDeep) :
	output(outputWidth, outputHeight, outputDeep),
	deltas(outputWidth, outputHeight, outputDeep)
{
	inputSize.width = inputWidth;
	inputSize.height = inputHeight;
	inputSize.deep = inputDeep;

	outputSize.width = outputWidth;
	outputSize.height = outputHeight;
	outputSize.deep = outputDeep;
}

// получение размера входного объёма
VolumeSize NetworkLayer::GetInputSize() const {
	return inputSize;
}

// получение размера выходного объёма
VolumeSize NetworkLayer::GetOutputSize() const {
	return outputSize;
}

// получение выхода слоя
Volume& NetworkLayer::GetOutput() {
	return output;
}

// получение дельт
Volume& NetworkLayer::GetDeltas() {
	return deltas;
}
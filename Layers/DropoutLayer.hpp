#pragma once

#include <iostream>
#include <iomanip>
#include <random>
#include "NetworkLayer.hpp"

class DropoutLayer : public NetworkLayer {
	double p; // вероятность исключения нейронов
	double q; // вероятность нейронов остаться (1 - p)
	int total; // общее количество элементов

	std::default_random_engine generator;
	std::binomial_distribution<int> distribution;

public:
	DropoutLayer(int width, int height, int deep, double p);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количество обучаемых параметров

	void ForwardOutput(const Volume& input); // прямое распространение
	void Forward(const Volume& input); // прямое распространение
	void Backward(Volume& prevDeltas); // обратное распространение
	void Save(std::ostream &f); // сохранение слоя в файл
};

DropoutLayer::DropoutLayer(int width, int height, int deep, double p) : NetworkLayer(width, height, deep, width, height, deep), distribution(1, 1 - p) {
	this->p = p;
	this->q = 1 - p;
	this->total = width * height * deep;
}

// вывод конфигурации
void DropoutLayer::PrintConfig() const {
	std::cout << "| dropout        | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << "           0 | ";
	std::cout << "p:" << p;
	std::cout << std::endl;
}

// получение количество обучаемых параметров
int DropoutLayer::GetTrainableParams() const {
	return 0;
}

// прямое распространение (этап тестирования)
void DropoutLayer::ForwardOutput(const Volume& input) {
	#pragma omp parallel for
	for (int i = 0; i < total; i++) {
		output[i] = input[i] * q; // умножаем на вероятность остаться
		deltas[i] = 1; // все дельты равны 1
	}
}

// прямое распространение (этап обучения)
void DropoutLayer::Forward(const Volume& input) {
	#pragma omp parallel for
	for (int i = 0; i < total; i++) {
		if (distribution(generator)) { // если нейрон остаётся включён
			output[i] = input[i]; // сохраняем входной сигнал
			deltas[i] = 1; // дельта равна 1
		}
		else { // иначе
			output[i] = 0; // нейрон игнорируется
			deltas[i] = 0; // дельта равна нулю
		}
	}
}

// обратное распространение
void DropoutLayer::Backward(Volume& prevDeltas) {
	#pragma omp parallel for
	for (int i = 0; i < total; i++)
		prevDeltas[i] *= deltas[i]; // пропускают только те нейроны, которые были включены при прямом распространении
}

// сохранение слоя в файл
void DropoutLayer::Save(std::ostream &f) {
	f << "dropout " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " " << p << std::endl;
}
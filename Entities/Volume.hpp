#pragma once

#include <iostream>
#include <vector>

#include "GaussRandom.hpp"

// размерность объёма
struct VolumeSize {
	int deep; // глубина
	int height; // высота
	int width; // ширина

	friend std::ostream& operator<<(std::ostream& os, VolumeSize size) {
		std::string s = "[" + std::to_string(size.width) + "x" + std::to_string(size.height) + "x" + std::to_string(size.deep) + "]";
		return os << s;
	}
};

// объём
class Volume {
	VolumeSize size; // размерность объёма
	std::vector<std::vector<std::vector<double>>> values; // значения объёма

public:
	Volume(int width, int height, int deep); // создание из размеров

	double& operator()(int d, int i, int j); // индексация
	double operator()(int d, int i, int j) const; // индексация

	int Deep() const; // получение глубины
	int Height() const; // получение высоты
	int Width() const; // получение ширины

	void FillRandom(GaussRandom& random, double dev, double mean = 0); // заполнение случайными числами
};

// создание из размеров
Volume::Volume(int width, int height, int deep) {
	size.width = width;
	size.height = height;
	size.deep = deep;

	values = std::vector<std::vector<std::vector<double>>>(deep, std::vector<std::vector<double>>(height, std::vector<double>(width, 0)));
}

// индексация
double& Volume::operator()(int d, int i, int j) {
	return values[d][i][j];
}

// индексация
double Volume::operator()(int d, int i, int j) const {
	return values[d][i][j];
}

// получение глубины
int Volume::Deep() const {
	return size.deep;
}

// получение высоты
int Volume::Height() const {
	return size.height;
}

// получение ширины
int Volume::Width() const {
	return size.width;
}

// заполнение случайными числами
void Volume::FillRandom(GaussRandom& random, double dev, double mean) {
	for (int d = 0; d < size.deep; d++)
		for (int i = 0; i < size.height; i++)
			for (int j = 0; j < size.width; j++)
				values[d][i][j] = random.Next(dev, mean);
}
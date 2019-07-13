#pragma once

#include <iostream>
#include <iomanip>
#include <vector>

#include "GaussRandom.hpp"

class Matrix {
	int n; // число строк
	int m; // число столбцов
	std::vector<std::vector<double>> values; // значения

public:
	Matrix(int n, int m); // конструктор из заданных размеров

	double& operator()(int i, int j); // индексация
	double operator()(int i, int j) const; // индексация

	void FillRandom(GaussRandom& random, double dev, double mean = 0); // заполнение случайными числами
};

// конструктор из заданных размеров
Matrix::Matrix(int n, int m) {
	this->n = n;
	this->m = m;

	values = std::vector<std::vector<double>>(n, std::vector<double>(m, 0));
}

// индексация
double& Matrix::operator()(int i, int j) {
	return values[i][j];
}

// индексация
double Matrix::operator()(int i, int j) const {
	return values[i][j];
}

// заполнение случайными числами
void Matrix::FillRandom(GaussRandom& random, double dev, double mean) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			values[i][j] = random.Next(dev, mean);
}
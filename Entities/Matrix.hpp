#pragma once

#include <iostream>
#include <iomanip>
#include <vector>

class Matrix {
	int n; // число строк
	int m; // число столбцов
	std::vector<std::vector<double>> values; // значения

public:
	Matrix(int n, int m); // конструктор из заданных размеров

	double& operator()(int i, int j); // индексация
	double operator()(int i, int j) const; // индексация

	friend std::ostream& operator<<(std::ostream& os, const Matrix &matrix);
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

std::ostream& operator<<(std::ostream& os, const Matrix &matrix) {
	for (int i = 0; i < matrix.n; i++) {
		for (int j = 0; j < matrix.m; j++)
			os << std::setw(5) << matrix.values[i][j] << " ";

		os << std::endl;
	}

	return os;
}
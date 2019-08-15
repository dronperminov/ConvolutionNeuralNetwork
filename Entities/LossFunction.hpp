#pragma once

#include "Volume.hpp"

enum class LossType {
	MSE, // среднеквадратичное отклонение
	CrossEntropy, // перекрёстная этнропия
	BinaryCrossEntropy, // бинарная перекрёстная этнропия
};

class LossFunction {
	LossType type; // тип функции ошибки

public:
	LossFunction(LossType type);

	double CalculateLoss(const Volume &y, const Volume &t, Volume &deltas) const; // вычисление значений функции ошибки и её производных
	double CalculateLoss(const Volume &y, const Volume &t) const; // вычисление значений функции ошибки
	
	double CalculateLoss(const std::vector<Volume> &y, const std::vector<Volume> &t, std::vector<Volume> &deltas) const; // вычисление значений функции ошибки и её производных для батча
	double CalculateLoss(const std::vector<Volume> &y, const std::vector<Volume> &t) const; // вычисление значений функции ошибки для батча
};

LossFunction::LossFunction(LossType type) {
	this->type = type;
}

// вычисление значений функции ошибки и её производных
double LossFunction::CalculateLoss(const Volume &y, const Volume &t, Volume &deltas) const {
	int total = deltas.Width() * deltas.Height() * deltas.Deep();
	double loss = 0;

	if (type == LossType::MSE) {
		for (int i = 0; i < total; i++) {
			double e = y[i] - t[i];

			deltas[i] = 2*e;
			loss += e*e;
		}
	}
	else if (type == LossType::CrossEntropy) {
		for (int i = 0; i < total; i++) {
			double yi = std::max(1e-7, std::min(1 - 1e-7, y[i]));
			double ti = std::max(1e-7, std::min(1 - 1e-7, t[i]));

			deltas[i] = -ti / yi;
			loss -= ti * log(yi);
		}
	}
	else if (type == LossType::BinaryCrossEntropy) {
		for (int i = 0; i < total; i++) {
			double yi = std::max(1e-7, std::min(1 - 1e-7, y[i]));
			double ti = std::max(1e-7, std::min(1 - 1e-7, t[i]));

			deltas[i] = (yi - ti) / (yi * (1 - yi));
			loss -= ti * log(yi) + (1 - ti) * log(1 - yi);;
		}
	}

	return loss;
}

// вычисление значений функции ошибки
double LossFunction::CalculateLoss(const Volume &y, const Volume &t) const {
	int total = y.Width() * y.Height() * y.Deep();
	double loss = 0;

	if (type == LossType::MSE) {
		for (int i = 0; i < total; i++) {
			double e = y[i] - t[i];

			loss += e*e;
		}
	}
	else if (type == LossType::CrossEntropy) {
		for (int i = 0; i < total; i++) {
			double yi = std::max(1e-7, std::min(1 - 1e-7, y[i]));
			double ti = std::max(1e-7, std::min(1 - 1e-7, t[i]));

			loss -= ti * log(yi);
		}
	}
	else if (type == LossType::BinaryCrossEntropy) {
		for (int i = 0; i < total; i++) {
			double yi = std::max(1e-7, std::min(1 - 1e-7, y[i]));
			double ti = std::max(1e-7, std::min(1 - 1e-7, t[i]));

			loss -= ti * log(yi) + (1 - ti) * log(1 - yi);;
		}
	}

	return loss;
}

// вычисление значений функции ошибки и её производных для батча
double LossFunction::CalculateLoss(const std::vector<Volume> &y, const std::vector<Volume> &t, std::vector<Volume> &deltas) const {
	double loss = 0;

	for (size_t i = 0; i < deltas.size(); i++)
		loss += CalculateLoss(y[i], t[i], deltas[i]);

	return loss;
}

// вычисление значений функции ошибки для батча
double LossFunction::CalculateLoss(const std::vector<Volume> &y, const std::vector<Volume> &t) const {
	double loss = 0;

	for (size_t i = 0; i < y.size(); i++)
		loss += CalculateLoss(y[i], t[i]);

	return loss;
}
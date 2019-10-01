#pragma once

#include "Volume.hpp"

class LossFunction {
protected:
	enum class LossType {
		MSE, // средняя квадратическая ошибка
		MAE, // средняя абсолютная ошибка
		CrossEntropy, // перекрёстная этнропия
		BinaryCrossEntropy, // бинарная перекрёстная этнропия
		Logcosh, // логарифм гиперболического косинуса
		Exp, // эскпонециальная ошибка
		User // пользовательская функция
	};

private:
	LossType type; // тип функции ошибки
	std::string name;

public:
	LossFunction(LossType type, const std::string &name = "");

	static LossFunction MSE();
	static LossFunction MAE();
	static LossFunction CrossEntropy();
	static LossFunction BinaryCrossEntropy();
	static LossFunction Logcosh();
	static LossFunction Exp();

	virtual double CalculateLoss(const Volume &y, const Volume &t, Volume &deltas) const; // вычисление значений функции ошибки и её производных
	virtual double CalculateLoss(const Volume &y, const Volume &t) const; // вычисление значений функции ошибки
	
	double CalculateLoss(const std::vector<Volume> &y, const std::vector<Volume> &t, std::vector<Volume> &deltas) const; // вычисление значений функции ошибки и её производных для батча
	double CalculateLoss(const std::vector<Volume> &y, const std::vector<Volume> &t) const; // вычисление значений функции ошибки для батча

	std::string GetName() const; // получение названия функции
};

LossFunction::LossFunction(LossType type, const std::string &name) {
	this->type = type;
	this->name = name;
}

LossFunction LossFunction::MSE() {
	return LossFunction(LossType::MSE, "MSE");
}

LossFunction LossFunction::MAE() {
	return LossFunction(LossType::MAE, "MAE");
}

LossFunction LossFunction::CrossEntropy() {
	return LossFunction(LossType::CrossEntropy, "Cross entropy");
}

LossFunction LossFunction::BinaryCrossEntropy() {
	return LossFunction(LossType::BinaryCrossEntropy, "Binary cross entropy");
}

LossFunction LossFunction::Logcosh() {
	return LossFunction(LossType::Logcosh, "Log cosh");
}

LossFunction LossFunction::Exp() {
	return LossFunction(LossType::Exp, "Exp");
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
	else if (type == LossType::MAE) {
		for (int i = 0; i < total; i++) {
			deltas[i] = y[i] > t[i] ? 1 : -1;
			loss += fabs(y[i] - t[i]);
		}
	}
	else if (type == LossType::CrossEntropy) {
		for (int i = 0; i < total; i++) {
			double yi = y[i];
			double ti = t[i];

			deltas[i] = -ti / yi;
			loss -= ti * log(yi);
		}
	}
	else if (type == LossType::BinaryCrossEntropy) {
		for (int i = 0; i < total; i++) {
			double yi = y[i];
			double ti = t[i];

			deltas[i] = (yi - ti) / (yi * (1 - yi));
			loss -= ti * log(yi) + (1 - ti) * log(1 - yi);;
		}
	}
	else if (type == LossType::Logcosh) {
		for (int i = 0; i < total; i++) {
			deltas[i] = tanh(y[i] - t[i]);
			loss += log(cosh(y[i] - t[i]));
		}
	}
	else if (type == LossType::Exp) {
		for (int i = 0; i < total; i++) {
			deltas[i] = exp(y[i] - t[i]);
			loss += exp(y[i] - t[i]);
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
	else if (type == LossType::MAE) {
		for (int i = 0; i < total; i++) {
			loss += fabs(y[i] - t[i]);
		}
	}
	else if (type == LossType::CrossEntropy) {
		for (int i = 0; i < total; i++) {
			double yi = std::max(1e-7, std::min(1 - 1e-7, y[i]));
			double ti = t[i];

			loss -= ti * log(yi);
		}
	}
	else if (type == LossType::BinaryCrossEntropy) {
		for (int i = 0; i < total; i++) {
			double yi = y[i];
			double ti = t[i];

			loss -= ti * log(yi) + (1 - ti) * log(1 - yi);;
		}
	}
	else if (type == LossType::Logcosh) {
		for (int i = 0; i < total; i++) {
			loss += log(cosh(y[i] - t[i]));
		}
	}
	else if (type == LossType::Exp) {
		for (int i = 0; i < total; i++) {
			loss += exp(y[i] - t[i]);
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

// получение названия функции
std::string LossFunction::GetName() const {
	return name;
}
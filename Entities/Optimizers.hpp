#pragma once

#include <cmath>

enum class OptimizerType {
	SGD, // стохастический градиентный спуск
	SGDm, // стохастический градиентный спуск с моментом
	Adagrad, // адаптивный градиент
	Adadelta, // адаптивный градиент со скользящим средним
	NAG // ускоренный градиент Нестерова
};

class Optimizer {
	OptimizerType type; // тип оптимизатора
	double learningRate; // скорость обучения
	double param; // параметр оптимизатора

	void (Optimizer::*update)(double, double&, double&) const;

	void UpdateSGD(double grad, double &dw, double &w) const; // обновление веса для стохастического градиентного спуска
	void UpdateSGDm(double grad, double &dw, double &w) const; // обновление веса для стохастического градиентного спуска с моментом
	void UpdateAdagrad(double grad, double &dw, double &w) const; // обновление веса для адаптивного градиента
	void UpdateAdadelta(double grad, double &dw, double &w) const; // обновление веса для адаптивного градиента со скользящим средним
	void UpdateNAG(double grad, double &dw, double &w) const;// обновление веса для ускоренного градиента Нестерова

	Optimizer(OptimizerType type, double learningRate, double param);

public:
	static Optimizer SGD(double learningRate); // стохастический градиентный спуск
	static Optimizer SGDm(double learningRate, double moment = 0.9); // стохастический градиентный спуск с моментом
	static Optimizer Adagrad(double learningRate); // адаптивный градиент
	static Optimizer Adadelta(double learningRate, double gamma = 0.9); // адаптивный градиент со скользящим средним
	static Optimizer NAG(double learningRate, double mu = 0.9); // ускоренный градиент Нестерова

	void Update(double grad, double &dw, double &w) const; // обновление весовых коэффициентов
	void Print() const; // вывод информации об алгоритме
};

Optimizer::Optimizer(OptimizerType type, double learningRate, double param) {
	this->type = type;
	this->learningRate = learningRate;
	this->param = param;

	if (type == OptimizerType::SGD) {
		this->update = UpdateSGD;
	}
	else if (type == OptimizerType::SGDm) {
		this->update = UpdateSGDm;
	}
	else if (type == OptimizerType::Adagrad) {
		this->update = UpdateAdagrad;
	}
	else if (type == OptimizerType::Adadelta) {
		this->update = UpdateAdadelta;
	}
	else if (type == OptimizerType::NAG) {
		this->update = UpdateNAG;
	}
	else
		throw std::runtime_error("Invalid optimizer type");
}

// стохастический градиентный спуск
Optimizer Optimizer::SGD(double learningRate) {
	return Optimizer(OptimizerType::SGD, learningRate, 0);
}

// стохастический градиентный спуск с моментом
Optimizer Optimizer::SGDm(double learningRate, double moment) {
	return Optimizer(OptimizerType::SGDm, learningRate, moment);
}

// адаптивный градиент
Optimizer Optimizer::Adagrad(double learningRate) {
	return Optimizer(OptimizerType::Adagrad, learningRate, 0);
}

// адаптивный градиент со скользящим средним
Optimizer Optimizer::Adadelta(double learningRate, double gamma) {
	return Optimizer(OptimizerType::Adadelta, learningRate, gamma);
}

// ускоренный градиент Нестерова
Optimizer Optimizer::NAG(double learningRate, double mu) {
	return Optimizer(OptimizerType::NAG, learningRate, mu);
}

// обновление веса для стохастического градиентного спуска
void Optimizer::UpdateSGD(double grad, double &dw, double &w) const {
	w -= learningRate * grad;
}

// обновление веса для стохастического градиентного спуска с моментом
void Optimizer::UpdateSGDm(double grad, double &dw, double &w) const {
	dw = param * dw + learningRate * grad;
	w -= dw;
}

// обновление веса для адаптивного градиента
void Optimizer::UpdateAdagrad(double grad, double &dw, double &w) const {
	dw += grad * grad;
	w -= learningRate * grad / sqrt(dw + 1e-6);
}

// обновление веса для адаптивного градиента со скользящим средним
void Optimizer::UpdateAdadelta(double grad, double &dw, double &w) const {
	dw = dw * param + (1 - param) * grad * grad;
	w -= learningRate * grad / sqrt(dw + 1e-6);
}

// обновление веса для ускоренного градиента Нестерова
void Optimizer::UpdateNAG(double grad, double &dw, double &w) const {
	double prev = dw;
	dw = param * dw - learningRate * grad;
	w += param * (dw - prev) + dw;
}

// обновление весовых коэффициентов
void Optimizer::Update(double grad, double &dw, double &w) const {
	(this->*update)(grad, dw, w);
}

// вывод информации об алгоритме
void Optimizer::Print() const {
	std::cout << "optimizer: ";
	if (type == OptimizerType::SGD) {
		std::cout << "SGD, ";
	}
	else if (type == OptimizerType::SGDm) {
		std::cout << "SGDm, moment: " << param << ", ";
	}
	else if (type == OptimizerType::Adagrad) {
		std::cout << "Adagrad, ";
	}
	else if (type == OptimizerType::Adadelta) {
		std::cout << "Adadelta, gamma: " << param << ", ";
	}
	else if (type == OptimizerType::NAG) {
		std::cout << "NAG, mu: " << param << ", ";
	}

	std::cout << "learning rate: " << learningRate;
}
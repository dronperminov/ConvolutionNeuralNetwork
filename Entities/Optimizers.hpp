#pragma once

#include <cmath>

enum class OptimizerType {
	SGD, // стохастический градиентный спуск
	SGDm, // стохастический градиентный спуск с моментом
	Adagrad, // адаптивный градиент
	RMSprop, 
	Adadelta, // адаптивный градиент со скользящим средним
	NAG, // ускоренный градиент Нестерова
	Adam // Адам
};

class Optimizer {
	OptimizerType type; // тип оптимизатора
	double learningRate; // скорость обучения
	double param; // параметр оптимизатора
	double param2; // второй параметр оптимизатора
	int epoch; // номер эпохи

	void (Optimizer::*update)(double, double&, double&, double&) const;

	void UpdateSGD(double grad, double &dw, double &dw2, double &w) const; // обновление веса для стохастического градиентного спуска
	void UpdateSGDm(double grad, double &dw, double &dw2, double &w) const; // обновление веса для стохастического градиентного спуска с моментом
	void UpdateAdagrad(double grad, double &dw, double &dw2, double &w) const; // обновление веса для адаптивного градиента
	void UpdateRMSprop(double grad, double &dw, double &dw2, double &w) const; // обновление веса для RMSprop
	void UpdateAdadelta(double grad, double &dw, double &dw2, double &w) const; // обновление веса для адаптивного градиента со скользящим средним
	void UpdateNAG(double grad, double &dw, double &dw2, double &w) const;// обновление веса для ускоренного градиента Нестерова
	void UpdateAdam(double grad, double &dw, double &dw2, double &w) const;// обновление веса для Адам

	Optimizer(OptimizerType type, double learningRate, double param, double param2);

public:
	static Optimizer SGD(double learningRate); // стохастический градиентный спуск
	static Optimizer SGDm(double learningRate, double moment = 0.9); // стохастический градиентный спуск с моментом
	static Optimizer Adagrad(double learningRate); // адаптивный градиент
	static Optimizer Adadelta(double gamma = 0.9); // адаптивный градиент со скользящим средним
	static Optimizer RMSprop(double learningRate, double beta = 0.9); // RMSprop
	static Optimizer NAG(double learningRate, double mu = 0.9); // ускоренный градиент Нестерова
	static Optimizer Adam(double learningRate, double beta1 = 0.9, double beta2 = 0.999); // Адам

	void Update(double grad, double &dw, double &dw2, double &w) const; // обновление весовых коэффициентов
	void Print() const; // вывод информации об алгоритме

	void SetEpoch(int epoch); // задание текущей эпохи
	void SetLearningRate(double learningRate); // изменение скорости обучения
	void ChangeLearningRate(double v); // изменение скорости обучения в v раз
};

Optimizer::Optimizer(OptimizerType type, double learningRate, double param, double param2) {
	this->type = type;
	this->learningRate = learningRate;
	this->param = param;
	this->param2 = param2;
	this->epoch = 1;

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
	else if (type == OptimizerType::Adam) {
		this->update = UpdateAdam;
	}
	else if (type == OptimizerType::RMSprop) {
		this->update = UpdateRMSprop;
	}
	else
		throw std::runtime_error("Invalid optimizer type");
}

// стохастический градиентный спуск
Optimizer Optimizer::SGD(double learningRate) {
	return Optimizer(OptimizerType::SGD, learningRate, 0, 0);
}

// стохастический градиентный спуск с моментом
Optimizer Optimizer::SGDm(double learningRate, double moment) {
	return Optimizer(OptimizerType::SGDm, learningRate, moment, 0);
}

// адаптивный градиент
Optimizer Optimizer::Adagrad(double learningRate) {
	return Optimizer(OptimizerType::Adagrad, learningRate, 0, 0);
}

// адаптивный градиент со скользящим средним
Optimizer Optimizer::Adadelta(double gamma) {
	return Optimizer(OptimizerType::Adadelta, 0, gamma, 0);
}

// RMSprop
Optimizer Optimizer::RMSprop(double learningRate, double beta) {
	return Optimizer(OptimizerType::RMSprop, learningRate, beta, 0);
}

// ускоренный градиент Нестерова
Optimizer Optimizer::NAG(double learningRate, double mu) {
	return Optimizer(OptimizerType::NAG, learningRate, mu, 0);
}

// Адам
Optimizer Optimizer::Adam(double learningRate, double beta1, double beta2) {
	return Optimizer(OptimizerType::Adam, learningRate, beta1, beta2);
}

// обновление веса для стохастического градиентного спуска
void Optimizer::UpdateSGD(double grad, double &dw, double &dw2, double &w) const {
	w -= learningRate * grad;
}

// обновление веса для стохастического градиентного спуска с моментом
void Optimizer::UpdateSGDm(double grad, double &v, double &dw2, double &w) const {
	v = param * v - learningRate * grad;
	w += v;
}

// обновление веса для адаптивного градиента
void Optimizer::UpdateAdagrad(double grad, double &dw, double &dw2, double &w) const {
	dw += grad * grad;
	w -= learningRate * grad / sqrt(dw + 1e-6);
}

// обновление веса для RMSprop
void Optimizer::UpdateRMSprop(double grad, double &dw, double &dw2, double &w) const {
	dw = dw * param + (1 - param) * grad * grad;
	w -= learningRate * grad / sqrt(dw + 1e-6);
}

// обновление веса для адаптивного градиента со скользящим средним
void Optimizer::UpdateAdadelta(double grad, double &Eg, double &Ex, double &w) const {
	Eg = param * Eg + (1 - param) * grad * grad;
	double dw = -(sqrt(Ex + 1e-6)) / sqrt(Eg + 1e-6) * grad;
	Ex = param * Ex + (1 - param) * dw * dw;
	w += dw;
}

// обновление веса для ускоренного градиента Нестерова
void Optimizer::UpdateNAG(double grad, double &v, double &dw2, double &w) const {
	double prev = v;
	v = param * v - learningRate * grad;
	w += param * (v - prev) + v;
}

// обновление веса для Адам
void Optimizer::UpdateAdam(double grad, double &m, double &v, double &w) const {
	m = param * m + (1 - param) * grad;
	v = param2 * v + (1 - param2) * grad * grad;
	double mt = m / (1 - pow(param, epoch));
	double vt = v / (1 - pow(param2, epoch));

	w -= learningRate * mt / (sqrt(vt) + 1e-8);
}

// обновление весовых коэффициентов
void Optimizer::Update(double grad, double &dw, double &dw2, double &w) const {
	(this->*update)(grad, dw, dw2, w);
}

// вывод информации об алгоритме
void Optimizer::Print() const {
	std::cout << "optimizer: ";
	if (type == OptimizerType::SGD) {
		std::cout << "SGD";
	}
	else if (type == OptimizerType::SGDm) {
		std::cout << "SGDm, moment: " << param;
	}
	else if (type == OptimizerType::Adagrad) {
		std::cout << "Adagrad";
	}
	else if (type == OptimizerType::Adadelta) {
		std::cout << "Adadelta, gamma: " << param;
	}
	else if (type == OptimizerType::RMSprop) {
		std::cout << "RMSprop, gamma: " << param;
	}
	else if (type == OptimizerType::NAG) {
		std::cout << "NAG, mu: " << param;
	}
	else if (type == OptimizerType::Adam) {
		std::cout << "Adam, beta1: " << param << ", beta2: " << param2;
	}

	if (type != OptimizerType::Adadelta) {
		std::cout << ", learning rate: " << learningRate;
	}
}

// задание текущей эпохи
void Optimizer::SetEpoch(int epoch) {
	this->epoch = epoch;
}

// изменение скорости обучения
void Optimizer::SetLearningRate(double learningRate) {
	this->learningRate = learningRate;
}

// изменение скорости обучения в v раз
void Optimizer::ChangeLearningRate(double v) {
	learningRate *= v;
}
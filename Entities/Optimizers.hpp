#pragma once

#include <cmath>

enum class OptimizerType {
	SGD, // стохастический градиентный спуск
	SGDm, // стохастический градиентный спуск с моментом
	Adagrad, // адаптивный градиент
	RMSprop, 
	Adadelta, // адаптивный градиент со скользящим средним
	NAG, // ускоренный градиент Нестерова
	Adam, // адаптивный момент
	AdaMax,
	Nadam,
	AMSgrad,
	AdaBound
};

class Optimizer {
	OptimizerType type; // тип оптимизатора
	double learningRate; // скорость обучения
	double param; // параметр оптимизатора
	double param2; // второй параметр оптимизатора
	double param3; // третий параметр оптимизатора
	double param4; // четвёртый параметр оптимизатора
	int epoch; // номер эпохи

	void (Optimizer::*update)(double, double&, double&, double&) const;

	void UpdateSGD(double grad, double &dw, double &dw2, double &w) const; // обновление веса для стохастического градиентного спуска
	void UpdateSGDm(double grad, double &dw, double &dw2, double &w) const; // обновление веса для стохастического градиентного спуска с моментом
	void UpdateAdagrad(double grad, double &dw, double &dw2, double &w) const; // обновление веса для адаптивного градиента
	void UpdateRMSprop(double grad, double &dw, double &dw2, double &w) const; // обновление веса для RMSprop
	void UpdateAdadelta(double grad, double &dw, double &dw2, double &w) const; // обновление веса для адаптивного градиента со скользящим средним
	void UpdateNAG(double grad, double &dw, double &dw2, double &w) const; // обновление веса для ускоренного градиента Нестерова
	void UpdateAdam(double grad, double &dw, double &dw2, double &w) const; // обновление веса для адаптивного момента
	void UpdateAdaMax(double grad, double &dw, double &dw2, double &w) const; // обновление веса для AdaMax
	void UpdateNadam(double grad, double &dw, double &dw2, double &w) const; // обновление веса для Nadam
	void UpdateAMSgrad(double grad, double &dw, double &dw2, double &w) const; // обновление веса для AMSgrad
	void UpdateAdaBound(double grad, double &dw, double &dw2, double &w) const; // обновление веса для AdaBound

	Optimizer(OptimizerType type, double learningRate, double param = 0, double param2 = 0, double param3 = 0, double param4 = 0);

public:
	static Optimizer SGD(double learningRate = 0.01); // стохастический градиентный спуск
	static Optimizer SGDm(double learningRate = 0.01, double moment = 0.9); // стохастический градиентный спуск с моментом
	static Optimizer Adagrad(double learningRate = 0.01); // адаптивный градиент
	static Optimizer Adadelta(double gamma = 0.9); // адаптивный градиент со скользящим средним
	static Optimizer RMSprop(double learningRate = 0.001, double beta = 0.9); // RMSprop
	static Optimizer NAG(double learningRate = 0.01, double mu = 0.9); // ускоренный градиент Нестерова
	static Optimizer Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999); // адаптивный момент
	static Optimizer AdaMax(double learningRate = 0.002, double beta1 = 0.9, double beta2 = 0.999); // AdaMax
	static Optimizer Nadam(double learningRate = 0.002, double beta1 = 0.9, double beta2 = 0.999); // Nadam
	static Optimizer AMSgrad(double learningRate = 0.002, double beta1 = 0.9, double beta2 = 0.999); // AMSgrad
	static Optimizer AdaBound(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double finalLearningRate = 0.1, double gamma = 1e-3); // AdaBound

	void Update(double grad, double &dw, double &dw2, double &w) const; // обновление весовых коэффициентов
	void Print() const; // вывод информации об алгоритме

	void SetEpoch(int epoch); // задание текущей эпохи
	void SetLearningRate(double learningRate); // изменение скорости обучения
	void ChangeLearningRate(double v); // изменение скорости обучения в v раз
};

Optimizer::Optimizer(OptimizerType type, double learningRate, double param, double param2, double param3, double param4) {
	this->type = type;
	this->learningRate = learningRate;

	this->param = param;
	this->param2 = param2;
	this->param3 = param3;
	this->param4 = param4;

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
	else if (type == OptimizerType::RMSprop) {
		this->update = UpdateRMSprop;
	}
	else if (type == OptimizerType::NAG) {
		this->update = UpdateNAG;
	}
	else if (type == OptimizerType::Adam) {
		this->update = UpdateAdam;
	}
	else if (type == OptimizerType::AdaMax) {
		this->update = UpdateAdaMax;
	}
	else if (type == OptimizerType::Nadam) {
		this->update = UpdateNadam;
	}
	else if (type == OptimizerType::AMSgrad) {
		this->update = UpdateAMSgrad;
	}
	else if (type == OptimizerType::AdaBound) {
		this->update = UpdateAdaBound;
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

// адаптивный момент
Optimizer Optimizer::Adam(double learningRate, double beta1, double beta2) {
	return Optimizer(OptimizerType::Adam, learningRate, beta1, beta2);
}

// AdaMax
Optimizer Optimizer::AdaMax(double learningRate, double beta1, double beta2) {
	return Optimizer(OptimizerType::AdaMax, learningRate, beta1, beta2);
}

// Nadam
Optimizer Optimizer::Nadam(double learningRate, double beta1, double beta2) {
	return Optimizer(OptimizerType::Nadam, learningRate, beta1, beta2);
}

// AMSgrad
Optimizer Optimizer::AMSgrad(double learningRate, double beta1, double beta2) {
	return Optimizer(OptimizerType::AMSgrad, learningRate, beta1, beta2);
}

// AdaBound
Optimizer Optimizer::AdaBound(double learningRate, double beta1, double beta2, double finalLearningRate, double gamma) {
	return Optimizer(OptimizerType::AdaBound, learningRate, beta1, beta2, finalLearningRate, gamma);
}

// обновление веса для стохастического градиентного спуска
void Optimizer::UpdateSGD(double grad, double &dw, double &dw2, double &w) const {
	w -= learningRate * grad;
}

// обновление веса для стохастического градиентного спуска с моментом
void Optimizer::UpdateSGDm(double grad, double &v, double &dw2, double &w) const {
	v = param * v + learningRate * grad;
	w -= v;
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

// обновление веса для адаптивного момента
void Optimizer::UpdateAdam(double grad, double &m, double &v, double &w) const {
	m = param * m + (1 - param) * grad;
	v = param2 * v + (1 - param2) * grad * grad;
	double mt = m / (1 - pow(param, epoch));
	double vt = v / (1 - pow(param2, epoch));

	w -= learningRate * mt / (sqrt(vt) + 1e-8);
}

// обновление веса для AdaMax
void Optimizer::UpdateAdaMax(double grad, double &V, double &S, double &w) const {
	V = param * V + (1 - param) * grad;
	S = std::max(param2 * S, fabs(grad));

	double vt = V / (1 - pow(param, epoch));

	w -= learningRate * vt / (S + 1e-8);
}

// обновление веса для Nadam
void Optimizer::UpdateNadam(double grad, double &V, double &S, double &w) const {
	double Vt1 = V / (1 - pow(param, epoch));

	V = param * V + (1 - param) * grad;
	S = param2 * S + (1 - param2) * grad * grad;

	double Vt = V / (1 - pow(param, epoch));
	double St = S / (1 - pow(param2, epoch));

	w -= learningRate * (param * Vt1 + (1 - param) / (1 - pow(param, epoch)) * grad) / (sqrt(St) + 1e-7);
}

// обновление веса для AMSgrad
void Optimizer::UpdateAMSgrad(double grad, double &V, double &S, double &w) const {
	V = param * V + (1 - param) * grad;
	double S0 = S;
	S = param2 * S + (1 - param2) * grad * grad;
	double St = std::max(S0, S);

	w -= learningRate * V / (sqrt(St) + 1e-7);
}

// обновление веса для AdaBound
void Optimizer::UpdateAdaBound(double grad, double &m, double &v, double &w) const {
	m = param * m + (1 - param) * grad;
	v = param2 * v + (1 - param2) * grad * grad;
	double mt = m / (1 - pow(param, epoch));
	double vt = v / (1 - pow(param2, epoch));

	double lower_bound = param4 * (1 - 1 / (param3 * epoch + 1));
	double upper_bound = param4 * (1 + 1 / (param3 * epoch));
	double step = learningRate / (sqrt(vt) + 1e-8);

	if (step < lower_bound) {
		step = lower_bound;
	}
	else if (step > upper_bound) {
		step = upper_bound;
	}

	w -= step * mt;
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
	else if (type == OptimizerType::AdaMax) {
		std::cout << "AdaMax, beta1: " << param << ", beta2: " << param2;
	}
	else if (type == OptimizerType::Nadam) {
		std::cout << "Nadam, beta1: " << param << ", beta2: " << param2;
	}
	else if (type == OptimizerType::AMSgrad) {
		std::cout << "AMSgrad, beta1: " << param << ", beta2: " << param2;
	}
	else if (type == OptimizerType::AdaBound) {
		std::cout << "AdaBound, beta1: " << param << ", beta2: " << param2 << ", gamma: " << param4 << ", finalLearningRate: " << param3;
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
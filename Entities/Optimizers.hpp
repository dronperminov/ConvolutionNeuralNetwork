#pragma once

#include <cmath>
#include <algorithm>
#include "ArgParser.hpp"

#define OPTIMIZER_PARAMS_COUNT 3

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
	AdaBound,
	Unknown
};

class Optimizer {
	OptimizerType type; // тип оптимизатора
	double learningRate; // скорость обучения
	double weightDecay; // коэффициент регуляризации
	double beta1; // параметр оптимизатора
	double beta2; // второй параметр оптимизатора
	double param3; // третий параметр оптимизатора
	double param4; // четвёртый параметр оптимизатора
	int epoch; // номер эпохи

	void UpdateSGD(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для стохастического градиентного спуска
	void UpdateSGDm(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для стохастического градиентного спуска с моментом
	void UpdateAdagrad(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для адаптивного градиента
	void UpdateRMSprop(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для RMSprop
	void UpdateAdadelta(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для адаптивного градиента со скользящим средним
	void UpdateNAG(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для ускоренного градиента Нестерова
	void UpdateAdam(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для адаптивного момента
	void UpdateAdaMax(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для AdaMax
	void UpdateNadam(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для Nadam
	void UpdateAMSgrad(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для AMSgrad
	void UpdateAdaBound(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление веса для AdaBound

	Optimizer(OptimizerType type, double learningRate, double weightDecay, double beta1 = 0, double beta2 = 0, double param3 = 0, double param4 = 0);

public:
	Optimizer(std::string config); // оптимизатор по текстовому описанию

	static Optimizer SGD(double learningRate = 0.01, double weightDecay = 0); // стохастический градиентный спуск
	static Optimizer SGDm(double learningRate = 0.01, double weightDecay = 0, double moment = 0.9); // стохастический градиентный спуск с моментом
	static Optimizer Adagrad(double learningRate = 0.01, double weightDecay = 0); // адаптивный градиент
	static Optimizer Adadelta(double weightDecay = 0, double gamma = 0.9); // адаптивный градиент со скользящим средним
	static Optimizer RMSprop(double learningRate = 0.001, double weightDecay = 0, double beta = 0.9); // RMSprop
	static Optimizer NAG(double learningRate = 0.01, double weightDecay = 0, double mu = 0.9); // ускоренный градиент Нестерова
	static Optimizer Adam(double learningRate = 0.001, double weightDecay = 0, double beta1 = 0.9, double beta2 = 0.999); // адаптивный момент
	static Optimizer AdaMax(double learningRate = 0.002, double weightDecay = 0, double beta1 = 0.9, double beta2 = 0.999); // AdaMax
	static Optimizer Nadam(double learningRate = 0.002, double weightDecay = 0, double beta1 = 0.9, double beta2 = 0.999); // Nadam
	static Optimizer AMSgrad(double learningRate = 0.002, double weightDecay = 0, double beta1 = 0.9, double beta2 = 0.999); // AMSgrad
	static Optimizer AdaBound(double learningRate = 0.001, double weightDecay = 0, double beta1 = 0.9, double beta2 = 0.999, double finalLearningRate = 0.1, double gamma = 1e-3); // AdaBound

	void Update(double grad, double &dw, double &dw2, double &dw3, double &w) const; // обновление весовых коэффициентов
	void Print() const; // вывод информации об алгоритме

	void SetEpoch(int epoch); // задание текущей эпохи
	void SetLearningRate(double learningRate); // изменение скорости обучения
	void SetWeightDecay(double weightDecay); // изменение коэффициента регуляризации

	void ChangeLearningRate(double v); // изменение скорости обучения в v раз
};

Optimizer::Optimizer(OptimizerType type, double learningRate, double weightDecay, double beta1, double beta2, double param3, double param4) {
	this->type = type;
	this->learningRate = learningRate;

	this->beta1 = beta1;
	this->beta2 = beta2;
	this->param3 = param3;
	this->param4 = param4;

	this->epoch = 1;
	this->weightDecay = weightDecay;
}

// оптимизатор по текстовому описанию
Optimizer::Optimizer(std::string config) {
	std::transform(config.begin(), config.end(), config.begin(), ::tolower);
	ArgParser parser(config);

	type = OptimizerType::Unknown; // изначально тип неизвестен
	weightDecay = 0;
	epoch = 1;

	beta1 = 0;
	beta2 = 0;
	param3 = 0;
	param4 = 0;

	if (parser["sgd"]) {
		type = OptimizerType::SGD;
		learningRate = 0.01;
	}
	else if (parser["sgdm"]) {
		type = OptimizerType::SGDm;

		learningRate = 0.01;
		beta1 = 0.9;
	}
	else if (parser["adagrad"]) {
		type = OptimizerType::Adagrad;
		learningRate = 0.01;
	}
	else if (parser["adadelta"]) {
		type = OptimizerType::Adadelta;

		learningRate = 0;
		beta1 = 0.9;
	}
	else if (parser["rmsprop"]) {
		type = OptimizerType::RMSprop;

		learningRate = 0.001;
		beta1 = 0.9;
	}
	else if (parser["nag"]) {
		type = OptimizerType::NAG;

		learningRate = 0.01;
		beta1 = 0.9;
	}
	else if (parser["adam"]) {
		type = OptimizerType::Adam;

		learningRate = 0.001;
		beta1 = 0.9;
		beta2 = 0.999;
	}
	else if (parser["adamax"]) {
		type = OptimizerType::AdaMax;

		learningRate = 0.002;
		beta1 = 0.9;
		beta2 = 0.999;
	}
	else if (parser["nadam"]) {
		type = OptimizerType::Nadam;

		learningRate = 0.002;
		beta1 = 0.9;
		beta2 = 0.999;
	}
	else if (parser["amsgrad"]) {
		type = OptimizerType::AMSgrad;

		learningRate = 0.002;
		beta1 = 0.9;
		beta2 = 0.999;
	}
	else if (parser["adabound"]) {
		type = OptimizerType::AdaBound;

		learningRate = 0.001;
		beta1 = 0.9;
		beta2 = 0.999;
		param3 = 0.1;
		param4 = 1e-3;
	}

	if (type == OptimizerType::Unknown)
		throw std::runtime_error("Unknown optimizer");

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "learningrate" || arg == "lr") {
			learningRate = std::stod(parser.Get(arg));
		}
		else if (arg == "weightdecay" || arg == "decay" || arg == "wd") {
			weightDecay = std::stod(parser.Get(arg));
		}
		else if (arg == "epoch") {
			epoch = std::stoi(parser.Get(arg));
		}
		else if (type == OptimizerType::SGDm && (arg == "moment" || arg == "m")) {
			beta1 = std::stod(parser.Get(arg));
		}
		else if (type == OptimizerType::Adadelta && arg == "gamma") {
			beta1 = std::stod(parser.Get(arg));
		}
		else if (type == OptimizerType::RMSprop && arg == "beta") {
			beta1 = std::stod(parser.Get(arg));
		}
		else if (type == OptimizerType::NAG && arg == "mu") {
			beta1 = std::stod(parser.Get(arg));
		}
		else if (type == OptimizerType::Adam || type == OptimizerType::AdaMax || type == OptimizerType::Nadam || type == OptimizerType::AMSgrad) {
			if (arg == "beta1") {
				beta1 = std::stod(parser.Get(arg));
			}
			else if (arg == "beta2") {
				beta2 = std::stod(parser.Get(arg));
			}
			else if (arg != "adam" && arg != "adamax" && arg != "nadam" && arg != "amsgrad")
				throw std::runtime_error("Invalid optimizer argument '" + arg + "'");
		}
		else if (type == OptimizerType::AdaBound) {
			if (arg == "beta1") {
				beta1 = std::stod(parser.Get(arg));
			}
			else if (arg == "beta2") {
				beta2 = std::stod(parser.Get(arg));
			}
			else if (arg == "finallearningRate" || arg == "finallr" || arg == "endlr") {
				param3 = std::stod(parser.Get(arg));
			}
			else if (arg == "gamma") {
				param4 = std::stod(parser.Get(arg));
			}
			else if (arg != "adabound")
				throw std::runtime_error("Invalid optimizer argument '" + arg + "'");
		}
		else if (arg != "sgd" && arg != "sgdm" && arg != "adagrad" && arg != "adadelta" && arg != "rmsprop" && arg != "nag" && arg != "adam" && arg != "adamax" && arg != "nadam" && arg != "amsgrad" && arg != "adabound")
			throw std::runtime_error("Invalid optimizer argument '" + arg + "'");
	}
}

// стохастический градиентный спуск
Optimizer Optimizer::SGD(double learningRate, double weightDecay) {
	return Optimizer(OptimizerType::SGD, learningRate, weightDecay, 0, 0);
}

// стохастический градиентный спуск с моментом
Optimizer Optimizer::SGDm(double learningRate, double weightDecay, double moment) {
	return Optimizer(OptimizerType::SGDm, learningRate, weightDecay, moment, 0);
}

// адаптивный градиент
Optimizer Optimizer::Adagrad(double learningRate, double weightDecay) {
	return Optimizer(OptimizerType::Adagrad, learningRate, weightDecay, 0, 0);
}

// адаптивный градиент со скользящим средним
Optimizer Optimizer::Adadelta(double weightDecay, double gamma) {
	return Optimizer(OptimizerType::Adadelta, 0, weightDecay, gamma, 0);
}

// RMSprop
Optimizer Optimizer::RMSprop(double learningRate, double weightDecay, double beta) {
	return Optimizer(OptimizerType::RMSprop, learningRate, weightDecay, beta, 0);
}

// ускоренный градиент Нестерова
Optimizer Optimizer::NAG(double learningRate, double weightDecay, double mu) {
	return Optimizer(OptimizerType::NAG, learningRate, weightDecay, mu, 0);
}

// адаптивный момент
Optimizer Optimizer::Adam(double learningRate, double weightDecay, double beta1, double beta2) {
	return Optimizer(OptimizerType::Adam, learningRate, weightDecay, beta1, beta2);
}

// AdaMax
Optimizer Optimizer::AdaMax(double learningRate, double weightDecay, double beta1, double beta2) {
	return Optimizer(OptimizerType::AdaMax, learningRate, weightDecay, beta1, beta2);
}

// Nadam
Optimizer Optimizer::Nadam(double learningRate, double weightDecay, double beta1, double beta2) {
	return Optimizer(OptimizerType::Nadam, learningRate, weightDecay, beta1, beta2);
}

// AMSgrad
Optimizer Optimizer::AMSgrad(double learningRate, double weightDecay, double beta1, double beta2) {
	return Optimizer(OptimizerType::AMSgrad, learningRate, weightDecay, beta1, beta2);
}

// AdaBound
Optimizer Optimizer::AdaBound(double learningRate, double weightDecay, double beta1, double beta2, double finalLearningRate, double gamma) {
	return Optimizer(OptimizerType::AdaBound, learningRate, weightDecay, beta1, beta2, finalLearningRate, gamma);
}

// обновление веса для стохастического градиентного спуска
void Optimizer::UpdateSGD(double grad, double &dw, double &dw2, double &dw3, double &w) const {
	w -= learningRate * grad;
}

// обновление веса для стохастического градиентного спуска с моментом
void Optimizer::UpdateSGDm(double grad, double &m, double &dw2, double &dw3, double &w) const {
	m = beta1 * m + learningRate * grad;
	w -= m;
}

// обновление веса для адаптивного градиента
void Optimizer::UpdateAdagrad(double grad, double &G, double &dw2, double &dw3, double &w) const {
	G += grad * grad;
	w -= learningRate * grad / sqrt(G + 1e-6);
}

// обновление веса для RMSprop
void Optimizer::UpdateRMSprop(double grad, double &dw, double &dw2, double &dw3, double &w) const {
	dw = dw * beta1 + (1 - beta1) * grad * grad;
	w -= learningRate * grad / sqrt(dw + 1e-6);
}

// обновление веса для адаптивного градиента со скользящим средним
void Optimizer::UpdateAdadelta(double grad, double &Eg, double &Ex, double &dw3, double &w) const {
	Eg = beta1 * Eg + (1 - beta1) * grad * grad;
	double dw = -(sqrt(Ex + 1e-6)) / sqrt(Eg + 1e-6) * grad;
	Ex = beta1 * Ex + (1 - beta1) * dw * dw;
	w += dw;
}

// обновление веса для ускоренного градиента Нестерова
void Optimizer::UpdateNAG(double grad, double &v, double &dw2, double &dw3, double &w) const {
	double prev = v;
	v = beta1 * v - learningRate * grad;
	w += beta1 * (v - prev) + v;
}

// обновление веса для адаптивного момента
void Optimizer::UpdateAdam(double grad, double &m, double &v, double &dw3, double &w) const {
	m = beta1 * m + (1 - beta1) * grad;
	v = beta2 * v + (1 - beta2) * grad * grad;

	double mt = m / (1 - pow(beta1, epoch));
	double vt = v / (1 - pow(beta2, epoch));

	w -= learningRate * mt / (sqrt(vt) + 1e-8);
}

// обновление веса для AdaMax
void Optimizer::UpdateAdaMax(double grad, double &m, double &v, double &dw3, double &w) const {
	m = beta1 * m + (1 - beta1) * grad;
	v = std::max(beta2 * v, fabs(grad));

	double mt = m / (1 - pow(beta1, epoch));

	w -= learningRate * mt / (v + 1e-8);
}

// обновление веса для Nadam
void Optimizer::UpdateNadam(double grad, double &m, double &v, double &dw3, double &w) const {
	double mt1 = m / (1 - pow(beta1, epoch));

	m = beta1 * m + (1 - beta1) * grad;
	v = beta2 * v + (1 - beta2) * grad * grad;

	double Vt = m / (1 - pow(beta1, epoch));
	double St = v / (1 - pow(beta2, epoch));

	w -= learningRate * (beta1 * mt1 + (1 - beta1) / (1 - pow(beta1, epoch)) * grad) / (sqrt(St) + 1e-7);
}

// обновление веса для AMSgrad
void Optimizer::UpdateAMSgrad(double grad, double &m, double &v, double &vt, double &w) const {
	m = beta1 * m + (1 - beta1) * grad;
	v = beta2 * v + (1 - beta2) * grad * grad;

	vt = std::max(v, vt);

	w -= learningRate * m / (sqrt(vt) + 1e-7);
}

// обновление веса для AdaBound
void Optimizer::UpdateAdaBound(double grad, double &m, double &v, double &dw3, double &w) const {
	m = beta1 * m + (1 - beta1) * grad;
	v = beta2 * v + (1 - beta2) * grad * grad;
	double mt = m / (1 - pow(beta1, epoch));
	double vt = v / (1 - pow(beta2, epoch));

	double lower_bound = param3 * (1 - 1 / (param4 * epoch + 1));
	double upper_bound = param3 * (1 + 1 / (param4 * epoch));
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
void Optimizer::Update(double grad, double &dw, double &dw2, double &dw3, double &w) const {
	w -= learningRate * weightDecay * w;

	switch (type) {
		case OptimizerType::SGD:
			UpdateSGD(grad, dw, dw2, dw3, w);
			break;
	
		case OptimizerType::SGDm:
			UpdateSGDm(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::Adagrad:
			UpdateAdagrad(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::Adadelta:
			UpdateAdadelta(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::RMSprop:
			UpdateRMSprop(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::NAG:
			UpdateNAG(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::Adam:
			UpdateAdam(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::AdaMax:
			UpdateAdaMax(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::Nadam:
			UpdateNadam(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::AMSgrad:
			UpdateAMSgrad(grad, dw, dw2, dw3, w);
			break;

		case OptimizerType::AdaBound:
			UpdateAdaBound(grad, dw, dw2, dw3, w);
			break;
	}

}

// вывод информации об алгоритме
void Optimizer::Print() const {
	std::cout << "optimizer: ";
	if (type == OptimizerType::SGD) {
		std::cout << "SGD";
	}
	else if (type == OptimizerType::SGDm) {
		std::cout << "SGDm, moment: " << beta1;
	}
	else if (type == OptimizerType::Adagrad) {
		std::cout << "Adagrad";
	}
	else if (type == OptimizerType::Adadelta) {
		std::cout << "Adadelta, gamma: " << beta1;
	}
	else if (type == OptimizerType::RMSprop) {
		std::cout << "RMSprop, gamma: " << beta1;
	}
	else if (type == OptimizerType::NAG) {
		std::cout << "NAG, mu: " << beta1;
	}
	else if (type == OptimizerType::Adam) {
		std::cout << "Adam, beta1: " << beta1 << ", beta2: " << beta2;
	}
	else if (type == OptimizerType::AdaMax) {
		std::cout << "AdaMax, beta1: " << beta1 << ", beta2: " << beta2;
	}
	else if (type == OptimizerType::Nadam) {
		std::cout << "Nadam, beta1: " << beta1 << ", beta2: " << beta2;
	}
	else if (type == OptimizerType::AMSgrad) {
		std::cout << "AMSgrad, beta1: " << beta1 << ", beta2: " << beta2;
	}
	else if (type == OptimizerType::AdaBound) {
		std::cout << "AdaBound, beta1: " << beta1 << ", beta2: " << beta2 << ", gamma: " << param4 << ", finalLearningRate: " << param3;
	}

	if (type != OptimizerType::Adadelta) {
		std::cout << ", learning rate: " << learningRate;
	}

	if (weightDecay != 0)
		std::cout << ", weight decay: " << weightDecay;
}

// задание текущей эпохи
void Optimizer::SetEpoch(int epoch) {
	this->epoch = epoch;
}

// изменение скорости обучения
void Optimizer::SetLearningRate(double learningRate) {
	this->learningRate = learningRate;
}

// изменение коэффициента регуляризации
void Optimizer::SetWeightDecay(double weightDecay) {
	this->weightDecay = weightDecay;
}


// изменение скорости обучения в v раз
void Optimizer::ChangeLearningRate(double v) {
	learningRate *= v;
}
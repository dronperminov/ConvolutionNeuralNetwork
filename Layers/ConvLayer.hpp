#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class ConvLayer : public NetworkLayer {
	std::vector<Volume> W; // фильтры
	std::vector<Volume> dW; // градиенты фильтров
	std::vector<std::vector<Volume>> paramsW; // параметры фильтров

	std::vector<double> b; // смещения
	std::vector<double> db; // градиенты смещений
	std::vector<std::vector<double>> paramsb; // параметры смещений

	int P; // дополнение нулями
	int S; // шаг свёртки

	int fc; // количество фильтров
	int fs; // размер фильтров
	int fd; // глубина фильтров

	void InitParams(); // инициализация параметров для обучения
	void InitWeights(); // инициализация весовых коэффициентов
	void LoadWeights(std::ifstream &f); // считывание весовых коэффициентов из файла

public:
	ConvLayer(VolumeSize size, int fc, int fs, int P, int S);
	ConvLayer(VolumeSize size, int fc, int fs, int P, int S, std::ifstream &f);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение
	void UpdateWeights(const Optimizer &optimizer, bool trainable); // обновление весовых коэффициентов

	void ResetCache(); // сброс параметров
	void Save(std::ofstream &f) const; // сохранение слоя в файл
	void SetBatchSize(int batchSize); // установка размера батча

	void SetWeight(int index, int i, int j, int k, double weight);
	void SetBias(int index, double bias);

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index) const; // получение градиента веса по индексу
	void ZeroGradient(int index); // обнуление градиента веса по индексу
};

ConvLayer::ConvLayer(VolumeSize size, int fc, int fs, int P, int S) : NetworkLayer(size, (size.width - fs + 2 * P) / S + 1, (size.height - fs + 2 * P) / S + 1, fc) {
	this->P = P;
	this->S = S;

	this->fc = fc;
	this->fs = fs;
	this->fd = size.deep;

	name = "conv";
	info = std::to_string(fc) + " filters [" + std::to_string(fs) + "x" + std::to_string(fs) + "x" + std::to_string(fd) + "] P:" + std::to_string(P) + " S:" + std::to_string(S);

	for (int i = 0; i < fc; i++) {
		W.push_back(Volume(fs, fs, fd));
		dW.push_back(Volume(fs, fs, fd));

		b.push_back(0);
		db.push_back(0);
	}

	InitParams();
	InitWeights();
}

ConvLayer::ConvLayer(VolumeSize size, int fc, int fs, int P, int S, std::ifstream &f) : NetworkLayer(size, (size.width - fs + 2 * P) / S + 1, (size.height - fs + 2 * P) / S + 1, fc) {
	this->P = P;
	this->S = S;

	this->fc = fc;
	this->fs = fs;
	this->fd = size.deep;

	name = "conv";
	info = std::to_string(fc) + " filters [" + std::to_string(fs) + "x" + std::to_string(fs) + "x" + std::to_string(fd) + "] P:" + std::to_string(P) + " S:" + std::to_string(S);

	for (int i = 0; i < fc; i++) {
		W.push_back(Volume(fs, fs, fd));
		dW.push_back(Volume(fs, fs, fd));

		b.push_back(0);
		db.push_back(0);
	}

	InitParams();
	LoadWeights(f);
}

// инициализация параметров для обучения
void ConvLayer::InitParams() {
	for (int i = 0; i < OPTIMIZER_PARAMS_COUNT; i++) {
		paramsW.push_back(std::vector<Volume>(fc, Volume(fs, fs, fd)));
		paramsb.push_back(std::vector<double>(fc, 0));
	}
}

// инициализация весовых коэффициентов
void ConvLayer::InitWeights() {
	for (int index = 0; index < fc; index++) {
		for (int i = 0; i < fs; i++)
			for (int j = 0; j < fs; j++)
				for (int k = 0; k < fd; k++)
					W[index](k, i, j) = random.Next(sqrt(2.0 / (fs*fs*fd)), 0);

		b[index] = 0.01;
	}
}

// считывание весовых коэффициентов из файла
void ConvLayer::LoadWeights(std::ifstream &f) {
	for (int index = 0; index < fc; index++) {
		for (int d = 0; d < fd; d++)
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					f >> W[index](d, i, j);

		f >> b[index];
	}
}

// получение количество обучаемых параметров
int ConvLayer::GetTrainableParams() const {
	return fc * (fs * fs * fd + 1);
}

// прямое распространение
void ConvLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(4)
	for (size_t n = 0; n < X.size(); n++) {
		for (int f = 0; f < fc; f++) {
			for (int i = 0; i < outputSize.height; i++) {
				for (int j = 0; j < outputSize.width; j++) {
					double sum = b[f];

					for (int k = 0; k < fs; k++) {
						int i0 = S * i + k - P;

						if (i0 < 0 || i0 >= inputSize.height)
							continue;

						for (int l = 0; l < fs; l++) {
							int j0 = S * j + l - P;

							if (j0 < 0 || j0 >= inputSize.width)
								continue;

							for (int c = 0; c < fd; c++)
								sum += X[n](c, i0, j0) * W[f](c, k, l);
						}
					}

					output[n](f, i, j) = sum;
				}
			}
		}
	}
}

// обратное распространение
void ConvLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	VolumeSize size;

	size.height = S * (outputSize.height - 1) + 1;
	size.width = S * (outputSize.width - 1) + 1;
	size.deep = outputSize.deep;

	std::vector<Volume> deltas(dout.size(), Volume(size));

	for (size_t n = 0; n < dout.size(); n++) {
		for (int d = 0; d < size.deep; d++)
			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					deltas[n](d, i * S, j * S) = dout[n](d, i, j);
	}

	#pragma omp parallel for
	for (int f = 0; f < fc; f++) {
		for (size_t n = 0; n < dout.size(); n++) {
			for (int k = 0; k < size.height; k++) {
				for (int l = 0; l < size.width; l++) {
					double delta = deltas[n](f, k, l); // значение фильтра

					for (int i = 0; i < fs; i++) {
						int i0 = i + k - P;

						if (i0 < 0 || i0 >= inputSize.height)
							continue;

						for (int j = 0; j < fs; j++) {
							int j0 = j + l - P;

							if (j0 < 0 || j0 >= inputSize.width)
								continue;

							for (int c = 0; c < fd; c++)
								dW[f](c, i, j) += delta * X[n](c, i0, j0);
						}
					}

					db[f] += delta;
				}
			}
		}
	}

	if (calc_dX) {
		int pad = fs - 1 - P;

		#pragma omp parallel for collapse(3)
		for (size_t n = 0; n < dout.size(); n++) {
			for (int i = 0; i < inputSize.height; i++) {
				for (int j = 0; j < inputSize.width; j++) {
					for (int c = 0; c < fd; c++) {
						double sum = 0;

						for (int k = 0; k < fs; k++) {
							int i0 = i+k-pad;

							if (i0 < 0 || i0 >= size.height)
								continue;

							for (int l = 0; l < fs; l++) {
								int j0 = j+l-pad;

								if (j0 < 0 || j0 >= size.width)
									continue;

								for (int f = 0; f < fc; f++)
									sum += W[f](c, fs - 1 - k, fs - 1 - l) * deltas[n](f, i0, j0);
							}
						}

						dX[n](c, i, j) = sum;
					}
				}
			}
		}
	}
}

// обновление весовых коэффициентов
void ConvLayer::UpdateWeights(const Optimizer &optimizer, bool trainable) {
	int batchSize = output.size();
	int total = fd * fs * fs;

	#pragma omp parallel for
	for (int index = 0; index < fc; index++) {
		for (int i = 0; i < total; i++) {
			if (trainable)
				optimizer.Update(dW[index][i] / batchSize, paramsW[0][index][i], paramsW[1][index][i], paramsW[2][index][i], W[index][i]);

			dW[index][i] = 0;
		}

		if (trainable)
			optimizer.Update(db[index] / batchSize, paramsb[0][index], paramsb[1][index], paramsb[2][index], b[index]);

		db[index] = 0;
	}
}

// сброс параметров
void ConvLayer::ResetCache() {
	int total = fd * fs * fs;

	for (int index = 0; index < OPTIMIZER_PARAMS_COUNT; index++) {
		for (int i = 0; i < fc; i++) {
			for (int j = 0; j < total; j++)
				paramsW[index][i][j] = 0;

			paramsb[index][i] = 0;
		}
	}
}

// сохранение слоя в файл
void ConvLayer::Save(std::ofstream &f) const {
	f << "conv " << inputSize << " ";
	f << fs << " " << fc << " " << P << " " << S << std::endl;

	for (int index = 0; index < fc; index++) {
		for (int d = 0; d < fd; d++)
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					f << std::setprecision(15) << W[index](d, i, j) << " ";

		f << std::setprecision(15) << b[index] << std::endl;
	}
}

// установка размера батча
void ConvLayer::SetBatchSize(int batchSize) {
	output = std::vector<Volume>(batchSize, Volume(outputSize));
	dX = std::vector<Volume>(batchSize, Volume(inputSize));
}

void ConvLayer::SetWeight(int index, int i, int j, int k, double weight) {
	W[index](i, j, k) = weight;
}

void ConvLayer::SetBias(int index, double bias) {
	b[index] = bias;
}

// установка веса по индексу
void ConvLayer::SetParam(int index, double weight) {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1) {
		b[findex] = weight;
	}
	else {
		W[findex][windex] = weight;
	}
}

// получение веса по индексу
double ConvLayer::GetParam(int index) const {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1)
		return b[findex];

	return W[findex][windex];
}

// получение градиента веса по индексу
double ConvLayer::GetGradient(int index) const {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1)
		return db[findex];

	return dW[findex][windex];
}

// обнуление градиента веса по индексу
void ConvLayer::ZeroGradient(int index) {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1)
		db[findex] = 0;
	else
		dW[findex][windex] = 0;
}
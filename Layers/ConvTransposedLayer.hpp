#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class ConvTransposedLayer : public NetworkLayer {
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
	ConvTransposedLayer(VolumeSize size, int fc, int fs, int P, int S);
	ConvTransposedLayer(VolumeSize size, int fc, int fs, int P, int S, std::ifstream &f);

	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение
	void UpdateWeights(const Optimizer &optimizer, bool trainable); // обновление весовых коэффициентов

	void ResetCache(); // сброс параметров
	void Save(std::ofstream &f) const; // сохранение слоя в файл

	void SetWeight(int index, int i, int j, int k, double weight);
	void SetBias(int index, double bias);

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index) const; // получение градиента веса по индексу
	void ZeroGradient(int index); // обнуление градиента веса по индексу
};

ConvTransposedLayer::ConvTransposedLayer(VolumeSize size, int fc, int fs, int P, int S) : NetworkLayer(size, S * (size.width - 1) + fs - 2 * P, S * (size.height - 1) + fs - 2 * P, fc) {
	this->P = P;
	this->S = S;

	this->fc = fc;
	this->fs = fs;
	this->fd = size.deep;

	name = "conv transposed";
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

ConvTransposedLayer::ConvTransposedLayer(VolumeSize size, int fc, int fs, int P, int S, std::ifstream &f) : NetworkLayer(size, S * (size.width - 1) + fs - 2 * P, S * (size.height - 1) + fs - 2 * P, fc) {
	this->P = P;
	this->S = S;

	this->fc = fc;
	this->fs = fs;
	this->fd = size.deep;

	name = "conv transposed";
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
void ConvTransposedLayer::InitParams() {
	for (int i = 0; i < OPTIMIZER_PARAMS_COUNT; i++) {
		paramsW.push_back(std::vector<Volume>(fc, Volume(fs, fs, fd)));
		paramsb.push_back(std::vector<double>(fc, 0));
	}
}

// инициализация весовых коэффициентов
void ConvTransposedLayer::InitWeights() {
	for (int index = 0; index < fc; index++) {
		for (int i = 0; i < fs; i++)
			for (int j = 0; j < fs; j++)
				for (int k = 0; k < fd; k++)
					W[index](k, i, j) = random.Next(sqrt(2.0 / (fs*fs*fd)), 0);

		b[index] = 0.01;
	}
}

// считывание весовых коэффициентов из файла
void ConvTransposedLayer::LoadWeights(std::ifstream &f) {
	for (int index = 0; index < fc; index++) {
		for (int d = 0; d < fd; d++)
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					f >> W[index](d, i, j);

		f >> b[index];
	}
}

// получение количество обучаемых параметров
int ConvTransposedLayer::GetTrainableParams() const {
	return fc * (fs * fs * fd + 1);
}

// прямое распространение
void ConvTransposedLayer::Forward(const std::vector<Volume> &X) {
	VolumeSize size;

	size.height = S * (inputSize.height - 1) + 1;
	size.width = S * (inputSize.width - 1) + 1;
	size.deep = inputSize.deep;

	std::vector<Volume> input(X.size(), Volume(size));

	#pragma omp parallel for collapse(4)
	for (size_t n = 0; n < X.size(); n++)
		for (int d = 0; d < size.deep; d++)
			for (int i = 0; i < inputSize.height; i++)
				for (int j = 0; j < inputSize.width; j++)
					input[n](d, i * S, j * S) = X[n](d, i, j);

	int pad = fs - 1 - P;

	#pragma omp parallel for collapse(3)
	for (size_t n = 0; n < X.size(); n++) {
		for (int i = 0; i < outputSize.height; i++) {
			for (int j = 0; j < outputSize.width; j++) {
				for (int f = 0; f < fc; f++) {
					double sum = b[f];

					for (int k = 0; k < fs; k++) {
						int y = i+k-pad;

						if (y < 0 || y >= size.height)
							continue;

						for (int l = 0; l < fs; l++) {
							int x = j+l-pad;

							if (x < 0 || x >= size.width)
								continue;

							for (int c = 0; c < fd; c++)
								sum += W[f](c, fs - 1 - k, fs - 1 - l) * input[n](c, y, x);
						}
					}

					output[n](f, i, j) = sum;
				}
			}
		}
	}
}

// обратное распространение
void ConvTransposedLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	#pragma omp parallel for
	for (int f = 0; f < fc; f++) {
		for (size_t n = 0; n < dout.size(); n++) {
			for (int y = 0; y < inputSize.height; y++) {
				for (int x = 0; x < inputSize.width; x++) {
					for (int i = 0; i < fs; i++) {
						int k = i + y * S - P;
						
						if (k < 0 || k >= outputSize.height)
							continue;

						for (int j = 0; j < fs; j++) {
							int l = j + x * S - P;

							if (l < 0 || l >= outputSize.width)
								continue;

							for (int c = 0; c < fd; c++)
								dW[f](c, i, j) += dout[n](f, k, l) * X[n](c, y, x);
						}
					}

				}
			}
			
			for (int k = 0; k < outputSize.height; k++)
				for (int l = 0; l < outputSize.width; l++)
					db[f] += dout[n](f, k, l);
		}		
	}

	if (calc_dX) {
		#pragma omp parallel for collapse(4)
		for (size_t n = 0; n < dout.size(); n++) {
			for (int c = 0; c < fd; c++) {
				for (int i = 0; i < inputSize.height; i++) {
					for (int j = 0; j < inputSize.width; j++) {
						double sum = 0;

						for (int k = 0; k < fs; k++) {
							int y = S * i + k - P;

							if (y < 0 || y >= outputSize.height)
								continue;

							for (int l = 0; l < fs; l++) {
								int x = S * j + l - P;

								if (x < 0 || x >= outputSize.width)
									continue;

								for (int f = 0; f < fc; f++)
									sum += dout[n](f, y, x) * W[f](c, k, l);
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
void ConvTransposedLayer::UpdateWeights(const Optimizer &optimizer, bool trainable) {
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
void ConvTransposedLayer::ResetCache() {
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
void ConvTransposedLayer::Save(std::ofstream &f) const {
	f << "convtransposed " << inputSize << " ";
	f << fs << " " << fc << " " << P << " " << S << std::endl;

	for (int index = 0; index < fc; index++) {
		for (int d = 0; d < fd; d++)
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					f << std::setprecision(15) << W[index](d, i, j) << " ";

		f << std::setprecision(15) << b[index] << std::endl;
	}
}

void ConvTransposedLayer::SetWeight(int index, int i, int j, int k, double weight) {
	W[index](i, j, k) = weight;
}

void ConvTransposedLayer::SetBias(int index, double bias) {
	b[index] = bias;
}

// установка веса по индексу
void ConvTransposedLayer::SetParam(int index, double weight) {
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
double ConvTransposedLayer::GetParam(int index) const {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1)
		return b[findex];

	return W[findex][windex];
}

// получение градиента веса по индексу
double ConvTransposedLayer::GetGradient(int index) const {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1)
		return db[findex];

	return dW[findex][windex];
}

// обнуление градиента веса по индексу
void ConvTransposedLayer::ZeroGradient(int index) {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1)
		db[findex] = 0;
	else
		dW[findex][windex] = 0;
}
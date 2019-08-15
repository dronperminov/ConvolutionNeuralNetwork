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

	std::vector<Volume> df;

	int P; // дополнение нулями
	int S; // шаг свёртки

	int fc; // количество фильтров
	int fs; // размер фильтров
	int fd; // глубина фильтров

	void InitParams(); // инициализация параметров для обучения
	void InitWeights(); // инициализация весовых коэффициентов
	void LoadWeights(std::ifstream &f); // считывание весовых коэффициентов из файла

public:
	ConvLayer(int width, int height, int deep, int fc, int fs, int P, int S);
	ConvLayer(int width, int height, int deep, int fc, int fs, int P, int S, std::ifstream &f);

	void PrintConfig() const; // вывод конфигурации
	int GetTrainableParams() const; // получение количества обучаемых параметров

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение
	void UpdateWeights(const Optimizer &optimizer); // обновление весовых коэффициентов

	void ResetCache(); // сброс параметров
	void Save(std::ofstream &f) const; // сохранение слоя в файл

	void SetWeight(int index, int i, int j, int k, double weight);
	void SetBias(int index, double bias);

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index) const; // получение градиента веса по индексу
	void ZeroGradient(int index); // обнуление градиента веса по индексу
};

ConvLayer::ConvLayer(int width, int height, int deep, int fc, int fs, int P, int S) : NetworkLayer(width, height, deep, (width - fs + 2 * P) / S + 1, (height - fs + 2 * P) / S + 1, fc) {
	if ((width - fs + 2 * P) % S != 0 || (height - fs + 2 * P) % S != 0)
		throw std::runtime_error("Invalid params of ConvLayer. Unable to convolve");

	this->P = P;
	this->S = S;

	this->fc = fc;
	this->fs = fs;
	this->fd = deep;

	for (int i = 0; i < fc; i++) {
		W.push_back(Volume(fs, fs, fd));
		dW.push_back(Volume(fs, fs, fd));

		b.push_back(0);
		db.push_back(0);
	}

	InitParams();
	InitWeights();
}

ConvLayer::ConvLayer(int width, int height, int deep, int fc, int fs, int P, int S, std::ifstream &f) : NetworkLayer(width, height, deep, (width - fs + 2 * P) / S + 1, (height - fs + 2 * P) / S + 1, fc) {
	if ((width - fs + 2 * P) % S != 0 || (height - fs + 2 * P) % S != 0)
		throw std::runtime_error("Invalid params of ConvLayer. Unable to convolve");

	this->P = P;
	this->S = S;

	this->fc = fc;
	this->fs = fs;
	this->fd = deep;

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
					W[index](k, i, j) = random.Next(sqrt(2.0 / (fs*fs)), 0);

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

void ConvLayer::PrintConfig() const {
	std::cout << "| convolution    | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << std::setw(12) << (fc * (fs*fs*fd + 1)) << " | ";
	std::cout << fc << " filters [" << fs << "x" << fs << "x" << fd << "] P:" << P << " S:" << S << std::endl;
}

// получение количество обучаемых параметров
int ConvLayer::GetTrainableParams() const {
	return fc * (fs * fs * fd + 1);
}

// прямое распространение
void ConvLayer::Forward(const std::vector<Volume> &X) {
	output = std::vector<Volume>(X.size(), Volume(outputSize));
	df = std::vector<Volume>(X.size(), Volume(outputSize));

	// выполняем свёртку с каждым фильтром
	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int index = 0; index < fc; index++) {
			for (int y = 0, y0 = -P; y < outputSize.height; y++, y0 += S) {
				int imin = std::max(-y0, 0);
				int imax = std::min(fs, inputSize.height - y0);

				for (int x = 0, x0 = -P; x < outputSize.width; x++, x0 += S) {
					int jmin = std::max(-x0, 0);
					int jmax = std::min(fs, inputSize.width - x0);

					double sum = b[index]; // значение элемента ij результирующей матрицы

					// проходимся по всем значениям фильтра
					for (int i = imin; i < imax; i++) {
						int i0 = y0 + i;

						for (int j = jmin; j < jmax; j++) {
							int j0 = x0 + j;

							for (int k = 0; k < fd; k++) {
								double weight = W[index](k, i, j); // значение фильтра
								double value = X[batchIndex](k, i0, j0); // значение входного объёма

								sum += weight * value; // прибавляем взвешенное произведение
							}

						}
					}

					// записываем значение в матрицу
					if (sum > 0) {
						output[batchIndex](index, y, x) = sum;
						df[batchIndex](index, y, x) = 1;
					}
					else {
						output[batchIndex](index, y, x) = 0;
						df[batchIndex](index, y, x) = 0;
					}
				}
			}
		}
	}
}

// обратное распространение
void ConvLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (calc_dX) {
		dX = std::vector<Volume>(dout.size(), Volume(inputSize));
		int P = fs - 1 - this->P;

		#pragma omp parallel for collapse(2)
		for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
			for (int d = 0; d < inputSize.deep; d++) {
				for (int y = 0, y0 = -P; y < inputSize.height; y++, y0 += S) {
					int imin = std::max(-y0, 0);
					int imax = std::min(fs, outputSize.height - y0);

					for (int x = 0, x0 = -P; x < inputSize.width; x++, x0 += S) {
						int jmin = std::max(-x0, 0);
						int jmax = std::min(fs, outputSize.width - x0);

						double sum = 0; // значение элемента ij результирующей матрицы

						for (int i = imin; i < imax; i++) {
							int i0 = y0 + i;

							for (int j = jmin; j < jmax; j++) {
								int j0 = x0 + j;

								for (int k = 0; k < fc; k++) {
									double weight = W[k](d, fs - 1 - i, fs - 1 - j); // значение фильтра
									double value = dout[batchIndex](k, i0, j0) * df[batchIndex](k, i0, j0); // значение входного объёма

									sum += weight * value; // прибавляем взвешенное произведение
								}

							}
						}

						dX[batchIndex](d, y, x) = sum;
					}
				}
			}
		}
	}

	#pragma omp parallel for collapse(2) 
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int index = 0; index < fc; index++) {
			for (int y = 0, y0 = -P; y < fs; y++, y0 += S) {
				int imin = std::max(-y0, 0);
				int imax = std::min(outputSize.height, inputSize.height - y0);

				for (int x = 0, x0 = -P; x < fs; x++, x0 += S) {
					int jmin = std::max(-x0, 0);
					int jmax = std::min(outputSize.width, inputSize.width - x0);

					for (int d = 0; d < fd; d++) {
						double sum = 0;

						for (int i = imin; i < imax; i++) {
							int i0 = y0 + i;

							for (int j = jmin; j < jmax; j++) {
								int j0 = x0 + j;

								double weight = dout[batchIndex](index, i, j) * df[batchIndex](index, i, j); // значение фильтра
								double value = X[batchIndex](d, i0, j0); // значение входного объёма

								sum += weight * value; // прибавляем взвешенное произведение
							}
						}

						#pragma omp atomic
						dW[index](d, y, x) += sum;
					}
				}
			}

			for (int i = 0; i < outputSize.height; i++)
				for (int j = 0; j < outputSize.width; j++)
					#pragma omp atomic
					db[index] += dout[batchIndex](index, i, j) * df[batchIndex](index, i, j);
		}
	}
}

// обновление весовых коэффициентов
void ConvLayer::UpdateWeights(const Optimizer &optimizer) {
	int batchSize = output.size();
	int total = fd * fs * fs;

	#pragma omp parallel for
	for (int index = 0; index < fc; index++) {
		for (int i = 0; i < total; i++) {
			optimizer.Update(dW[index][i] / batchSize, paramsW[0][index][i], paramsW[1][index][i], paramsW[2][index][i], W[index][i]);
			dW[index][i] = 0;
		}

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
	f << "conv " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " ";
	f << fs << " " << fc << " " << P << " " << S << std::endl;

	for (int index = 0; index < fc; index++) {
		for (int d = 0; d < fd; d++)
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					f << std::setprecision(15) << W[index](d, i, j) << " ";

		f << std::setprecision(15) << b[index] << std::endl;
	}
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
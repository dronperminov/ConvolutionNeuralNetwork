#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "NetworkLayer.hpp"

enum class Padding {
	Full,
	Same,
	Valid
};

class ConvLayer : public NetworkLayer {
	std::vector<Volume> filters; // фильтры
	std::vector<Volume> dws; // изменения фильтров
	std::vector<Volume> dwws; // дополнительные изменения фильтров
	std::vector<Volume> gradFilters; // градиенты фильтров

	std::vector<double> biases; // смещения
	std::vector<double> db; // изменения смещений
	std::vector<double> dbb; // дополнительные изменения смещений
	std::vector<double> gradBiases; // градиенты смещений

	int P; // дополнение нулями
	int S; // шаг свёртки

	int fc; // количество фильтров
	int fs; // размер фильтров
	int fd; // глубина фильтров

public:
	ConvLayer(int width, int height, int deep, int fc, int fs, int P = 0, int S = 1);
	ConvLayer(int width, int height, int deep, int fc, int fs, int P, int S, std::ifstream &f);

	void PrintConfig() const;
	int GetTrainableParams() const; // получение количество обучаемых параметров
	
	void Forward(const Volume& input); // прямое распространение
	void Backward(Volume& prevDeltas); // обратное распространение
	void UpdateWeights(const Optimizer& optimizer, const Volume& input); // обновление весовых коэффициентов
	
	void CalculateGradients(const Volume &input); // вычисление градиентов
	void UpdateWeights(const Optimizer& optimizer, int batchSize); // обновление весовых коэффициентов

	void ResetCache(); // сброс параметров
	void Save(std::ostream &f); // сохранение слоя в файл

	void SetFilter(int index, int d, int i, int j, double weight); // установка веса фильтра
	void SetBias(int index, double bias); // установка веса смещения

	void SetParam(int index, double weight); // установка веса по индексу
	double GetParam(int index) const; // получение веса по индексу
	double GetGradient(int index, const Volume &input) const; // получение градиента веса по индексу
};

ConvLayer::ConvLayer(int width, int height, int deep, int fc, int fs, int P, int S) :
	NetworkLayer(width, height, deep, (width - fs + 2 * P) / S + 1, (height - fs + 2 * P) / S + 1, fc) {
	
	if ((width - fs + 2 * P) % S != 0 || (height - fs + 2 * P) % S != 0)
		throw std::runtime_error("Invalid params of ConvLayer. Unable to convolve");

	this->fc = fc;
	this->fs = fs;
	this->fd = deep;
	this->P = P;
	this->S = S;

	for (int i = 0; i < fc; i++) {
		filters.push_back(Volume(fs, fs, fd));
		dws.push_back(Volume(fs, fs, fd));
		dwws.push_back(Volume(fs, fs, fd));
		gradFilters.push_back(Volume(fs, fs, fd));

		biases.push_back(0);
		db.push_back(0);
		dbb.push_back(0);
		gradBiases.push_back(0);

		filters[i].FillRandom(random, sqrt(2.0 / (fs * fs * fd)));
		biases[i] = 0.01;
	}
}

ConvLayer::ConvLayer(int width, int height, int deep, int fc, int fs, int P, int S, std::ifstream &f) :
	NetworkLayer(width, height, deep, (width - fs + 2 * P) / S + 1, (height - fs + 2 * P) / S + 1, fc) {
	
	if ((width - fs + 2 * P) % S != 0 || (height - fs + 2 * P) % S != 0)
		throw std::runtime_error("Invalid params of ConvLayer. Unable to convolve");

	this->fc = fc;
	this->fs = fs;
	this->fd = deep;
	this->P = P;
	this->S = S;

	for (int index = 0; index < fc; index++) {
		filters.push_back(Volume(fs, fs, fd));
		dws.push_back(Volume(fs, fs, fd));
		dwws.push_back(Volume(fs, fs, fd));
		gradFilters.push_back(Volume(fs, fs, fd));
		
		biases.push_back(0);
		db.push_back(0);
		dbb.push_back(0);
		gradBiases.push_back(0);

		std::string filter;
		f >> filter;

		if (filter != "filter")
			throw std::runtime_error("Invalid convolution layer description");

		for (int d = 0; d < fd; d++)
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					f >> filters[index](d, i, j);

		f >> biases[index];
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
void ConvLayer::Forward(const Volume& input) {
	// выполняем свёртку с каждым фильтром
	#pragma omp parallel for
	for (int index = 0; index < fc; index++) {
		for (int y = 0, y0 = -P; y < outputSize.height; y++, y0 += S) {
			int imin = std::max(-y0, 0);
			int imax = std::min(fs, inputSize.height - y0);

			for (int x = 0, x0 = -P; x < outputSize.width; x++, x0 += S) {
				int jmin = std::max(-x0, 0);
				int jmax = std::min(fs, inputSize.width - x0);

				double sum = biases[index]; // значение элемента ij результирующей матрицы

				// проходимся по всем значениям фильтра
				for (int i = imin; i < imax; i++) {
					int i0 = y0 + i;

					for (int j = jmin; j < jmax; j++) {
						int j0 = x0 + j;

						for (int k = 0; k < fd; k++) {
							double weight = filters[index](k, i, j); // значение фильтра
							double value = input(k, i0, j0); // значение входного объёма

							sum += weight * value; // прибавляем взвешенное произведение
						}

					}
				}

				// записываем значение в матрицу
				if (sum > 0) {
					output(index, y, x) = sum;
					deltas(index, y, x) = 1;
				}
				else {
					output(index, y, x) = 0;
					deltas(index, y, x) = 0;
				}
			}
		}
	}
}

// обратное распространение
void ConvLayer::Backward(Volume& prevDeltas) {
	int P = fs - 1 - this->P;

	#pragma omp parallel for
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
							double weight = filters[k](d, fs - 1 - i, fs - 1 - j); // значение фильтра
							double value = deltas(k, i0, j0); // значение входного объёма

							sum += weight * value; // прибавляем взвешенное произведение
						}

					}
				}

				prevDeltas(d, y, x) *= sum;
			}
		}
	}
}

// обновление весовых коэффициентов
void ConvLayer::UpdateWeights(const Optimizer& optimizer, const Volume& input) {
	#pragma omp parallel for
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

							double weight = deltas(index, i, j); // значение фильтра
							double value = input(d, i0, j0); // значение входного объёма

							sum += weight * value; // прибавляем взвешенное произведение
						}
					}

					optimizer.Update(sum, dws[index](d, y, x), dwws[index](d, y, x), filters[index](d, y, x));
				}
			}
		}

		double dbi = 0;
		
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < outputSize.height; i++)
			for (int j = 0; j < outputSize.width; j++)
				dbi += deltas(index, i, j);

		optimizer.Update(dbi, db[index], dbb[index], biases[index]);
	}
}

// вычисление градиентов
void ConvLayer::CalculateGradients(const Volume &input) {
	#pragma omp parallel for
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

							double weight = deltas(index, i, j); // значение фильтра
							double value = input(d, i0, j0); // значение входного объёма

							sum += weight * value; // прибавляем взвешенное произведение
						}
					}

					gradFilters[index](d, y, x) += sum;
				}
			}
		}

		#pragma omp parallel for collapse(2)
		for (int i = 0; i < outputSize.height; i++)
			for (int j = 0; j < outputSize.width; j++)
				gradBiases[index] += deltas(index, i, j);
	}
}

// обновление весовых коэффициентов
void ConvLayer::UpdateWeights(const Optimizer& optimizer, int batchSize) {
	#pragma omp parallel for
	for (int index = 0; index < fc; index++) {
		for (int d = 0; d < fd; d++) {
			for (int i = 0; i < fs; i++) {
				for (int j = 0; j < fs; j++) {
					optimizer.Update(gradFilters[index](d, i, j) / batchSize, dws[index](d, i, j), dwws[index](d, i, j), filters[index](d, i, j));
					gradFilters[index](d, i, j) = 0;
				}
			}
		}

		optimizer.Update(gradBiases[index] / batchSize, db[index], dbb[index], biases[index]);
		gradBiases[index] = 0;
	}
}

// сброс параметров
void ConvLayer::ResetCache() {
	#pragma omp parallel for
	for (int index = 0; index < fc; index++) {
		for (int k = 0; k < fd; k++) {
			for (int i = 0; i < fs; i++) {
				for (int j = 0; j < fs; j++) {
					dws[index](k, i, j) = 0;
					dwws[index](k, i, j) = 0;
				}
			}
		}

		db[index] = 0;
		dbb[index] = 0;
	}
}

// сохранение слоя в файл
void ConvLayer::Save(std::ostream &f) {
	f << "conv " << inputSize.width << " " << inputSize.height << " " << inputSize.deep << " ";
	f << fs << " " << fc << " " << P << " " << S << std::endl;

	for (int index = 0; index < fc; index++) {
		f << "filter";

		for (int d = 0; d < fd; d++)
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					f << " " << std::setprecision(15) << filters[index](d, i, j);

		f << " " << std::setprecision(15) << biases[index] << std::endl;
	}
}

// установка веса фильтра
void ConvLayer::SetFilter(int index, int d, int i, int j, double weight) {
	filters[index](d, i, j) = weight;
}

// установка веса смещения
void ConvLayer::SetBias(int index, double bias) {
	biases[index] = bias;
}

// установка веса по индексу
void ConvLayer::SetParam(int index, double weight) {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1) {
		biases[findex] = weight;
	}
	else {
		filters[findex][windex] = weight;
	}
}

// получение веса по индексу
double ConvLayer::GetParam(int index) const {
	int params = fs * fs * fd + 1;
	int findex = index / params;
	int windex = index % params;

	if (windex == params - 1)
		return biases[findex];

	return filters[findex][windex];
}

// получение градиента веса по индексу
double ConvLayer::GetGradient(int gradIndex, const Volume &input) const {
	int params = fs * fs * fd + 1;
	int index = gradIndex / params;
	int windex = gradIndex % params;

	if (windex == params - 1) {
		double gradBias = 0;

		for (int i = 0; i < outputSize.height; i++)
			for (int j = 0; j < outputSize.width; j++)
				gradBias += deltas(index, i, j);

		return gradBias;
	}
	
	Volume gradFilter(fs, fs, fd);

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

						double weight = deltas(index, i, j); // значение фильтра
						double value = input(d, i0, j0); // значение входного объёма

						sum += weight * value; // прибавляем взвешенное произведение
					}
				}

				gradFilter(d, y, x) += sum;
			}
		}

	}

	return gradFilter[windex];
}
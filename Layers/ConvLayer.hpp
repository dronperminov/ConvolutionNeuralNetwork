#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "NetworkLayer.hpp"

class ConvLayer : public NetworkLayer {
	std::vector<Volume> filters; // фильтры
	std::vector<Volume> dws; // изменения фильтров
	std::vector<double> biases; // смещения
	std::vector<double> db; // изменения смещений

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
	void UpdateWeights(const Optimizer& optimizer); // обновление весовых коэффициентов
	
	void ResetCache(); // сброс параметров
	void Save(std::ostream &f); // сохранение слоя в файл

	void SetFilter(int index, int d, int i, int j, double weight); // установка веса фильтра
	void SetBias(int index, double bias); // установка веса смещения
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

		biases.push_back(0);
		db.push_back(0);

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
		
		biases.push_back(0);
		db.push_back(0);

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
	std::cout << "|     Convolution layer    | ";
	std::cout << std::setw(12) << inputSize << " | ";
	std::cout << std::setw(13) << outputSize << " | ";
	std::cout << std::setw(12) << (fc * (fs*fs + 1)) << " | ";
	std::cout << fc << " filters [" << fs << "x" << fs << "x" << fd << "] P:" << P << " S:" << S << std::endl;
}

// получение количество обучаемых параметров
int ConvLayer::GetTrainableParams() const {
	return fc * (fs * fs + 1);
}

// прямое распространение
void ConvLayer::Forward(const Volume& input) {
	#pragma omp parallel for collapse(3)
	for (int d = 0; d < inputSize.deep; d++)
		for (int i = 0; i < inputSize.height; i++)
			for (int j = 0; j < inputSize.width; j++)
				this->input(d, i, j) = input(d, i, j);

	// выполняем свёртку с каждым фильтром
	#pragma omp parallel for
	for (int index = 0; index < fc; index++) {
		for (int y = 0, y0 = -P; y < outputSize.height; y++, y0 += S) {
			for (int x = 0, x0 = -P; x < outputSize.width; x++, x0 += S) {
				double sum = biases[index]; // значение элемента ij результирующей матрицы

				// проходимся по всем значениям фильтра
				for (int i = 0; i < fs; i++) {
					int i0 = y0 + i;

					if (i0 < 0 || i0 >= inputSize.height)
						continue;

					for (int j = 0; j < fs; j++) {
						int j0 = x0 + j;

						if (j0 < 0 || j0 >= inputSize.width)
							continue;

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
	int P = this->P + fs - 1;

	#pragma omp parallel for
	for (int d = 0; d < inputSize.deep; d++) {
		for (int y = 0, y0 = -P; y < inputSize.height; y++, y0 += S) {
			for (int x = 0, x0 = -P; x < inputSize.width; x++, x0 += S) {
				double sum = 0; // значение элемента ij результирующей матрицы

				for (int i = 0; i < fs; i++) {
					int i0 = y0 + i;

					if (i0 < 0 || i0 >= outputSize.height)
						continue;

					for (int j = 0; j < fs; j++) {
						int j0 = x0 + j;

						if (j0 < 0 || j0 >= outputSize.width)
							continue;

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
void ConvLayer::UpdateWeights(const Optimizer& optimizer) {
	#pragma omp parallel for
	for (int index = 0; index < fc; index++) {
		for (int d = 0; d < fd; d++) {
			for (int y = 0, y0 = -P; y < fs; y++, y0 += S) {
				for (int x = 0, x0 = -P; x < fs; x++, x0 += S) {
					double sum = 0;

					for (int i = 0; i < outputSize.height; i++) {
						int i0 = y0 + i;

						if (i0 < 0 || i0 >= inputSize.height)
							continue;

						for (int j = 0; j < outputSize.width; j++) {
							int j0 = x0 + j;

							// если находимся внутри исходного объёма
							if (j0 >= 0 && j0 < inputSize.width) {
								double weight = deltas(index, i, j); // значение фильтра
								double value = input(d, i0, j0); // значение входного объёма

								sum += weight * value; // прибавляем взвешенное произведение
							}
						}
					}

					optimizer.Update(sum, dws[index](d, y, x), filters[index](d, y, x));
				}
			}
		}

		double dbi = 0;

		for (int i = 0; i < outputSize.height; i++)
			for (int j = 0; j < outputSize.width; j++)
				dbi += deltas(index, i, j);

		optimizer.Update(dbi, db[index], biases[index]);
	}
}

// сброс параметров
void ConvLayer::ResetCache() {
	#pragma omp parallel for
	for (int index = 0; index < fc; index++) {
		for (int k = 0; k < fd; k++)
			for (int i = 0; i < fs; i++)
				for (int j = 0; j < fs; j++)
					dws[index](k, i, j) = 0;

		db[index] = 0;
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
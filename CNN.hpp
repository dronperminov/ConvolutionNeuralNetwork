#pragma once

#include <iostream>
#include <vector>
#include <chrono>

#include "Entities/ArgParser.hpp"
#include "Entities/TimeSpan.hpp"
#include "Layers/NetworkLayer.hpp"
#include "Layers/ConvLayer.hpp"
#include "Layers/MaxPoolingLayer.hpp"
#include "Layers/FlattenLayer.hpp"
#include "Layers/FullConnectedLayer.hpp"
#include "Layers/DropoutLayer.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

enum class ErrorType {
	MSE, // среднеквадратичное отклонение
	CrossEntropy // перекрёстная этнропия
};

class CNN {
	VolumeSize inputSize; // размер входного объёма
	VolumeSize outputSize; // размер выходного объёма

	std::vector<NetworkLayer *> layers; // слои сети

	double (CNN::*cost)(const Volume&, const Volume&); // указатель на функцию стоимости

	double MSE(const Volume& y, const Volume& t); // среднеквадратичное отклонение
	double CrossEntropy(const Volume &y, const Volume &t); // перекрёстная энтропия

public:
	CNN(int width, int height, int deep); // инициализация сети
	CNN(const std::string& path); // загрузка сети из файла

	Volume& GetOutput(const Volume& input); // получение ответа сети

	void PringConfig() const; // вывод конфигурации

	void AddLayer(const std::string& layerConf); // добавление слоя по текстовому описанию
	double Train(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, int maxEpochs, const Optimizer &optimizer, ErrorType errorType = ErrorType::MSE, int logPeriod = 100);
	void Save(const std::string &path); // сохранение сети в файл
};

// среднеквадратичное отклонение
double CNN::MSE(const Volume& y, const Volume& t) {
	double error = 0;
	int last = layers.size() - 1;

	#pragma omp for collapse(3)
	for (int d = 0; d < outputSize.deep; d++) {
		for (int i = 0; i < outputSize.height; i++) {
			for (int j = 0; j < outputSize.width; j++) {
				double e = y(d, i, j) - t(d, i, j); // находим разность между выходом сети и обучающим примером
				layers[last]->GetDeltas()(d, i, j) *= e; // умножаем на неё дельту выходного слоя
				error += e * e; // прибавляем квадрат разности
			}
		}
	}

	return error;
}

// перекрётсная энтропия
double CNN::CrossEntropy(const Volume &y, const Volume &t) {
	double error = 0;
	size_t last = layers.size() - 1;

	for (int d = 0; d < outputSize.deep; d++) {
		for (int i = 0; i < outputSize.height; i++) {
			for (int j = 0; j < outputSize.width; j++) {
				double yi = y(d, i, j);
				double ti = t(d, i, j);

				error -= ti * log(yi + 1e-14) + (1 - ti) * log(1 - yi + 1e-14);

				layers[last]->GetDeltas()(d, i, j) *= (yi - ti) / (yi * (1 - yi));
			}
		}
	}

	return error;
}

// инициализация сети
CNN::CNN(int width, int height, int deep) {
	inputSize.width = width;
	inputSize.height = height;
	inputSize.deep = deep;
}

// загрузка сети из файла
CNN::CNN(const std::string& path) {
	std::ifstream f(path.c_str());

	if (!f)
		throw std::runtime_error("Unable to open file with model");

	f >> inputSize.width >> inputSize.height >> inputSize.deep;

	std::string layerType;
	NetworkLayer *layer = nullptr;

	while (f >> layerType) {
		if (layerType == "conv" || layerType == "convolution") {
			int w, h, d;
			int fc, fs, P, S;

			f >> w >> h >> d;
			f >> fs >> fc >> P >> S;
			
			layer = new ConvLayer(w, h, d, fc, fs, P, S, f);
		}
		else if (layerType == "maxpool" || layerType == "maxpooling") {
			int w, h, d, scale;

			f >> w >> h >> d >> scale;
			layer = new MaxPoolingLayer(w, h, d, scale);
		}
		else if (layerType == "flatten") {
			int w, h, d;

			f >> w >> h >> d;
			layer = new FlattenLayer(w, h, d);
		}
		else if (layerType == "fc" || layerType == "fullconnected") {
			int inputs;
			int outputs;
			std::string type;

			f >> inputs >> outputs >> type;

			layer = new FullConnectedLayer(inputs, outputs, type, f);
		}
		else if (layerType == "dropout") {
			int w, h, d;
			double p;

			f >> w >> h >> d >> p;
			layer = new DropoutLayer(w, h, d, p);
		}
		else
			throw std::runtime_error("Invalid layer type");

		layers.push_back(layer);
	}

	outputSize = layers[layers.size() - 1]->GetOutputSize();
	std::cout << "CNN succesfully loaded from '" << path << "'" << std::endl;
}

// получение выходного объёма
Volume& CNN::GetOutput(const Volume& input) {
	layers[0]->ForwardOutput(input);

	for (size_t i = 1; i < layers.size(); i++)
		layers[i]->ForwardOutput(layers[i - 1]->GetOutput());

	return layers[layers.size() - 1]->GetOutput();
}

// вывод конфигурации
void CNN::PringConfig() const {
	std::cout << "+--------------------------+--------------+---------------+--------------+----------------------------" << std::endl;
	std::cout << "|        layer type        |  input size  |  output size  | Train params | configuration: " << std::endl;
	std::cout << "+--------------------------+--------------+---------------+--------------+----------------------------" << std::endl;

	int trainable = 0;

	for (size_t i = 0; i < layers.size(); i++) {
		layers[i]->PrintConfig();
		trainable += layers[i]->GetTrainableParams();
	}

	std::cout << "+--------------------------+--------------+---------------+--------------+----------------------------" << std::endl;
	std::cout << "Total trainable params: " << trainable << std::endl;
	std::cout << std::endl;
}

// добавление слоя по текстовому описанию
void CNN::AddLayer(const std::string& layerConf) {
	VolumeSize size = layers.size() == 0 ? inputSize : layers[layers.size() - 1]->GetOutputSize();
	NetworkLayer *layer = nullptr;

	ArgParser parser(layerConf);

	if (parser["conv"] || parser["convolution"]) {
		if (!parser["filters"])
			throw std::runtime_error("Unable to add conv layer. Filters count is not set");

		std::string fs = parser.Get("filter_size", "3");
		std::string fc = parser.Get("filters");

		std::string S = parser.Get("S", "1");
		std::string P = parser.Get("P", "0");

		layer = new ConvLayer(size.width, size.height, size.deep, std::stoi(fc), std::stoi(fs), std::stoi(P), std::stoi(S));
	}
	else if (parser["maxpool"] || parser["pooling"] || parser["maxpooling"]) {
		std::string scale = parser.Get("scale", "2");
		layer = new MaxPoolingLayer(size.width, size.height, size.deep, std::stoi(scale));
	}
	else if (parser["flatten"]) {
		layer = new FlattenLayer(size.width, size.height, size.deep);
	}
	else if (parser["fc"] || parser["fullconnected"]) {
		if (!parser["outputs"])
			throw std::runtime_error("Unable to add full connected layer. Outputs is not set");

		std::string outputs = parser.Get("outputs");
		std::string type = parser.Get("activation", "sigmoid");

		layer = new FullConnectedLayer(size.width * size.height * size.deep, std::stoi(outputs), type);
	}
	else if (parser["dropout"]) {
		std::string p = parser.Get("p", "0.5");

		layer = new DropoutLayer(size.width, size.height, size.deep, std::stod(p));
	}
	else {
		throw std::runtime_error("Invalid layer name '" + layerConf + "'");
	}

	layers.push_back(layer);
	outputSize = layer->GetOutputSize();
}

// обучение сети
double CNN::Train(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, int maxEpochs, const Optimizer& optimizer, ErrorType errorType, int logPeriod) {
	size_t last = layers.size() - 1;
	double error = 0;

	if (errorType == ErrorType::MSE) {
		cost = MSE;
	}
	else if (errorType == ErrorType::CrossEntropy) {
		cost = CrossEntropy;
	}

	for (int epoch = 0; epoch < maxEpochs; epoch++) {
		auto t0 = Time::now();

		error = 0;

		// сбрасываем все накопления
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->ResetCache();

		// идём по всем обучающим примерам
		for (size_t index = 0; index < inputData.size(); index++) {
			// распространяем сигнал по слоям
			layers[0]->Forward(inputData[index]);

			for (size_t i = 1; i < layers.size(); i++)
				layers[i]->Forward(layers[i - 1]->GetOutput());

			error += (this->*cost)(layers[last]->GetOutput(), outputData[index]); // находим ошибку сети

			// распространяем ошибку по слоям обратно и обновляем весовые коэффициенты слоёв
			for (size_t i = last; i > 0; i--) {
				layers[i]->Backward(layers[i - 1]->GetDeltas());
				layers[i]->UpdateWeights(optimizer, layers[i - 1]->GetOutput());
			}

			layers[0]->UpdateWeights(optimizer, inputData[index]);

			if ((index + 1) % logPeriod == 0) {
				ms d = std::chrono::duration_cast<ms>(Time::now() - t0);
				double dt = d.count() / (index + 1.0);
				double t = (inputData.size() - index - 1) * dt;

				std::cout << index + 1 << " / " << inputData.size();
				std::cout << " [";
				int k = 30 * (index + 1) / inputData.size();

				for (int i = 0; i < k - 1; i++)
					std::cout << "=";

				std::cout << ">";

				for (int i = k; i < 30; i++)
					std::cout << ".";

				std::cout << "] - ";
				std::cout << "left: " << TimeSpan(t) << ", error: " << error << "\r";
			}
		}

		auto t1 = Time::now();
		ms d = std::chrono::duration_cast<ms>(t1 - t0);

		std::cout << "epoch: " << epoch << " ";
		std::cout << "error: " << error << " ";
		optimizer.Print();

		std::cout << " elapsed: " << TimeSpan(d.count()) << "ms                       ";
		std::cout << std::endl;
	}

	return error;
}

// сохранение сети в файл
void CNN::Save(const std::string &path) {
	std::ofstream f(path.c_str()); // создаём файл

	f << inputSize.width << " " << inputSize.height << " " << inputSize.deep << std::endl; // записываем размер входа

	// сохраняем каждый слой
	for (size_t i = 0; i < layers.size(); i++)
		layers[i]->Save(f);

	f.close(); // закрываем файл

	std::cout << "Saved to " << path << std::endl;
}
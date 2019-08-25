#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "Layers/NetworkLayer.hpp"
#include "Layers/ConvLayer.hpp"
#include "Layers/MaxPoolingLayer.hpp"
#include "Layers/FullyConnectedLayer.hpp"

#include "Layers/DropoutLayer.hpp"
#include "Layers/BatchNormalizationLayer.hpp"
#include "Layers/BatchNormalization2DLayer.hpp"

#include "Layers/Activations/SigmoidLayer.hpp"
#include "Layers/Activations/LogSigmoidLayer.hpp"
#include "Layers/Activations/TanhLayer.hpp"
#include "Layers/Activations/ReLULayer.hpp"
#include "Layers/Activations/ELULayer.hpp"
#include "Layers/Activations/ParametricReLULayer.hpp"
#include "Layers/Activations/SwishLayer.hpp"
#include "Layers/Activations/SoftsignLayer.hpp"
#include "Layers/Activations/SoftplusLayer.hpp"
#include "Layers/Activations/SoftmaxLayer.hpp"

#include "Entities/ArgParser.hpp"
#include "Entities/LossFunction.hpp"
#include "Entities/TimeSpan.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::time_point<Time> TimePoint; 
typedef std::chrono::milliseconds ms;

class Network {
	VolumeSize inputSize; // входной размер сети
	VolumeSize outputSize; // выходной размер сети

	std::vector<NetworkLayer*> layers; // слои сети
	std::vector<std::vector<Volume>> inputBatches;
	std::vector<std::vector<Volume>> outputBatches;

	std::vector<Volume>& Forward(const std::vector<Volume> &input);
	void InitBatches(const std::vector<Volume> &inputData, const std::vector<Volume> outputData, size_t batchSize);
	void SetBatchSize(int batchSize); // установка размера батча
	void ResetCache(); // сброс промежуточных данных

	double TrainBatch(const std::vector<Volume> &inputBatch, const std::vector<Volume> outputBatch, LossFunction &E, const Optimizer &optimizer); // обучение батча

public:
	Network(int width, int height, int deep);
	Network(const std::string &path);

	void AddLayer(const std::string& layerConf); // добавление слоя по текстовому описанию
	void PrintConfig() const; // вывод конфигурации

	Volume& GetOutput(const Volume& input); // получение выхода сети
	double Train(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, size_t batchSize, size_t epochs, const Optimizer &optimizer, LossType lossType); // обучение сети
	double GetError(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, LossType lossType); // получение ошибки на заданной выборке без изменения весовых коэффициентов
	void LRFind(const std::string &path, const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, size_t batchSize, double minLR, double maxLR, Optimizer &optimizer, LossType lossType); // поиск оптимальной скорости обучения

	void Save(const std::string &path, bool verbose = true); // сохранение сети в файл
	void Load(const std::string &path, bool verbose = true); // загрузка сети из файла

	void GradientChecking(const std::vector<Volume> &input, const std::vector<Volume> &output, LossType lossType); // численная проверка расчёта градиентов
};

Network::Network(int width, int height, int deep) {
	inputSize.width = width;
	inputSize.height = height;
	inputSize.deep = deep;
}

Network::Network(const std::string &path) {
	Load(path);
}

// прямое распространение сигналов по сети
std::vector<Volume>& Network::Forward(const std::vector<Volume> &input) {
	layers[0]->Forward(input);

	for (size_t i = 1; i < layers.size(); i++)
		layers[i]->Forward(layers[i - 1]->GetOutput());

	return layers[layers.size() - 1]->GetOutput();
}

// инициализация индексов батчей
void Network::InitBatches(const std::vector<Volume> &inputData, const std::vector<Volume> outputData, size_t batchSize) {
	// формируем индексы для обучающего множества
	size_t total = inputData.size();
	std::vector<int> indexes;

	for (size_t i = 0; i < total; i++)
		indexes.push_back(i);

	// перемешиваем индексы обучающего множества
	for (size_t i = total - 1; i > 0; i--)
		std::swap(indexes[i], indexes[rand() % (i + 1)]);

	// формируем индексы батчей
	inputBatches.clear();
	outputBatches.clear();

	for (size_t index = 0; index < total; index += batchSize) {
		std::vector<Volume> inputBatch;
		std::vector<Volume> outputBatch;

		// формируем индексы батча
		for (size_t i = 0; i < batchSize && index + i < total; i++) {
			inputBatch.push_back(inputData[indexes[index + i]]);
			outputBatch.push_back(outputData[indexes[index + i]]);
		}

		inputBatches.push_back(inputBatch);
		outputBatches.push_back(outputBatch);
	}
}

// установка размера батча
void Network::SetBatchSize(int batchSize) {
	for (int i = 0; i < layers.size(); i++)
		layers[i]->SetBatchSize(batchSize);
}

 // сброс промежуточных данных
void Network::ResetCache() {
	for (size_t i = 0; i < layers.size(); i++)
		layers[i]->ResetCache();
}

// добавление слоя по текстовому описанию
void Network::AddLayer(const std::string& layerConf) {
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

		layer = new ConvLayer(size, std::stoi(fc), std::stoi(fs), std::stoi(P), std::stoi(S));
	}
	else if (parser["maxpool"] || parser["pooling"] || parser["maxpooling"]) {
		std::string scale = parser.Get("scale", "2");

		layer = new MaxPoolingLayer(size, std::stoi(scale));
	}
	else if (parser["fc"] || parser["fullconnected"]) {
		if (!parser["outputs"])
			throw std::runtime_error("Unable to add full connected layer. Outputs is not set");

		std::string outputs = parser.Get("outputs");
		std::string type = parser.Get("activation", "none");

		layer = new FullyConnectedLayer(size, std::stoi(outputs), type);
	}
	else if (parser["softmax"]) {
		layer = new SoftmaxLayer(size);
	}
	else if (parser["softsign"]) {
		layer = new SoftsignLayer(size);
	}
	else if (parser["softplus"]) {
		layer = new SoftplusLayer(size);
	}
	else if (parser["sigmoid"]) {
		layer = new SigmoidLayer(size);
	}
	else if (parser["logsigmoid"]) {
		layer = new LogSigmoidLayer(size);
	}
	else if (parser["tanh"]) {
		layer = new TanhLayer(size);
	}
	else if (parser["relu"]) {
		layer = new ReLULayer(size);
	}
	else if (parser["elu"]) {
		std::string alpha = parser.Get("alpha", "1");

		layer = new ELULayer(size, std::stod(alpha));
	}
	else if (parser["prelu"] || parser["parametricrelu"]) {
		layer = new ParametricReLULayer(size);
	}
	else if (parser["swish"]) {
		layer = new SwishLayer(size);
	}
	else if (parser["dropout"]) {
		std::string p = parser.Get("p", "0.5");

		layer = new DropoutLayer(size, std::stod(p));
	}
	else if (parser["batchnormalization"]) {
		std::string momentum = parser.Get("momentum", "0.9");

		layer = new BatchNormalizationLayer(size, std::stod(momentum));
	}
	else if (parser["batchnormalization2D"]) {
		std::string momentum = parser.Get("momentum", "0.9");

		layer = new BatchNormalization2DLayer(size, std::stod(momentum));
	}
	else {
		throw std::runtime_error("Invalid layer name '" + layerConf + "'");
	}

	layers.push_back(layer);
	outputSize = layer->GetOutputSize();
}

// вывод конфигурации
void Network::PrintConfig() const {
	std::cout << "+----------------+--------------+---------------+--------------+----------------------------" << std::endl;
	std::cout << "|   layer type   |  input size  |  output size  | Train params | configuration: " << std::endl;
	std::cout << "+----------------+--------------+---------------+--------------+----------------------------" << std::endl;

	int trainable = 0;

	for (size_t i = 0; i < layers.size(); i++) {
		layers[i]->PrintConfig();
		trainable += layers[i]->GetTrainableParams();
	}

	std::cout << "+----------------+--------------+---------------+--------------+----------------------------" << std::endl;
	std::cout << "Total trainable params: " << trainable << std::endl;
	std::cout << std::endl;
}

// получение выхода сети
Volume& Network::GetOutput(const Volume& input) {
	SetBatchSize(1);
	layers[0]->ForwardOutput({ input });

	for (size_t i = 1; i < layers.size(); i++)
		layers[i]->ForwardOutput(layers[i - 1]->GetOutput());

	return layers[layers.size() - 1]->GetOutput()[0];
}

// обучение батча
double Network::TrainBatch(const std::vector<Volume> &inputBatch, const std::vector<Volume> outputBatch, LossFunction &E, const Optimizer &optimizer) {
	size_t size = inputBatch.size();
	size_t last = layers.size() - 1;

	std::vector<Volume> output = Forward(inputBatch); // получаем выход сети
	std::vector<Volume> deltas(size, Volume(outputSize)); // создаём дельты

	double loss = E.CalculateLoss(output, outputBatch, deltas); // расчитываем ошибку

	// распространям ошибку по слоям
	if (last == 0) {
		layers[last]->Backward(deltas, inputBatch, true);
	}
	else {
		layers[last]->Backward(deltas, layers[last - 1]->GetOutput(), true);

		for (size_t i = last - 1; i > 0; i--)
			layers[i]->Backward(layers[i + 1]->GetDeltas(), layers[i - 1]->GetOutput(), true);

		layers[0]->Backward(layers[1]->GetDeltas(), inputBatch, false);
	}

	// обновляем высовые кожффициенты слоёв
	for (size_t i = 0; i < layers.size(); i++)
		layers[i]->UpdateWeights(optimizer);

	return loss; // возвращаем ошибку
}

// обучение сети
double Network::Train(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, size_t batchSize, size_t epochs, const Optimizer &optimizer, LossType lossType) {
	LossFunction E(lossType); // создаём функцию ошибки
	double loss = 0;

	for (size_t epoch = 1; epoch <= epochs; epoch++) {
		InitBatches(inputData, outputData, batchSize);
		SetBatchSize(batchSize);
		ResetCache();

		TimePoint t0 = Time::now();
		int passed = 0; // количество просмотренных примеров
		loss = 0; // ошибка

		for (size_t batch = 0; batch < inputBatches.size(); batch++) {
			size_t size = inputBatches[batch].size();

			if (batch == inputBatches.size() - 1)
				SetBatchSize(size);

			loss += TrainBatch(inputBatches[batch], outputBatches[batch], E, optimizer); // обучаем на очередном батче
			passed += size; // увеличиваем счётчик просмотренных примеров

			// выводим промежуточную информацию
			ms d = std::chrono::duration_cast<ms>(Time::now() - t0);
			double dt = (double) d.count() / passed;
			double t = (inputData.size() - passed) * dt;
			std::cout << passed << "/" << inputData.size() << ", loss: " << loss / passed << "left: " << TimeSpan(t) << ", total time: " << TimeSpan(dt * inputData.size()) << "\r";
		}

		loss /= inputData.size(); // находим среднюю ошибку
	}

	return loss;
}

// получение ошибки на заданной выборке без изменения весовых коэффициентов
double Network::GetError(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, LossType lossType) {
	LossFunction E(lossType);
	double loss = 0;
	size_t total = inputData.size();

	// вычисляем ошибку на обучающих примерах
	for (size_t index = 0; index < total; index++)
		loss += E.CalculateLoss(GetOutput(inputData[index]), outputData[index]);

	return loss / total; // возвращаем среднюю ошибку
}

// поиск оптимальной скорости обучения
void Network::LRFind(const std::string &path, const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, size_t batchSize, double minLR, double maxLR, Optimizer &optimizer, LossType lossType) {
	int batches = inputData.size() / batchSize; // количество батчей
	double scale = pow(maxLR / minLR, 1.0 / batches); // шаг изменения скорости обучения

	LossFunction E(lossType); // создаём функцию ошибки
	std::ofstream f(path);

	f << "learning rate;total loss;batch loss;smoothed total loss;smoothed loss" << std::endl;

	InitBatches(inputData, outputData, batchSize); // распределяем данные по батчам
	SetBatchSize(batchSize); // задаём размер батча
	ResetCache(); // сбрасываем промежуточные данные

	double learningRate = minLR; // текущая скорость обучения
	double loss = 0; // ошибка
	double totalLoss = 0; // общая ошибка

	double smoothedLoss = 0;
	double smoothedTotalLoss = 0;
	double beta = 0.9; // коэффициент сглаживания

	int passed = 0; // количество просмотренных примеров

	for (size_t batch = 0; batch < inputBatches.size(); batch++) {
		size_t size = inputBatches[batch].size();

		if (batch == inputBatches.size() - 1)
			SetBatchSize(size);

		optimizer.SetLearningRate(learningRate);

		loss = TrainBatch(inputBatches[batch], outputBatches[batch], E, optimizer); // обучаем на очередном батче
		totalLoss += loss;
		passed += size; // увеличиваем счётчик просмотренных примеров

		smoothedTotalLoss = smoothedTotalLoss * beta + (1 - beta) * (loss / passed); // находим сглаженное значение ошибки
		smoothedLoss = smoothedLoss * beta + (1 - beta) * (loss / size); // находим сглаженное значение ошибки

		f << learningRate << ";" << totalLoss / passed << ";" << loss / size << ";" << smoothedTotalLoss << ";" << smoothedLoss << std::endl;

		learningRate *= scale; // изменяем скорость обучения

		std::cout << passed << "/" << inputData.size() << "\r";
	}

	f.close();
}

// сохранение сети в файл
void Network::Save(const std::string &path, bool verbose) {
	std::ofstream f(path); // создаём файл для сети

	f << inputSize.width << " " << inputSize.height << " " << inputSize.deep << std::endl; // записываем размер входа

	// сохраняем в файл каждый слой
	for (size_t i = 0; i < layers.size(); i++)
		layers[i]->Save(f);

	f.close(); // закрываем файл

	if (verbose)
		std::cout << "Network saved to '" << path << "'" << std::endl;
}

// загрузка сети из файла
void Network::Load(const std::string &path, bool verbose) {
	std::ifstream f(path.c_str());

	if (!f)
		throw std::runtime_error("Unable to open file with model ('" + path + "'");

	f >> inputSize.width >> inputSize.height >> inputSize.deep;

	layers.clear();
	std::string layerType;
	NetworkLayer *layer = nullptr;

	while (f >> layerType) {
		VolumeSize size;

		f >> size;

		if (layerType == "conv" || layerType == "convolution") {
			int fc, fs, P, S;

			f >> fs >> fc >> P >> S;
			
			layer = new ConvLayer(size, fc, fs, P, S, f);
		}
		else if (layerType == "maxpool" || layerType == "maxpooling") {
			int scale;
			f >> scale;

			layer = new MaxPoolingLayer(size, scale);
		}
		else if (layerType == "fc" || layerType == "fullconnected") {
			int outputs;
			std::string type;
			f >> outputs >> type;

			layer = new FullyConnectedLayer(size, outputs, type, f);
		}
		else if (layerType == "dropout") {
			double p;
			f >> p;

			layer = new DropoutLayer(size, p);
		}
		else if (layerType == "batchnormalization") {
			double momentum;
			f >> momentum;

			layer = new BatchNormalizationLayer(size, momentum, f);
		}
		else if (layerType == "batchnormalization2D") {
			double momentum;
			f >> momentum;

			layer = new BatchNormalization2DLayer(size, momentum, f);
		}
		else if (layerType == "sigmoid") {
			layer = new SigmoidLayer(size);
		}
		else if (layerType == "logsigmoid") {
			layer = new LogSigmoidLayer(size);
		}
		else if (layerType == "tanh") {
			layer = new TanhLayer(size);
		}
		else if (layerType == "relu") {
			layer = new ReLULayer(size);
		}
		else if (layerType == "elu") {
			double alpha;
			f >> alpha;

			layer = new ELULayer(size, alpha);
		}
		else if (layerType == "prelu") {
			layer = new ParametricReLULayer(size, f);
		}
		else if (layerType == "swish") {
			layer = new SwishLayer(size);
		}
		else if (layerType == "softsign") {
			layer = new SoftsignLayer(size);
		}
		else if (layerType == "softplus") {
			layer = new SoftplusLayer(size);
		}
		else if (layerType == "softmax") {
			layer = new SoftmaxLayer(size);
		}
		else
			throw std::runtime_error("Invalid layer type '" + layerType + "'");

		layers.push_back(layer);
	}

	if (layers.size() == 0)
		throw std::runtime_error("Invalid file");

	outputSize = layers[layers.size() - 1]->GetOutputSize();

	if (verbose)
		std::cout << "CNN succesfully loaded from '" << path << "'" << std::endl;
}

void Network::GradientChecking(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, LossType lossType) {
	LossFunction E(lossType);

	size_t last = layers.size() - 1;
	size_t batchSize = inputData.size();

	SetBatchSize(batchSize);

	for (size_t i = 0; i < layers.size(); i++) {
		int trainableParams = layers[i]->GetTrainableParams();

		if (trainableParams == 0)
			continue; // нет смысла проверять слои без настраиваемых параметров

		std::cout << "Layer " << (i + 1) << ": ";

		for (int index = 0; index < trainableParams; index++) {
			double weight = layers[i]->GetParam(index);
			double eps = 1e-6;

			layers[i]->SetParam(index, weight + eps);
			double E1 = E.CalculateLoss(Forward(inputData), outputData);

			layers[i]->SetParam(index, weight - eps);
			double E2 = E.CalculateLoss(Forward(inputData), outputData);

			layers[i]->SetParam(index, weight);
			layers[i]->ZeroGradient(index);

			std::vector<Volume> deltas(batchSize, Volume(outputSize));
			E.CalculateLoss(Forward(inputData), outputData, deltas);

			if (last == 0) {
				layers[0]->Backward(deltas, inputData, false);
			}
			else {
				layers[last]->Backward(deltas, layers[last - 1]->GetOutput(), true);

				for (size_t i = last - 1; i > 0; i--)
					layers[i]->Backward(layers[i + 1]->GetDeltas(), layers[i - 1]->GetOutput(), true);

				layers[0]->Backward(layers[1]->GetDeltas(), inputData, false);
			}

			double grad = layers[i]->GetGradient(index);
			double num_grad = (E1 - E2) / (eps * 2);

			if (fabs(grad - num_grad) > 1e-7) {
				std::cout << "grad: " << grad << ", num_grad: " << num_grad << ", |grad - num_grad|: " << fabs(grad - num_grad) << std::endl;
				throw std::runtime_error("GradientChecking failed at layer " + std::to_string(i + 1));
			}
		}

		std::cout << "OK" << std::endl;
	}

	std::cout << "Gradient checking: OK" << std::endl;
}
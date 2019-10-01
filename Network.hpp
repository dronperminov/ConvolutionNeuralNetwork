#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include "Layers/Layers.hpp"

#include "Entities/DataAugmentation.hpp"
#include "Entities/LossFunction.hpp"
#include "Entities/TimeSpan.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::time_point<Time> TimePoint; 
typedef std::chrono::milliseconds ms;

class Network {
	VolumeSize inputSize; // входной размер сети
	VolumeSize outputSize; // выходной размер сети

	std::vector<NetworkLayer*> layers; // слои сети
	std::vector<bool> isLearnable; // обучаемы ли слои

	std::vector<std::vector<Volume>> inputBatches;
	std::vector<std::vector<Volume>> outputBatches;

	std::vector<Volume>& Forward(const std::vector<Volume> &input, int start = 0);
	std::vector<Volume>& GetOutput(const std::vector<Volume> &input, int start, int end);

	void InitBatches(const std::vector<Volume> &inputData, const std::vector<Volume> outputData, size_t batchSize, const std::string augmentation = "");
	void SetBatchSize(int batchSize); // установка размера батча
	void ResetCache(); // сброс промежуточных данных

	double TrainBatch(const std::vector<Volume> &inputBatch, const std::vector<Volume> outputBatch, const LossFunction &E, const Optimizer &optimizer, int start = 0); // обучение батча

public:
	Network(int width, int height, int deep);
	Network(const std::string &path);

	void AddLayer(const std::string& layerConf); // добавление слоя по текстовому описанию
	void AddBlock(const std::vector<std::vector<std::string>>& blockConf, const std::string mergeType = "sum"); // добавление блока
	void RemoveLayer(int layer); // удаление слоя
	void PrintConfig() const; // вывод конфигурации

	int LayersCount() const; // количество слоёв
	VolumeSize GetOutputSize() const; // получение размера выхода сети
	Volume& GetOutput(const Volume& input); // получение выхода сети
	Volume& GetOutputFromLayer(const Volume& input, int start); // получение выхода сети, начиная со слоя start

	std::vector<Volume>& GetOutput(); // получение текущего выхода сети
	std::vector<Volume>& GetOutput(const std::vector<Volume> &inputs); // получение выхода сети
	std::vector<Volume>& GetOutputFromLayer(const std::vector<Volume>& inputs, int start); // получение выхода сети, начиная со слоя start
	std::vector<Volume>& GetOutputAtLayer(const std::vector<Volume> &inputs, int layer); // получение выхода сети на заданном слое

	double TrainOnBatch(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, const Optimizer &optimizer, const LossFunction &E, int start = 0);
	double Train(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, size_t batchSize, size_t epochs, const Optimizer &optimizer, const LossFunction &E, const std::string augmentation = ""); // обучение сети
	double GetError(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, const LossFunction &E); // получение ошибки на заданной выборке без изменения весовых коэффициентов
	void LRFind(const std::string &path, const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, size_t batchSize, double minLR, double maxLR, Optimizer &optimizer, const LossFunction &E); // поиск оптимальной скорости обучения

	void Save(const std::string &path, bool verbose = true) const; // сохранение сети в файл
	void Load(const std::string &path, bool verbose = true); // загрузка сети из файла
	void Visualize(const Volume& input, const std::string &path, int blockSize = 1); // визуализация активаций нейронной сети на каждом из уровней

	void SetLayerLearnable(int layer, bool learnable); // изменение обучаемости слоя

	void GradientChecking(const std::vector<Volume> &input, const std::vector<Volume> &output, const LossFunction &E); // численная проверка расчёта градиентов
	void PrintGradientsStats();
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
std::vector<Volume>& Network::Forward(const std::vector<Volume> &input, int start) {
	layers[start]->Forward(input);

	for (size_t i = start + 1; i < layers.size(); i++)
		layers[i]->Forward(layers[i - 1]->GetOutput());

	return layers[layers.size() - 1]->GetOutput();
}

std::vector<Volume>& Network::GetOutput(const std::vector<Volume> &inputs, int start, int end) {
	SetBatchSize(inputs.size());

	layers[start]->ForwardOutput(inputs);

	for (size_t i = start + 1; i <= end; i++)
		layers[i]->ForwardOutput(layers[i - 1]->GetOutput());

	return layers[end]->GetOutput();
}

// инициализация индексов батчей
void Network::InitBatches(const std::vector<Volume> &inputData, const std::vector<Volume> outputData, size_t batchSize, const std::string augmentation) {
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

	if (augmentation != "") {
		DataAugmentation generator(augmentation);

		for (size_t index = 0; index < total; index += batchSize) {
			std::vector<Volume> inputBatch;
			std::vector<Volume> outputBatch;

			// формируем индексы батча
			for (size_t i = 0; i < batchSize && index + i < total; i++) {
				inputBatch.push_back(generator.Make(inputData[indexes[index + i]]));
				outputBatch.push_back(outputData[indexes[index + i]]);
			}

			inputBatches.push_back(inputBatch);
			outputBatches.push_back(outputBatch);
		}
	}
	else {
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
	NetworkLayer *layer = CreateLayer(size, layerConf);

	layers.push_back(layer);
	isLearnable.push_back(true);

	outputSize = layer->GetOutputSize();
}

// добавление блока
void Network::AddBlock(const std::vector<std::vector<std::string>>& blockConf, const std::string type) {
	VolumeSize size = layers.size() == 0 ? inputSize : layers[layers.size() - 1]->GetOutputSize();
	NetworkBlock *block = new NetworkBlock(size, type);

	for (size_t i = 0; i < blockConf.size(); i++) {
		std::vector<std::string> conf = blockConf[i];
		block->AddBlock();

		NetworkLayer* prevLayer = nullptr;

		for (size_t j = 0; j < conf.size(); j++) {
			VolumeSize layerSize = prevLayer == nullptr ? size : prevLayer->GetOutputSize();
			NetworkLayer *blockLayer = CreateLayer(layerSize, conf[j]);
			prevLayer = blockLayer;

			block->AddLayer(i, blockLayer);
		}
	}

	block->Compile();

	layers.push_back(block);
	isLearnable.push_back(true);

	outputSize = block->GetOutputSize();
}

// удаление слоя
void Network::RemoveLayer(int layer) {
	if (layer < 0 || layer >= layers.size())
		throw std::runtime_error("Invalid layer for remove");

	if (layer > 0 && layer < layers.size() - 1) {
		VolumeSize prev = layers[layer - 1]->GetOutputSize();
		VolumeSize next = layers[layer + 1]->GetInputSize();

		if (prev.width != next.width || prev.height != next.height || prev.deep != next.deep)
			throw std::runtime_error("Unable to remove layer. Sizes incorrect");
	}

	layers.erase(layers.begin() + layer); // удаляем слой

	// если слоёв не осталось
	if (layers.size() == 0) {
		outputSize = inputSize; // выходной размер равен входному
	}
	else if (layer == 0) {
		inputSize = layers[0]->GetInputSize(); // обновляем входной размер
	}
	else if (layer == layers.size()) {
		outputSize = layers[layer - 1]->GetOutputSize(); // обновляем выходной размер
	}
}

// вывод конфигурации
void Network::PrintConfig() const {
	std::cout << "+------------------+--------------+---------------+--------------+----------------------------" << std::endl;
	std::cout << "|    layer type    |  input size  |  output size  | Train params | configuration: " << std::endl;
	std::cout << "+------------------+--------------+---------------+--------------+----------------------------" << std::endl;

	int trainable = 0;

	for (size_t i = 0; i < layers.size(); i++) {
		layers[i]->PrintConfig();
		trainable += layers[i]->GetTrainableParams();
	}

	std::cout << "+------------------+--------------+---------------+--------------+----------------------------" << std::endl;
	std::cout << "Total trainable params: " << trainable << std::endl;
	std::cout << std::endl;
}

// количество слоёв
int Network::LayersCount() const {
	return layers.size();
}

// получение размера выхода сети
VolumeSize Network::GetOutputSize() const {
	return outputSize;
}

// получение выхода сети
Volume& Network::GetOutput(const Volume& input) {
	SetBatchSize(1);
	layers[0]->ForwardOutput({ input });

	for (size_t i = 1; i < layers.size(); i++)
		layers[i]->ForwardOutput(layers[i - 1]->GetOutput());

	return layers[layers.size() - 1]->GetOutput()[0];
}

// получение выхода сети, начиная со слоя start
Volume& Network::GetOutputFromLayer(const Volume& input, int start) {
	return GetOutput({ input }, start, layers.size() - 1)[0];
}

// получение текущего выхода сети
std::vector<Volume>& Network::GetOutput() {
	return layers[layers.size() - 1]->GetOutput();
}

// получение выхода сети
std::vector<Volume>& Network::GetOutput(const std::vector<Volume> &inputs) {
	return GetOutput(inputs, 0, layers.size() - 1);
}

// получение выхода сети, начиная со слоя start
std::vector<Volume>& Network::GetOutputFromLayer(const std::vector<Volume>& inputs, int start) {
	return GetOutput(inputs, start, layers.size() - 1);
}

// получение выхода сети на заданном слое
std::vector<Volume>& Network::GetOutputAtLayer(const std::vector<Volume> &inputs, int layer) {
	return GetOutput(inputs, 0, layer);
}

// обучение батча
double Network::TrainBatch(const std::vector<Volume> &inputBatch, const std::vector<Volume> outputBatch, const LossFunction &E, const Optimizer &optimizer, int start) {
	size_t size = inputBatch.size();
	size_t last = layers.size() - 1;

	std::vector<Volume> output = Forward(inputBatch, start); // получаем выход сети
	std::vector<Volume> deltas(size, Volume(outputSize)); // создаём дельты

	double loss = E.CalculateLoss(output, outputBatch, deltas); // расчитываем ошибку

	// распространям ошибку по слоям
	if (last == 0) {
		layers[last]->Backward(deltas, inputBatch, true);
	}
	else {
		layers[last]->Backward(deltas, layers[last - 1]->GetOutput(), true);

		for (size_t i = last - 1; i > start; i--)
			layers[i]->Backward(layers[i + 1]->GetDeltas(), layers[i - 1]->GetOutput(), true);

		layers[start]->Backward(layers[start + 1]->GetDeltas(), inputBatch, false);
	}

	// обновляем высовые кожффициенты слоёв
	for (size_t i = start; i < layers.size(); i++)
		layers[i]->UpdateWeights(optimizer, isLearnable[i]);

	return loss; // возвращаем ошибку
}

double Network::TrainOnBatch(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, const Optimizer &optimizer, const LossFunction &E, int start) {
	SetBatchSize(inputData.size());

	return TrainBatch(inputData, outputData, E, optimizer, start);
}

// обучение сети
double Network::Train(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, size_t batchSize, size_t epochs, const Optimizer &optimizer, const LossFunction &E, const std::string augmentation) {
	double loss = 0;

	for (size_t epoch = 1; epoch <= epochs; epoch++) {
		InitBatches(inputData, outputData, batchSize, augmentation);
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
double Network::GetError(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, const LossFunction &E) {
	double loss = 0;
	size_t total = inputData.size();

	// вычисляем ошибку на обучающих примерах
	for (size_t index = 0; index < total; index++)
		loss += E.CalculateLoss(GetOutput(inputData[index]), outputData[index]);

	return loss / total; // возвращаем среднюю ошибку
}

// поиск оптимальной скорости обучения
void Network::LRFind(const std::string &path, const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, size_t batchSize, double minLR, double maxLR, Optimizer &optimizer, const LossFunction &E) {
	int batches = inputData.size() / batchSize; // количество батчей
	double scale = pow(maxLR / minLR, 1.0 / batches); // шаг изменения скорости обучения

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
void Network::Save(const std::string &path, bool verbose) const {
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

	f >> inputSize;

	layers.clear();
	std::string layerType;

	while (f >> layerType) {
		VolumeSize size;
		f >> size;

		layers.push_back(LoadLayer(size, layerType, f));
		isLearnable.push_back(true);
	}

	if (layers.size() == 0)
		throw std::runtime_error("Invalid file");

	outputSize = layers[layers.size() - 1]->GetOutputSize();

	if (verbose)
		std::cout << "Network succesfully loaded from '" << path << "'" << std::endl;
}

// визуализация активаций нейронной сети на каждом из уровней
void Network::Visualize(const Volume& input, const std::string &path, int blockSize) {
	Volume &output = GetOutput(input);

	input.Save(path + "input", blockSize);

	for (size_t i = 0; i < layers.size(); i++)
		layers[i]->GetOutput()[0].Save(path + "layer" + std::to_string(i + 1), blockSize);
}

// изменение обучаемости слоя
void Network::SetLayerLearnable(int layer, bool learnable) {
	if (layer < 0 || layer >= layers.size())
		throw std::runtime_error("Invalid layer");

	isLearnable[layer] = learnable;
}

void Network::GradientChecking(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, const LossFunction &E) {
	size_t last = layers.size() - 1;
	size_t batchSize = inputData.size();

	SetBatchSize(batchSize);

	if (E.GetName() != "")
		std::cout << "Loss: " << E.GetName() << std::endl;
	
	for (size_t i = 0; i < layers.size(); i++) {
		int trainableParams = layers[i]->GetTrainableParams();

		if (trainableParams == 0)
			continue; // нет смысла проверять слои без настраиваемых параметров

		std::cout << "Layer " << (i + 1) << ": ";

		double maxGrad = 0;
		double maxDf = 0;

		for (int index = 0; index < trainableParams; index++) {
			double weight = layers[i]->GetParam(index);
			double eps = 4e-6;

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

			maxGrad = std::max(fabs(grad), maxGrad);
			maxDf = std::max(fabs(grad - num_grad), maxDf);

			if (fabs(grad - num_grad) > 1e-7) {
				std::cout << index << ". grad: " << grad << ", num_grad: " << num_grad << ", |grad - num_grad|: " << fabs(grad - num_grad) << std::endl;
				throw std::runtime_error("GradientChecking failed at layer " + std::to_string(i + 1));
			}
		}

		std::cout << "OK (max grad: " << maxGrad << ", max df: " << maxDf << ", " << trainableParams << " / " << trainableParams << ")" << std::endl;
	}

	std::cout << "Gradient checking: OK" << std::endl;
}

// вывод статистики по градиентам
void Network::PrintGradientsStats() {
	std::cout << "+-------+---------------+---------------+" << std::endl;
	std::cout << "| layer |      min      |      max      |" << std::endl;
	std::cout << "+-------+---------------+---------------+" << std::endl;

	for (size_t i = 0; i < layers.size(); i++) {
		std::vector<Volume> &dX = layers[i]->GetDeltas();

		double min = dX[0].Min();
		double max = dX[0].Max();

		for (size_t j = 1; j < dX.size(); j++) {
			double minValue = dX[j].Min();
			double maxValue = dX[j].Max();

			if (minValue < min)
				min = minValue;

			if (maxValue > max)
				max = maxValue;
		}

		std::cout << "| " << std::setw(5) << (i + 1) << " | ";
		std::cout << std::setw(13) << min << " | ";
		std::cout << std::setw(13) << max << " |";
		std::cout << std::endl;
	}

	std::cout << "+-------+---------------+---------------+" << std::endl;
}
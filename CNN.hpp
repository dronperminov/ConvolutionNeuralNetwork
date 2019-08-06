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
#include "Layers/SoftmaxLayer.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::time_point<Time> TimePoint; 
typedef std::chrono::milliseconds ms;

enum class ErrorType {
	MSE, // среднеквадратичное отклонение
	CrossEntropy, // перекрёстная этнропия
	BinaryCrossEntropy, // бинарная перекрёстная этнропия
};

class CNN {
	VolumeSize inputSize; // размер входного объёма
	VolumeSize outputSize; // размер выходного объёма

	std::vector<NetworkLayer *> layers; // слои сети

	double (CNN::*cost)(const Volume&, const Volume&); // указатель на функцию стоимости

	double MSE(const Volume& y, const Volume& t); // среднеквадратичное отклонение
	double CrossEntropy(const Volume &y, const Volume &t); // перекрёстная энтропия
	double BinaryCrossEntropy(const Volume &y, const Volume &t); // бинарная перекрёстная энтропия

	void Forward(const Volume &input); // прямое распространение

	std::string SetCost(ErrorType errorType); // установка функции ошибки
	void Log(TimePoint t0, const std::string &costType, double error, int index, int total) const; // вывод информации
	void PrintEpochInfo(TimePoint t0, int epoch, const Optimizer &optimizer, const std::string costType, double error) const; // вывод информации по завершении эпохи

public:
	CNN(int width, int height, int deep); // инициализация сети
	CNN(const std::string& path); // загрузка сети из файла

	Volume& GetOutput(const Volume& input); // получение ответа сети

	void PringConfig() const; // вывод конфигурации

	void AddLayer(const std::string& layerConf); // добавление слоя по текстовому описанию
	void Save(const std::string &path, bool verbose = true); // сохранение сети в файл
	void Load(const std::string &path, bool verbose = true); // загрузка сети из файла
	
	double Train(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, int maxEpochs, const Optimizer &optimizer, ErrorType errorType = ErrorType::MSE, int logPeriod = 100);
	double TrainMiniBatch(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, int batchSize, int maxEpochs, const Optimizer &optimizer, ErrorType errorType = ErrorType::MSE, int logPeriod = 100);
	double GetError(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, ErrorType errorType = ErrorType::MSE);
	double FindLearningRate(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, int batchSize, double startLR, double endLR, Optimizer &optimizer, ErrorType errorType);
	
	void GradientChecking(const Volume &input, const Volume &output, ErrorType errorType);
};

// среднеквадратичное отклонение
double CNN::MSE(const Volume& y, const Volume& t) {
	double error = 0;
	int last = layers.size() - 1;
	int total = outputSize.deep * outputSize.height * outputSize.width;

	for (int i = 0; i < total; i++) {
		double e = y[i] - t[i]; // находим разность между выходом сети и обучающим примером
		layers[last]->GetDeltas()[i] *= 2*e; // умножаем на неё дельту выходного слоя
		error += e * e; // прибавляем квадрат разности
	}

	return error;
}

// перекрётсная энтропия
double CNN::CrossEntropy(const Volume &y, const Volume &t) {
	double error = 0;
	size_t last = layers.size() - 1;
	int total = outputSize.deep * outputSize.height * outputSize.width;

	for (int i = 0; i < total; i++) {
		double yi = std::max(1e-7, std::min(1 - 1e-7, y[i]));
		double ti = t[i];

		error -= ti * log(yi);

		layers[last]->GetDeltas()[i] *= -ti / yi;
	}

	return error;
}

// перекрётсная энтропия
double CNN::BinaryCrossEntropy(const Volume &y, const Volume &t) {
	double error = 0;
	size_t last = layers.size() - 1;
	int total = outputSize.deep * outputSize.height * outputSize.width;

	for (int i = 0; i < total; i++) {
		double yi = std::max(1e-7, std::min(1 - 1e-7, y[i]));
		double ti = t[i];

		error -= ti * log(yi) + (1 - ti) * log(1 - yi);

		layers[last]->GetDeltas()[i] *= (yi - ti) / (yi * (1 - yi));
	}

	return error;
}

// прямое распространение
void CNN::Forward(const Volume &input) {
	layers[0]->Forward(input);

	for (size_t i = 1; i < layers.size(); i++)
		layers[i]->Forward(layers[i - 1]->GetOutput());
}

// инициализация сети
CNN::CNN(int width, int height, int deep) {
	inputSize.width = width;
	inputSize.height = height;
	inputSize.deep = deep;
}

// загрузка сети из файла
CNN::CNN(const std::string& path) {
	Load(path);
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
		std::string type = parser.Get("activation", "none");

		layer = new FullConnectedLayer(size.width * size.height * size.deep, std::stoi(outputs), type);
	}
	else if (parser["dropout"]) {
		std::string p = parser.Get("p", "0.5");

		layer = new DropoutLayer(size.width, size.height, size.deep, std::stod(p));
	}
	else if (parser["softmax"]) {
		layer = new SoftmaxLayer(size.width, size.height, size.deep);
	}
	else {
		throw std::runtime_error("Invalid layer name '" + layerConf + "'");
	}

	layers.push_back(layer);
	outputSize = layer->GetOutputSize();
}

// установка функции ошибки
std::string CNN::SetCost(ErrorType errorType) {
	if (errorType == ErrorType::MSE) {
		cost = MSE;
		return "MSE";
	}
	
	if (errorType == ErrorType::CrossEntropy) {
		cost = CrossEntropy;
		return "CrossEntropy";
	}

	if (errorType == ErrorType::BinaryCrossEntropy) {
		cost = BinaryCrossEntropy;
		return "BinaryCrossEntropy";
	}

	throw std::runtime_error("Unknown error type");
}

// вывод информации
void CNN::Log(TimePoint t0, const std::string &costType, double error, int index, int total) const {
	ms d = std::chrono::duration_cast<ms>(Time::now() - t0);
	double dt = d.count() / (index + 1.0);
	double t = (total - index - 1) * dt;

	std::cout << index + 1 << " / " << total;
	std::cout << " [";
	int k = 30 * (index + 1) / total;

	for (int i = 0; i < k - 1; i++)
		std::cout << "=";

	std::cout << ">";

	for (int i = k; i < 30; i++)
		std::cout << ".";

	std::cout << "] - ";
	std::cout << "left: " << TimeSpan(t) << ", " << costType << ": " << error << "\r";
}

// вывод информации по завершении эпохи
void CNN::PrintEpochInfo(TimePoint t0, int epoch, const Optimizer &optimizer, const std::string costType, double error) const {
	TimePoint t1 = Time::now();
	ms d = std::chrono::duration_cast<ms>(t1 - t0);

	std::cout << "epoch: " << epoch << " ";
	std::cout << costType << ": " << error << " ";
	optimizer.Print();

	std::cout << " elapsed: " << TimeSpan(d.count()) << "ms                       ";
	std::cout << "\r";
}

// обучение сети
double CNN::Train(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, int maxEpochs, const Optimizer& optimizer, ErrorType errorType, int logPeriod) {
	size_t last = layers.size() - 1;
	size_t total = inputData.size();
	double error = 0;
	std::string costType = SetCost(errorType);

	for (int epoch = 0; epoch < maxEpochs; epoch++) {
		TimePoint t0 = Time::now();
		error = 0;

		// сбрасываем все накопления
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->ResetCache();

		// идём по всем обучающим примерам
		for (size_t index = 0; index < total; index++) {
			Forward(inputData[index]); // распространяем сигнал по слоям

			error += (this->*cost)(layers[last]->GetOutput(), outputData[index]); // находим ошибку сети

			// распространяем ошибку по слоям обратно и обновляем весовые коэффициенты слоёв
			for (size_t i = last; i > 0; i--) {
				layers[i]->Backward(layers[i - 1]->GetDeltas());
				layers[i]->UpdateWeights(optimizer, layers[i - 1]->GetOutput());
			}

			layers[0]->UpdateWeights(optimizer, inputData[index]);

			if ((index + 1) % logPeriod == 0)
				Log(t0, costType, error / (index + 1), index, inputData.size()); // выводим промежуточную информацию
		}

		PrintEpochInfo(t0, epoch, optimizer, costType, error / total); // выводим информацию по эпохе
	}

	return error / total;
}

// обучение сети мини пакетами
double CNN::TrainMiniBatch(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, int batchSize, int maxEpochs, const Optimizer &optimizer, ErrorType errorType, int logPeriod) {
	size_t last = layers.size() - 1;
	size_t total = inputData.size();
	double error = 0;
	std::string costType = SetCost(errorType);

	std::vector<int> indexes;
	
	for (size_t i = 0; i < total; i++)
		indexes.push_back(i);

	for (int epoch = 0; epoch < maxEpochs; epoch++) {
		for (size_t i = total; i > 0; i--)
			std::swap(indexes[i], indexes[rand() % (i + 1)]);
		
		TimePoint t0 = Time::now();
		error = 0;

		// сбрасываем все накопления
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->ResetCache();

		// идём по всем обучающим примерам
		for (size_t index = 0; index < total; index += batchSize) {
			int min = index + batchSize < total ? batchSize : total - index;
			
			for (int batch = 0; batch < min; batch++) {
				int curr = indexes[index + batch];

				Forward(inputData[curr]); // распространяем сигнал по слоям

				error += (this->*cost)(layers[last]->GetOutput(), outputData[curr]); // находим ошибку сети

				// распространяем ошибку по слоям обратно
				for (size_t i = last; i > 0; i--) {
					layers[i]->Backward(layers[i - 1]->GetDeltas());
					layers[i]->CalculateGradients(layers[i - 1]->GetOutput());
				}

				// вычисляем градиенты
				layers[0]->CalculateGradients(inputData[curr]);

				if ((index + batch + 1) % logPeriod == 0)
					Log(t0, costType, error / (index + batch + 1), index + batch, total); // выводим промежуточную информацию
			}

			// обновляем весовые коэффициенты
			for (size_t i = 0; i < layers.size(); i++)
				layers[i]->UpdateWeights(optimizer, batchSize);
		}

		PrintEpochInfo(t0, epoch, optimizer, costType, error / total); // выводим информацию по эпохе
	}

	return error / total;
}

double CNN::FindLearningRate(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, int batchSize, double startLR, double endLR, Optimizer &optimizer, ErrorType errorType) {
	size_t last = layers.size() - 1;
	size_t total = inputData.size();

	size_t numBatches = total / batchSize;
	double scale = pow(endLR / startLR, 1.0 / numBatches);
	double learningRate = startLR;

	double error = 0;
	size_t passed = 0;

	double minError = -1;
	double minLearningRate = startLR;

	std::string costType = SetCost(errorType);

	std::vector<int> indexes;
	
	for (size_t i = 0; i < total; i++)
		indexes.push_back(i);

	for (size_t i = total; i > 0; i--)
		std::swap(indexes[i], indexes[rand() % (i + 1)]);

	// сбрасываем все накопления
	for (size_t i = 0; i < layers.size(); i++)
		layers[i]->ResetCache();

	// идём по всем обучающим примерам
	for (size_t index = 0; index < total; index += batchSize) {
		optimizer.SetLearningRate(learningRate);		

		for (int batch = 0; batch < batchSize && index + batch < total; batch++) {
			int curr = indexes[index + batch];

			Forward(inputData[curr]); // распространяем сигнал по слоям

			error += (this->*cost)(layers[last]->GetOutput(), outputData[curr]); // находим ошибку сети

			// распространяем ошибку по слоям обратно
			for (size_t i = last; i > 0; i--)
				layers[i]->Backward(layers[i - 1]->GetDeltas());

			// вычисляем градиенты
			layers[0]->CalculateGradients(inputData[curr]);

			for (size_t i = 1; i < layers.size(); i++)
				layers[i]->CalculateGradients(layers[i - 1]->GetOutput());

			passed++;
		}

		// обновляем весовые коэффициенты
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->UpdateWeights(optimizer, batchSize);

		if (minError == -1 || error / passed < minError) {
			minError = error / passed;
			minLearningRate = learningRate;
		}

		std::cout << "batch " << ((1 + index) / batchSize) << "/" << numBatches << ", learning rate: " << learningRate << ", loss: " << error / passed << ", min loss: " << minError << ", min lr: " << minLearningRate << "\r";
		learningRate *= scale;
	}

	return minLearningRate / 10;
}

// получение ошибки на заданной выборке без изменения весовых коэффициентов
double CNN::GetError(const std::vector<Volume> &inputData, const std::vector<Volume> &outputData, ErrorType errorType) {
	size_t last = layers.size() - 1;
	size_t total = inputData.size();
	double error = 0;
	
	SetCost(errorType);

	// идём по всем обучающим примерам
	for (size_t index = 0; index < total; index++) {
		// распространяем сигнал по слоям
		layers[0]->Forward(inputData[index]);

		for (size_t i = 1; i < layers.size(); i++)
			layers[i]->Forward(layers[i - 1]->GetOutput());

		error += (this->*cost)(layers[last]->GetOutput(), outputData[index]); // находим ошибку сети
	}

	return error / total;
}

void CNN::GradientChecking(const Volume &input, const Volume &output, ErrorType errorType) {
	SetCost(errorType);
	size_t last = layers.size() - 1;

	for (size_t i = 0; i < layers.size(); i++) {
		int trainableParams = layers[i]->GetTrainableParams();

		if (trainableParams == 0)
			continue; // нет смысла проверять слои без настраиваемых параметров

		std::cout << "Layer " << (i + 1) << ": ";

		for (int index = 0; index < trainableParams; index++) {

			double weight = layers[i]->GetParam(index);
			double eps = 1e-5;

			layers[i]->SetParam(index, weight + eps);
			Forward(input);
			double E1 = (this->*cost)(layers[last]->GetOutput(), output);

			layers[i]->SetParam(index, weight - eps);
			Forward(input);
			double E2 = (this->*cost)(layers[last]->GetOutput(), output);

			layers[i]->SetParam(index, weight);
			Forward(input);

			double E = (this->*cost)(layers[last]->GetOutput(), output);

			// распространяем ошибку по слоям обратно
			for (size_t j = last; j > 0; j--)
				layers[j]->Backward(layers[j - 1]->GetDeltas());

			double grad = layers[i]->GetGradient(index, i == 0 ? input : layers[i - 1]->GetOutput());
			double num_grad = (E1 - E2) / (eps * 2);


			if (fabs(grad - num_grad) > 1e-7)
				throw std::runtime_error("GradientChecking failed at layer " + std::to_string(i));
		}
		
		std::cout << "OK" << std::endl;
	}

	std::cout << "Gradient checking: OK" << std::endl;
}

// сохранение сети в файл
void CNN::Save(const std::string &path, bool verbose) {
	std::ofstream f(path.c_str()); // создаём файл

	f << inputSize.width << " " << inputSize.height << " " << inputSize.deep << std::endl; // записываем размер входа

	// сохраняем каждый слой
	for (size_t i = 0; i < layers.size(); i++)
		layers[i]->Save(f);

	f.close(); // закрываем файл

	if (verbose)
		std::cout << "Saved to " << path << std::endl;
}

// загрузка сети из файла
void CNN::Load(const std::string &path, bool verbose) {
	std::ifstream f(path.c_str());

	if (!f)
		throw std::runtime_error("Unable to open file with model");

	f >> inputSize.width >> inputSize.height >> inputSize.deep;

	layers.clear();
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
		else if (layerType == "softmax") {
			int w, h, d;

			f >> w >> h >> d;
			layer = new SoftmaxLayer(w, h, d);
		}
		else
			throw std::runtime_error("Invalid layer type");

		layers.push_back(layer);
	}

	outputSize = layers[layers.size() - 1]->GetOutputSize();

	if (verbose)
		std::cout << "CNN succesfully loaded from '" << path << "'" << std::endl;
}
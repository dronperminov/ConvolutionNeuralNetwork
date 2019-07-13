#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "Volume.hpp"
#include "../CNN.hpp"

class DataLoader {
public:
	std::vector<Volume> trainInputData; // обучающая входная выборка
	std::vector<Volume> trainOutputData; // обучающая выходная выборка

private:
	std::vector<std::string> labels; // метки классов
	VolumeSize inputSize; // размер входа

	std::vector<std::string> SplitByChar(const std::string s, char c = ',') const; // разбиение строки по символу
	int GetLabelIndex(const std::string &label) const; // индекс класса
	int GetOutputIndex(CNN &cnn, const Volume& input) const; // получение индекса максимального аргумента

	Volume GetVolume(const std::vector<std::string> &args, int start = 1);
	void ReadTrain(const std::string &trainPath, int maxTrainData);
	void ReadLabels(const std::string& path); // считывание меток классов

public:
	DataLoader(const std::string &trainPath, int width, int height, int deep, const std::string &labelsPath, int maxTrainData = 100);

	double Test(CNN &cnn, const std::string &testPath, const std::string &msg, int maxCount = -1); // проверка точности предсказаний сети
};

// разбиение строки по символу
std::vector<std::string> DataLoader::SplitByChar(const std::string s, char c) const {
	std::vector<std::string> args;
	size_t i = 0;

	while (i < s.length()) {
		std::string arg = "";

		while (i < s.length() && s[i] != c) {
			arg += s[i];
			i++;
		}

		if (arg.length())
			args.push_back(arg);

		while (i < s.length() && s[i] == ',')
			i++;
	}

	return args;
}

// индекс класса
int DataLoader::GetLabelIndex(const std::string &label) const {
	for (size_t i = 0; i < labels.size(); i++)
		if (labels[i] == label)
			return i;

	return -1;
}

// получение индекса максимального аргумента
int DataLoader::GetOutputIndex(CNN &cnn, const Volume& input) const {
	Volume& output = cnn.GetOutput(input);
	int imax = 0;

	for (int i = 0; i < output.Deep(); i++)
		if (output(i, 0, 0) > output(imax, 0, 0))
			imax = i;

	return imax;
}

Volume DataLoader::GetVolume(const std::vector<std::string> &args, int start) {
	Volume input(inputSize.width, inputSize.height, inputSize.deep);

	int index = start;

	if (args.size() != inputSize.width * inputSize.height * inputSize.deep + start)
		throw std::runtime_error("Invalid file");

	for (int i = 0; i < inputSize.height; i++)
		for (int j = 0; j < inputSize.width; j++)
			for (int d = 0; d < inputSize.deep; d++)
				input(d, i, j) = std::stod(args[index++]) / 256.0;

	return input;
}

void DataLoader::ReadTrain(const std::string &trainPath, int maxTrainData) {
	std::ifstream f(trainPath.c_str());

	if (!f)
		throw std::runtime_error("Invalid path to train file");

	std::string line;

	std::getline(f, line);

	trainInputData.clear();
	trainOutputData.clear();

	while (std::getline(f, line) && trainInputData.size() < maxTrainData) {
		std::vector<std::string> args = SplitByChar(line, ',');

		int label = GetLabelIndex(args[0]);

		if (label == -1)
			throw std::runtime_error("Unknown label");

		trainInputData.push_back(GetVolume(args));
		Volume output(1, 1, labels.size());

		for (int i = 0; i < labels.size(); i++)
			output(i, 0, 0) = label == i ? 1 : 0;
		
		trainOutputData.push_back(output);
	}

	f.close();
}

// считывание меток классов
void DataLoader::ReadLabels(const std::string& path) {
	std::ifstream f(path.c_str());

	if (!f)
		throw std::runtime_error("Invalid path to labels file");

	std::string line;

	labels.clear();

	while (std::getline(f, line))
		labels.push_back(line);

	f.close();

	std::cout << "Succesfully loaded " << labels.size() << " labels" << std::endl;
}

DataLoader::DataLoader(const std::string &trainPath, int width, int height, int deep, const std::string &labelsPath, int maxTrainData) {
	ReadLabels(labelsPath);

	inputSize.width = width;
	inputSize.height = height;
	inputSize.deep = deep;

	ReadTrain(trainPath, maxTrainData); // формируем обучающую выборку
	std::cout << "Succesfully loaded " << trainInputData.size() << " train samples" << std::endl;
}

double DataLoader::Test(CNN &cnn, const std::string &testPath, const std::string &msg, int maxCount) {
	std::ifstream f(testPath.c_str());
	std::string line;

	std::getline(f, line);

	int correct = 0;
	int total = 0;

	while (std::getline(f, line) && total != maxCount) {
		std::vector<std::string> args = SplitByChar(line, ',');
		
		int label = GetLabelIndex(args[0]);

		if (label == -1)
			throw std::runtime_error("Unknown label");

		Volume input = GetVolume(args);
		int index = GetOutputIndex(cnn, input);

		total++;

		if (index == label)
			correct++;

		std::cout << msg << correct << " / " << total << ": " << correct * 100.0 / total << "%               \r";
	}

	f.close();

	std::cout << msg << correct << " / " << total << ": " << correct * 100.0 / total << "%            " << std::endl;

	return (double)correct / total;
}
#include <iostream>
#include <fstream>
#include <vector>
#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

void PrintLine(int maxEpochs) {
	cout << "+----------------------+";
	for (int i = 0; i < maxEpochs; i++)
		cout << "-----------------------+";
	cout << endl;
}

void PrintHeader(int maxEpochs) {
	PrintLine(maxEpochs);

	cout << "|      Algorithm       |";
	for (int i = 0; i < maxEpochs; i++)
		cout << "        Epoch " << (i + 1) << "        |";
	cout << endl;

	PrintLine(maxEpochs);
}

int main() {
	string dir = "../dataset/"; // путь к папке с файлами
	string train = dir + "mnist_train.csv"; // обучающая выборка
	string test = dir + "mnist_test.csv"; // тестовая выборка
	string labels = dir + "mnist.txt"; // файл с классами

	int width = 28; // ширина изображений
	int height = 28; // высота изображений
	int deep = 1; // количество каналов

	int trainCount = 10000; // число обучающих примеров

	double learningRate = 0.0015; // скорость обучения
	int batchSize = 64; // размер батча
	int maxEpochs = 5; // число эпох обучения

	vector<LossType> losses = { LossType::CrossEntropy, LossType::BinaryCrossEntropy, LossType::MSE, LossType::Exp, LossType::Logcosh, LossType::MAE }; // функции ошибки
	vector<string> names = { "cross entropy", "binary cross entropy", "MSE", "exp", "logcosh", "MAE" };
	vector<Network> networks;
	Optimizer optimizer = Optimizer::AdaMax(learningRate);

	for (size_t i = 0; i < losses.size(); i++) {
		networks.push_back(Network(width, height, deep));

		networks[i].AddLayer("conv filter_size=5 filters=16");
		networks[i].AddLayer("maxpool");

		networks[i].AddLayer("conv filter_size=5 filters=32");
		networks[i].AddLayer("maxpool");
		networks[i].AddLayer("batchnormalization2D");

		networks[i].AddLayer("fullconnected outputs=128 activation=relu");
		networks[i].AddLayer("batchnormalization");
		networks[i].AddLayer("relu");
		networks[i].AddLayer("fullconnected outputs=10");
		networks[i].AddLayer("softmax");
	}
	
	networks[0].PrintConfig();

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	PrintHeader(maxEpochs);

	for (size_t i = 0; i < losses.size(); i++) {
		std::vector<double> errors;
		std::vector<double> trainAcc;
		std::vector<double> testAcc;

		srand(i);

		for (int k = 0; k < maxEpochs; k++) {
			optimizer.SetEpoch(k + 1);

			double error = networks[i].Train(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizer, losses[i]); // обучаем в течение одной эпохи
			double test_acc = loader.Test(networks[i], test, "", 10000); // проверяем точность на тестовой выборке
			double train_acc = loader.Test(networks[i], train, "", 10000); // проверяем точность на обучающей выборке

			errors.push_back(error);
			trainAcc.push_back(train_acc);
			testAcc.push_back(test_acc);
		}

		cout << "| " << setfill(' ') << left << setw(20) << names[i] << " |";

		for (int k = 0; k < maxEpochs; k++) {
			cout.precision(5);
			cout << " " << setw(6) << right << fixed << errors[k] << ",";
			cout << " " << setprecision(2) << testAcc[k] << ",";
			cout << " " << setprecision(2) << trainAcc[k] << " |";
		}

		cout << endl;
		cout.precision(6);
	}

	PrintLine(maxEpochs);
}
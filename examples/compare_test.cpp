#include <iostream>
#include <fstream>
#include <vector>
#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

void PrintLine(int maxEpochs) {
	cout << "+----------------------------+";
	for (int i = 0; i < maxEpochs; i++)
		cout << "-----------------------+";
	cout << endl;
}

void PrintHeader(int maxEpochs) {
	PrintLine(maxEpochs);

	cout << "|         Algorithm          |";
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
	LossFunction loss = LossFunction::CrossEntropy(); // функция ошибки

	vector<Network> networks;
	Optimizer optimizer = Optimizer::AdaMax(learningRate);

	for (size_t i = 0; i < 8; i++) {
		networks.push_back(Network(width, height, deep));

		networks[i].AddLayer("conv filter_size=5 filters=16");
		networks[i].AddLayer("relu");
		networks[i].AddLayer("maxpool");

		if (i == 1 || i == 4 || i == 5 || i == 7)
			networks[i].AddLayer("batchnormalization2D");

		networks[i].AddLayer("conv filter_size=5 filters=32");
		networks[i].AddLayer("relu");
		networks[i].AddLayer("maxpool");

		if (i == 2 || i == 4 || i == 6 || i == 7)
			networks[i].AddLayer("batchnormalization2D");

		networks[i].AddLayer("fullconnected outputs=128");

		if (i == 3 || i == 5 || i == 6 || i == 7)
			networks[i].AddLayer("batchnormalization");

		networks[i].AddLayer("relu");

		networks[i].AddLayer("fullconnected outputs=10");
		networks[i].AddLayer("softmax");

		networks[i].PrintConfig();
	}

	optimizer.Print();

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	PrintHeader(maxEpochs);

	for (size_t i = 0; i < networks.size(); i++) {
		std::vector<double> errors;
		std::vector<double> trainAcc;
		std::vector<double> testAcc;

		srand(1);

		for (int k = 0; k < maxEpochs; k++) {
			optimizer.SetEpoch(k + 1);

			double error = networks[i].Train(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizer, loss); // обучаем в течение одной эпохи
			double test_acc = loader.Test(networks[i], test, "", 10000); // проверяем точность на тестовой выборке
			double train_acc = loader.Test(networks[i], train, "", 10000); // проверяем точность на обучающей выборке

			errors.push_back(error);
			trainAcc.push_back(train_acc);
			testAcc.push_back(test_acc);
		}

		cout << "| " << setfill(' ') << left << setw(26) << ("network " + to_string(i)) << " |";

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
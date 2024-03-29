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

	double learningRate = 0.01; // скорость обучения
	int batchSize = 64; // размер батча
	int maxEpochs = 5; // число эпох обучения
	LossFunction loss = LossFunction::CrossEntropy(); // функция ошибки

	vector<Network> networks;
	vector<Optimizer> optimizers;
	vector<string> names;

	names.push_back("SGD");
	optimizers.push_back(Optimizer::SGD(learningRate));

	names.push_back("SGDm 0.9");
	optimizers.push_back(Optimizer::SGDm(learningRate));

	names.push_back("Adagrad");
	optimizers.push_back(Optimizer::Adagrad(learningRate));

	names.push_back("NAG");
	optimizers.push_back(Optimizer::NAG(learningRate));

	names.push_back("Adadelta");
	optimizers.push_back(Optimizer::Adadelta());

	names.push_back("RMSprop");
	optimizers.push_back(Optimizer::RMSprop(learningRate));

	names.push_back("Adam");
	optimizers.push_back(Optimizer::Adam(learningRate));

	names.push_back("AMSgrad");
	optimizers.push_back(Optimizer::AMSgrad(0.001));

	names.push_back("Adamax");
	optimizers.push_back(Optimizer::AdaMax(learningRate));

	names.push_back("Nadam");
	optimizers.push_back(Optimizer::Nadam(learningRate));

	for (size_t i = 0; i < optimizers.size(); i++) {
		networks.push_back(Network(width, height, deep));

		networks[i].AddLayer("fullconnected outputs=128 activation=relu");
		networks[i].AddLayer("batchnormalization");
		networks[i].AddLayer("fullconnected outputs=10");
		networks[i].AddLayer("softmax");
	}

	networks[0].PrintConfig(); // выводим конфигурацию сети

	cout << "Optimizers:" << endl;

	for (size_t i = 0; i < optimizers.size(); i++) {
		cout << (i + 1) << ". ";
		optimizers[i].Print();
		cout << endl;
	}

	cout << endl;

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	PrintHeader(maxEpochs);

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (size_t i = 0; i < optimizers.size(); i++) {
		std::vector<double> errors;
		std::vector<double> trainAcc;
		std::vector<double> testAcc;

		srand(1);
		
		for (int j = 0; j < maxEpochs; j++) {
			optimizers[i].SetEpoch(j + 1);

			double error = networks[i].Train(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizers[i], loss); // обучаем в течение одной эпохи
			double test_acc = loader.Test(networks[i], test, "", 10000); // проверяем точность на тестовой выборке
			double train_acc = loader.Test(networks[i], train, "", 10000); // проверяем точность на обучающей выборке

			errors.push_back(error);
			trainAcc.push_back(train_acc);
			testAcc.push_back(test_acc);
		}

		cout << "| " << setfill(' ') << left << setw(20) << names[i] << " |";

		for (int j = 0; j < maxEpochs; j++) {
			cout.precision(5);
			cout << " " << setw(6) << right << fixed << errors[j] << ",";
			cout << " " << setprecision(2) << testAcc[j] << ",";
			cout << " " << setprecision(2) << trainAcc[j] << " |";
		}

		cout << endl;
	}

	PrintLine(maxEpochs);
}
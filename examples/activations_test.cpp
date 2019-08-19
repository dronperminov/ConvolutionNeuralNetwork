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

	double learningRate = 0.01; // скорость обучения
	int batchSize = 64; // размер батча
	int maxEpochs = 5; // число эпох обучения
	LossType lossType = LossType::CrossEntropy; // функция ошибки

	vector<Network> networks;
	vector<Optimizer> optimizers;
	vector<string> names;
	vector<string> activations = { "relu", "prelu", "elu", "swish", "sigmoid", "tanh" };

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
		for (size_t k = 0; k < activations.size(); k++) {
			networks.push_back(Network(width, height, deep));

			int index = i * activations.size() + k;

			networks[index].AddLayer("fullconnected outputs=128");
			networks[index].AddLayer(activations[k]);
			networks[index].AddLayer("fullconnected outputs=10");
			networks[index].AddLayer("softmax");
		}
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
		for (size_t j = 0; j < activations.size(); j++) {
			std::vector<double> errors;
			std::vector<double> trainAcc;
			std::vector<double> testAcc;

			for (int k = 0; k < maxEpochs; k++) {
				optimizers[i].SetEpoch(k + 1);

				int index = i * activations.size() + j;

				srand(i);

				double error = networks[index].Train(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizers[i], lossType); // обучаем в течение одной эпохи
				double test_acc = loader.Test(networks[index], test, "", 10000); // проверяем точность на тестовой выборке
				double train_acc = loader.Test(networks[index], train, "", 10000); // проверяем точность на обучающей выборке

				errors.push_back(error);
				trainAcc.push_back(train_acc);
				testAcc.push_back(test_acc);
			}

			cout << "| " << setfill(' ') << left << setw(26) << (names[i] + ", " + activations[j]) << " |";

			for (int k = 0; k < maxEpochs; k++) {
				cout.precision(5);
				cout << " " << setw(6) << right << fixed << errors[k] << ",";
				cout << " " << setprecision(2) << testAcc[k] << ",";
				cout << " " << setprecision(2) << trainAcc[k] << " |";
			}

			cout << endl;
		}

		cout << endl;
	}

	PrintLine(maxEpochs);
}
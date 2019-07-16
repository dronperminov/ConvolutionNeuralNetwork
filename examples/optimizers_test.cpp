#include <iostream>
#include <fstream>
#include <vector>
#include "../CNN.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

int main() {
	string dir = "../dataset/"; // путь к папке с файлами
	string train = dir + "mnist_train.csv"; // обучающая выборка
	string test = dir + "mnist_test.csv"; // тестовая выборка
	string labels = dir + "mnist.txt"; // файл с классами

	int width = 28; // ширина изображений
	int height = 28; // высота изображений
	int deep = 1; // количество каналов

	int trainCount = 10000; // число обучающих примеров

	double learningRate = 0.004; // скорость обучения
	int maxEpochs = 5; // число эпох обучения

	vector<CNN> cnns;
	vector<Optimizer> optimizers;
	vector<double> bests;
	vector<string> names;

	optimizers.push_back(Optimizer::SGD(learningRate));
	optimizers.push_back(Optimizer::SGDm(learningRate, 0.4));
	optimizers.push_back(Optimizer::SGDm(learningRate));
	optimizers.push_back(Optimizer::Adagrad(0.01));
	optimizers.push_back(Optimizer::Adadelta());
	optimizers.push_back(Optimizer::RMSprop(0.002));
	optimizers.push_back(Optimizer::NAG(learningRate));
	optimizers.push_back(Optimizer::Adam(0.002));

	names.push_back("sgd");
	names.push_back("sgdm 0.4");
	names.push_back("sgdm 0.9");
	names.push_back("adagrad");
	names.push_back("adadelta");
	names.push_back("rmsprop");
	names.push_back("nag");
	names.push_back("adam");

	for (int i = 0; i < optimizers.size(); i++) {
		cnns.push_back(CNN(width, height, deep));
		cnns[i].AddLayer("conv filters=16 filter_size=5");
		cnns[i].AddLayer("maxpool");
		cnns[i].AddLayer("conv filters=32 filter_size=5");
		cnns[i].AddLayer("maxpool");
		cnns[i].AddLayer("flatten");
		cnns[i].AddLayer("fullconnected outputs=10 activation=sigmoid");
	}

	cnns[0].PringConfig(); // выводим конфигурацию сети

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (int i = 0; i < maxEpochs; i++) {
		cout << (i + 1) << ":" << endl;

		for (int j = 0; j < cnns.size(); j++) {
			optimizers[j].SetEpoch(i + 1);
			double error = cnns[j].Train(loader.trainInputData, loader.trainOutputData, 1, optimizers[j]); // обучаем в течение одной эпохи

			cout << names[j] << "                                                                            " << endl;
			cout << "    error: " << error << endl;
			double test_acc = loader.Test(cnns[j], test, "    Test accuracy: ", 10000); // проверяем точность на тестовой выборке
			double train_acc = loader.Test(cnns[j], train, "    Train accuracy: ", 10000); // проверяем точность на обучающей выборке

			cout << endl;
		}

		cout << "============================================================================================" << endl;
	}
}
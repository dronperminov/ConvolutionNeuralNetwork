#include <iostream>
#include <fstream>
#include "../Network.hpp"
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

	int trainCount = 60000; // число обучающих примеров (вся выборка)

	double learningRate = 0.001; // скорость обучения
	int batchSize = 64; // размер батча
	int maxEpochs = 100; // число эпох обучения

	Network network(width, height, deep);
	
	network.AddLayer("conv filter_size=3 filters=16");
	network.AddLayer("relu");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("conv filter_size=3 filters=16");
	network.AddLayer("relu");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("maxpool");
	network.AddLayer("dropout p=0.4");

	network.AddLayer("conv filter_size=3 filters=32");
	network.AddLayer("relu");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("conv filter_size=3 filters=32");
	network.AddLayer("relu");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("maxpool");
	network.AddLayer("dropout p=0.4");

	network.AddLayer("fullconnected outputs=128");
	network.AddLayer("batchnormalization");
	network.AddLayer("relu");
	network.AddLayer("dropout p=0.4");

	network.AddLayer("fullconnected outputs=10");
	network.AddLayer("softmax");

	network.PrintConfig(); // выводим конфигурацию сети

	Optimizer optimizer = Optimizer::AdaMax(learningRate); // оптимизатор - Adamax

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	double bestAcc = 0;

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (int i = 0; i < maxEpochs; i++) {
		cout << (i + 1) << ":" << endl;

		optimizer.SetEpoch(i + 1);
		double error = network.Train(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizer, LossType::CrossEntropy); // обучаем в течение одной эпохи

		cout << "loss: " << error << endl;
		double test_acc = loader.Test(network, test, "Test accuracy: ", 10000); // проверяем точность на тестовой выборке

		// если тестовая точность стала выше максимальной
		if (bestAcc <= test_acc) {
			loader.Test(network, train, "Train accuracy: ", 10000); // проверяем точность на обучающей выборке

			bestAcc = test_acc; // обновляем максимальную точность
			network.Save(to_string(test_acc) + ".txt"); // и сохраняем сеть
		}

		cout << "Best accuracy: " << bestAcc << endl; // выводим лучшую точность
		cout << "==================================================================================" << endl;
	}
}
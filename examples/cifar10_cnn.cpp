#include <iostream>
#include <fstream>
#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

int main() {
	string dir = "../dataset/"; // путь к папке с файлами
	string train = dir + "cifar10_train.csv"; // обучающая выборка
	string test = dir + "cifar10_test.csv"; // тестовая выборка
	string labels = dir + "cifar10.txt"; // файл с классами

	int width = 32; // ширина изображений
	int height = 32; // высота изображений
	int deep = 3; // количество каналов

	int trainCount = 50000; // число обучающих примеров (вся выборка)

	double learningRate = 0.009; // скорость обучения
	int batchSize = 64; // размер батча
	int maxEpochs = 50; // число эпох обучения

	Network network(width, height, deep);

	network.AddLayer("conv filter_size=3 filters=16 P=1");
	network.AddLayer("conv filter_size=3 filters=16 P=1");
	network.AddLayer("maxpool");
	network.AddLayer("dropout p=0.2");

	network.AddLayer("conv filter_size=3 filters=32 P=1");
	network.AddLayer("conv filter_size=3 filters=32 P=1");
	network.AddLayer("maxpool");
	network.AddLayer("dropout p=0.2");

	network.AddLayer("conv filter_size=3 filters=64 P=1");
	network.AddLayer("conv filter_size=3 filters=64 P=1");
	network.AddLayer("maxpool");
	network.AddLayer("dropout p=0.2");

	network.AddLayer("fullconnected outputs=128");
	network.AddLayer("batchnormalization");
	network.AddLayer("relu");
	network.AddLayer("dropout p=0.2");
	network.AddLayer("fullconnected outputs=10");
	network.AddLayer("softmax");

	network.PrintConfig(); // выводим конфигурацию сети

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	Optimizer optimizer = Optimizer::SGDm(learningRate);

	double bestAcc = 0;

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (int i = 0; i < maxEpochs; i++) {
		cout << (i + 1) << ":" << endl;

		optimizer.SetEpoch(i + 1);
		double loss = network.Train(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizer, LossType::CrossEntropy); // обучаем в течение одной эпохи


		double testAcc = loader.Test(network, test, "", 10000); // проверяем точность на тестовой выборке

		cout << "loss: " << loss << endl;
		cout << "test accuracy: " << testAcc << endl;

		// если тестовая точность стала выше максимальной
		if (bestAcc <= testAcc) {
			double trainAcc = loader.Test(network, train, "", 10000); // проверяем точность на обучающей выборке

			cout << "train accuracy: " << trainAcc << endl;

			bestAcc = testAcc; // обновляем максимальную точность
			network.Save("cifar10_" + to_string(bestAcc) + ".txt"); // и сохраняем сеть
		}
		else {
			optimizer.ChangeLearningRate(0.9);
		}

		cout << "Best: " << bestAcc << endl << endl; // выводим лучшую точность
		cout << "============================================================================================" << endl;
	}
}
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

	double learningRate = 0.001; // скорость обучения
	int batchSize = 64; // размер батча
	int maxEpochs = 200; // число эпох обучения

	Network network(width, height, deep);

	network.AddLayer("conv filter_size=3 filters=32 P=1");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("conv filter_size=3 filters=32 P=1");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("maxpool");
	network.AddLayer("dropout p=0.2");

	network.AddLayer("conv filter_size=3 filters=64 P=1");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("conv filter_size=3 filters=64 P=1");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("maxpool");
	network.AddLayer("dropout p=0.3");

	network.AddLayer("conv filter_size=3 filters=128 P=1");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("conv filter_size=3 filters=128 P=1");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("maxpool");
	network.AddLayer("dropout p=0.4");

	network.AddLayer("fullconnected outputs=10");
	network.AddLayer("softmax");

	network.PrintConfig(); // выводим конфигурацию сети

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	Optimizer optimizer = Optimizer::Adam(learningRate);

	double bestAcc = 0;
	string augmentation = "shift-x=0.1 shift-y=0.1 br-min=0.8 br-max=1.2 rotate=15";

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (int epoch = 1; epoch <= maxEpochs; epoch++) {
		cout << epoch << " / " << maxEpochs << ":" << endl;

		optimizer.SetEpoch(epoch);
		
		double loss = network.Train(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizer, LossType::CrossEntropy, augmentation); // обучаем в течение одной эпохи
		cout << "loss: " << loss << endl;

		double testAcc = loader.Test(network, test, "", 10000); // проверяем точность на тестовой выборке
		cout << "test accuracy: " << testAcc << endl;

		// если тестовая точность стала выше максимальной
		if (bestAcc <= testAcc) {
			double trainAcc = loader.Test(network, train, "", 10000); // проверяем точность на обучающей выборке

			cout << "train accuracy: " << trainAcc << endl;

			bestAcc = testAcc; // обновляем максимальную точность
			network.Save("cifar10_" + to_string(bestAcc) + ".txt"); // и сохраняем сеть
		}

		cout << "Best: " << bestAcc << endl << endl; // выводим лучшую точность
		cout << "============================================================================================" << endl;

		if (epoch % 10 == 0)
			optimizer.ChangeLearningRate(0.8);
	}
}
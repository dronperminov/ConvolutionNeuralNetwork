#include <iostream>
#include <fstream>
#include "../CNN.hpp"
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

	double learningRate = 0.0008; // скорость обучения
	int batchSize = 64; // размер батча
	int maxEpochs = 50; // число эпох обучения

	CNN cnn(width, height, deep);
	
	cnn.AddLayer("conv filter_size=3 filters=16 P=1");
	cnn.AddLayer("conv filter_size=3 filters=16 P=1");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("dropout p=0.2");

	cnn.AddLayer("conv filter_size=3 filters=32 P=1");
	cnn.AddLayer("conv filter_size=3 filters=32 P=1");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("dropout p=0.2");

	cnn.AddLayer("conv filter_size=3 filters=64 P=1");
	cnn.AddLayer("conv filter_size=3 filters=64 P=1");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("dropout p=0.2");

	cnn.AddLayer("fullconnected outputs=128 activation=relu");
	cnn.AddLayer("dropout p=0.2");
	cnn.AddLayer("fullconnected outputs=10");
	cnn.AddLayer("softmax");

	cnn.PringConfig(); // выводим конфигурацию сети

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные
	
	Optimizer optimizer = Optimizer::SGDm(learningRate);
	
	double bestAcc = 0;

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (int i = 0; i < maxEpochs; i++) {
		cout << (i + 1) << ":" << endl;

		optimizer.SetEpoch(i + 1);
		double loss = cnn.TrainMiniBatch(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizer, ErrorType::CrossEntropy); // обучаем в течение одной эпохи

		cout << "loss: " << loss << endl;

		double testAcc = loader.Test(cnn, test, "Test accuracy: ", 10000); // проверяем точность на тестовой выборке

		// если тестовая точность стала выше максимальной
		if (bestAcc <= testAcc) {
			loader.Test(cnn, train, "Train accuracy: ", 10000); // проверяем точность на обучающей выборке
			bestAcc = testAcc; // обновляем максимальную точность
			cnn.Save(to_string(testAcc) + ".txt"); // и сохраняем сеть
		}
		else {
			optimizer.ChangeLearningRate(0.9);
		}

		cout << "Best: " << bestAcc << endl << endl; // выводим лучшую точность
		cout << "============================================================================================" << endl;
	}
}
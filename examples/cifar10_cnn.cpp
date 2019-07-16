#include <iostream>
#include <fstream>
#include <vector>
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

	double learningRate = 0.0004; // скорость обучения
	int maxEpochs = 100; // число эпох обучения

	CNN cnn(width, height, deep);
	
	cnn.AddLayer("conv filter_size=5 filters=16 P=2");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("conv filter_size=5 filters=20 P=2");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("conv filter_size=5 filters=20 P=2");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("flatten");
	cnn.AddLayer("fullconnected outputs=10 activation=sigmoid");

	cnn.PringConfig(); // выводим конфигурацию сети

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные
	
	// Optimizer optimizer = Optimizer::SGD(learningRate); // оптимизатор - стохастический градиентный спуск
	// Optimizer optimizer = Optimizer::SGDm(learningRate); // оптимизатор - стохастический градиентный спуск с моментом
	// Optimizer optimizer = Optimizer::Adagrad(learningRate); // оптимизатор - адаптивный градиент
	// Optimizer optimizer = Optimizer::Adadelta(); // оптимизатор - адаптивный градиент со скользящим средним
	// Optimizer optimizer = Optimizer::RMSprop(learningRate); // оптимизатор - RMSprop
	// Optimizer optimizer = Optimizer::NAG(learningRate); // оптимизатор - ускоренный градиент Нестерова
	Optimizer optimizer = Optimizer::Adam(learningRate); // оптимизатор - Adam
	
	double best = 0;

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (int i = 0; i < maxEpochs; i++) {
		cout << (i + 1) << ":" << endl;

		optimizer.SetEpoch(i + 1);
		cnn.Train(loader.trainInputData, loader.trainOutputData, 1, optimizer); // обучаем в течение одной эпохи

		double test_acc = loader.Test(cnn, test, "Test accuracy: ", 10000); // проверяем точность на тестовой выборке
		double train_acc = loader.Test(cnn, train, "Train accuracy: ", 10000); // проверяем точность на обучающей выборке

		// если тестовая точность стала выше максимальной
		if (best <= test_acc) {
			best = test_acc; // обновляем максимальную точность
			cnn.Save(to_string(test_acc) + ".txt"); // и сохраняем сеть
		}

		cout << "Best: " << best << endl << endl; // выводим лучшую точность
		cout << "============================================================================================" << endl;
	}
}
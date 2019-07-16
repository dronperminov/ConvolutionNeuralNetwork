#include <iostream>
#include <fstream>
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

	int trainCount = 60000; // число обучающих примеров (вся выборка)

	double learningRate = 0.004; // скорость обучения
	int maxEpochs = 20; // число эпох обучения

	CNN cnn(width, height, deep);
	
	cnn.AddLayer("conv filter_size=5 filters=16 P=2");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("conv filter_size=5 filters=32 P=2");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("flatten");
	cnn.AddLayer("fullconnected outputs=128 activation=sigmoid");
	cnn.AddLayer("fullconnected outputs=10 activation=sigmoid");

	cnn.PringConfig(); // выводим конфигурацию сети

	// Optimizer optimizer = Optimizer::SGD(learningRate); // оптимизатор - стохастический градиентный спуск
	// Optimizer optimizer = Optimizer::SGDm(learningRate); // оптимизатор - стохастический градиентный спуск с моментом
	// Optimizer optimizer = Optimizer::Adagrad(learningRate); // оптимизатор - адаптивный градиент
	// Optimizer optimizer = Optimizer::Adadelta(learningRate); // оптимизатор - адаптивный градиент со скользящим средним
	Optimizer optimizer = Optimizer::NAG(learningRate); // оптимизатор - ускоренный градиент Нестерова

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	double bestAcc = 0;

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (int i = 0; i < maxEpochs; i++) {
		cout << (i + 1) << ":" << endl;

		cnn.Train(loader.trainInputData, loader.trainOutputData, 1, optimizer); // обучаем в течение одной эпохи

		double test_acc = loader.Test(cnn, test, "Test accuracy: ", 10000); // проверяем точность на тестовой выборке
		double train_acc = loader.Test(cnn, train, "Train accuracy: ", 10000); // проверяем точность на обучающей выборке

		// если тестовая точность стала выше максимальной
		if (bestAcc <= test_acc) {
			bestAcc = test_acc; // обновляем максимальную точность
			cnn.Save(to_string(test_acc) + ".txt"); // и сохраняем сеть
		}

		cout << "Best accuracy: " << bestAcc << endl; // выводим лучшую точность
		cout << "==================================================================================" << endl;
	}
}
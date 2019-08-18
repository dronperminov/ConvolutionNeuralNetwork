#include <iostream>
#include <fstream>
#include <vector>
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

	int trainCount = 60000;
	int testCount = 10000;

	Network network("../models/mnist_99.52.txt");

	network.PrintConfig(); // выводим конфигурацию сети

	DataLoader loaderTrain(train, width, height, deep, labels, trainCount); // загружаем обучающие данные
	DataLoader loaderTest(test, width, height, deep, labels, testCount); // загружаем проверочные данные

	LossType lossType = LossType::CrossEntropy; // функция - кросс энтропия
	
	cout << "Test error: " << setprecision(10) << network.GetError(loaderTest.trainInputData, loaderTest.trainOutputData, lossType) << endl;
	double test_acc = loaderTest.Test(network, test, "Test accuracy: ", 10000); // проверяем точность на тестовой выборке

	cout << "Train error: " << setprecision(10) << network.GetError(loaderTrain.trainInputData, loaderTrain.trainOutputData, lossType) << endl;
	double train_acc = loaderTrain.Test(network, train, "Train accuracy: ", 60000); // проверяем точность на обучающей выборке
}
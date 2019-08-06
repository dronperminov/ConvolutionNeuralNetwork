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

	int trainCount = 60000;
	int testCount = 10000;

	CNN cnn("cnn.txt");

	cnn.PringConfig(); // выводим конфигурацию сети

	DataLoader loaderTrain(train, width, height, deep, labels, trainCount); // загружаем обучающие данные
	DataLoader loaderTest(test, width, height, deep, labels, testCount); // загружаем проверочные данные

	ErrorType errorType = ErrorType::CrossEntropy;
	
	cout << "Train error: " << setprecision(10) << cnn.GetError(loaderTrain.trainInputData, loaderTrain.trainOutputData, errorType) << endl;
	double train_acc = loaderTrain.Test(cnn, train, "Train accuracy: ", 60000); // проверяем точность на обучающей выборке

	cout << "Test error: " << setprecision(10) << cnn.GetError(loaderTest.trainInputData, loaderTest.trainOutputData, errorType) << endl;
	double test_acc = loaderTest.Test(cnn, test, "Test accuracy: ", 10000); // проверяем точность на тестовой выборке
}
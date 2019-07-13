#include <iostream>
#include <fstream>
#include "CNN.hpp"
#include "Entities/DataLoader.hpp"

using namespace std;

int main() {
	string dir = "dataset/"; // ���� � ����� � �������
	string train = dir + "mnist_train.csv"; // ��������� �������
	string test = dir + "mnist_test.csv"; // �������� �������
	string labels = dir + "mnist.txt"; // ���� � ��������

	int width = 28; // ������ �����������
	int height = 28; // ������ �����������
	int deep = 1; // ���������� �������

	int trainCount = 60000; // ����� ��������� �������� (��� �������)

	double learningRate = 0.005; // �������� ��������
	int maxEpochs = 10; // ����� ���� ��������

	CNN cnn(width, height, deep);
	
	cnn.AddLayer("conv filter_size=5 filters=16");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("conv filter_size=5 filters=32");
	cnn.AddLayer("maxpool");
	cnn.AddLayer("flatten");
	cnn.AddLayer("fullconnected outputs=128 activation=sigmoid");
	cnn.AddLayer("fullconnected outputs=10 activation=sigmoid");

	cnn.PringConfig(); // ������� ������������ ����

	// Optimizer sgd = Optimizer::SGD(learningRate); // ����������� - �������������� ����������� �����
	// Optimizer sgdm = Optimizer::SGDm(learningRate); // ����������� - �������������� ����������� ����� � ��������
	// Optimizer adagrad = Optimizer::Adagrad(learningRate); // ����������� - ���������� ��������
	// Optimizer adadelta = Optimizer::Adadelta(learningRate); // ����������� - ���������� �������� �� ���������� �������
	Optimizer nag = Optimizer::NAG(learningRate); // ����������� - ���������� �������� ���������

	DataLoader loader(train, width, height, deep, labels, trainCount); // ��������� ��������� ������

	double bestAcc = 0;

	// ��������� �������� � ��������� � ����������� ��������� �����
	for (int i = 0; i < maxEpochs; i++) {
		cout << (i + 1) << ":" << endl;

		cnn.Train(loader.trainInputData, loader.trainOutputData, 1, nag); // ������� � ������� ����� �����

		double test_acc = loader.Test(cnn, test, "Test accuracy: ", 10000); // ��������� �������� �� �������� �������
		double train_acc = loader.Test(cnn, train, "Train accuracy: ", 10000); // ��������� �������� �� ��������� �������

		// ���� �������� �������� ����� ���� ������������
		if (bestAcc < test_acc) {
			bestAcc = test_acc; // ��������� ������������ ��������
			cnn.Save(to_string(test_acc) + ".txt"); // � ��������� ����
		}

		cout << "Best accuracy: " << bestAcc << endl; // ������� ������ ��������
		cout << "==================================================================================" << endl;
	}
}
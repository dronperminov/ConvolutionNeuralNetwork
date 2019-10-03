#include <iostream>
#include <fstream>
#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

vector<string> ResidualBlock(int fc, int fs = 3, int strides = 1, const string &activation = "relu", bool batch_normalization = true, bool conv_first = true) {
	vector<string> block;

	string conv = "conv fc=" + to_string(fc) + " fs=" + to_string(fs) + " S=" + to_string(strides) + " P=" + to_string((fs - 1) / 2);

	if (conv_first) {
		block.push_back(conv);

		if (batch_normalization)
			block.push_back("batchnormalization2D");

		if (activation != "")
			block.push_back(activation);
	}
	else {
		if (batch_normalization)
			block.push_back("batchnormalization2D");

		if (activation != "")
			block.push_back(activation);
	
		block.push_back(conv);
	}

	return block;
}

Network ResNet_v1(int width, int height, int deep, int depth, int classes) {
	if ((depth - 2) % 6 != 0)
		throw runtime_error("depth must be 6n+2");

	int fc = 16;
	int num_res_blocks = (depth - 2) / 6;

	Network network(width, height, deep);

	vector<string> x = ResidualBlock(fc);

	for (size_t i = 0; i < x.size(); i++)
		network.AddLayer(x[i]);

	for (int stack = 0; stack < 3; stack++) {
		for (int res_block = 0; res_block < num_res_blocks; res_block++) {
			int strides = 1;

			if (stack > 0 && res_block == 0)
				strides = 2;
			
			vector<string> y1 = ResidualBlock(fc, 3, strides);
			vector<string> y2 = ResidualBlock(fc, 3, 1, "");

			if (stack > 0 && res_block == 0)
				x = ResidualBlock(fc, 1, strides, "", false);
			else
				x = { "identity" };

			vector<string> y;
			y.insert(y.end(), y1.begin(), y1.end());
			y.insert(y.end(), y2.begin(), y2.end());

			network.AddBlock({ y, x }, "sum");
			network.AddLayer("relu");
		}

		fc *= 2;
	}

	network.AddLayer("avgpooling scale=8");
	network.AddLayer("fullconnected outputs=" + to_string(classes));
	network.AddLayer("softmax");

	return network;
}

Network ResNet_v2(int width, int height, int deep, int depth, int classes) {
	if ((depth - 2) % 9 != 0)
		throw runtime_error("depth must be 9n+2");

	int fc = 16;
	int num_res_blocks = (depth - 2) / 9;

	Network network(width, height, deep);

	vector<string> x = ResidualBlock(fc);

	for (size_t i = 0; i < x.size(); i++)
		network.AddLayer(x[i]);

	for (int stage = 0; stage < 3; stage++) {
		int num_filters_out = fc;

		for (int res_block = 0; res_block < num_res_blocks; res_block++) {
			string activation = "relu";
			bool batch_normalization = true;
			int strides = 1;

			if (stage == 0) {
				num_filters_out = fc * 4;

				if (res_block == 0) {
					activation = "";
					batch_normalization = false;
				}
			}
			else {
				num_filters_out = fc * 2;

				if (res_block == 0)
					strides = 2;
			}

			vector<string> y1 = ResidualBlock(fc, 1, strides, activation, batch_normalization, false);
			vector<string> y2 = ResidualBlock(fc, 3, 1, "relu", true, false);
			vector<string> y3 = ResidualBlock(num_filters_out, 1, 1, "relu", true, false);

			if (res_block == 0) {
				x = ResidualBlock(num_filters_out, 1, strides, "", false, true);
			}
			else {
				x = { "identity" };
			}

			vector<string> y;
			y.insert(y.end(), y1.begin(), y1.end());
			y.insert(y.end(), y2.begin(), y2.end());
			y.insert(y.end(), y3.begin(), y3.end());

			network.AddBlock({ y, x }, "sum");
			
		}
		
		fc = num_filters_out;
	}

	network.AddLayer("batchnormalization2D");
	network.AddLayer("relu");
	network.AddLayer("avgpooling scale=8");
	network.AddLayer("fullconnected outputs=" + to_string(classes));
	network.AddLayer("softmax");

	return network;
}

int main() {
	string dir = "../dataset/"; // путь к папке с файлами
	string train = dir + "cifar10_train.csv"; // обучающая выборка
	string test = dir + "cifar10_test.csv"; // тестовая выборка
	string labels = dir + "cifar10.txt"; // файл с классами

	int width = 32; // ширина изображений
	int height = 32; // высота изображений
	int deep = 3; // количество каналов

	int trainCount = 50000; // число обучающих примеров (вся выборка)

	float learningRate = 0.001; // скорость обучения
	int batchSize = 64; // размер батча
	int maxEpochs = 200; // число эпох обучения

	int version = 1;
	int n = 3;
	int depth;

	if (version == 1) {
		depth = 6 * n + 2;
	}
	else {
		depth = 9 * n + 2;
	}

	Network network = version == 1 ? ResNet_v1(width, height, deep, depth, 10) : ResNet_v2(width, height, deep, depth, 10);

	network.PrintConfig(); // выводим конфигурацию сети

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем обучающие данные

	Optimizer optimizer = Optimizer::Adam(learningRate);

	float bestAcc = 0;
	string augmentation = "shift-x=0.1 shift-y=0.1 rotate=15";

	// запускаем обучение с проверкой и сохранением наилучших сетей
	for (int epoch = 1; epoch <= maxEpochs; epoch++) {
		cout << epoch << " / " << maxEpochs << ":" << endl;

		optimizer.SetEpoch(epoch);
		
		float loss = network.Train(loader.trainInputData, loader.trainOutputData, batchSize, 1, optimizer, LossFunction::CrossEntropy(), augmentation); // обучаем в течение одной эпохи
		cout << "loss: " << loss << endl;

		float testAcc = loader.Test(network, test, "", 10000); // проверяем точность на тестовой выборке
		cout << "test accuracy: " << testAcc << endl;

		// если тестовая точность стала выше максимальной
		if (bestAcc <= testAcc) {
			float trainAcc = loader.Test(network, train, "", 10000); // проверяем точность на обучающей выборке

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
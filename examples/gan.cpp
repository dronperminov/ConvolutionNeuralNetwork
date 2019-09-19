#include <iostream>
#include <random>

#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

default_random_engine generator;
normal_distribution<double> distribution(0.0, 1.0);

// управление обучаемостью слоёв сети
void SetTrainable(Network &network, vector<int> &layers, bool trainable) {
	for (size_t i = 0; i < layers.size(); i++)
		network.SetLayerLearnable(layers[i], trainable);
}

// генерация реальных данных
vector<Volume> GenerateRealExamples(const vector<Volume> &realData, int batchSize) {
	vector<Volume> real;

	for (int i = 0; i < batchSize; i++)
		real.push_back(realData[rand() % realData.size()]);

	return real;
}

// генерация шума
vector<Volume> GenerateNoise(int batchSize, int randomDim) {
	vector<Volume> noise(batchSize, Volume(1, 1, randomDim));

	for (int i = 0; i < batchSize; i++)
		for (int j = 0; j < randomDim; j++)
			noise[i][j] = distribution(generator);

	return noise;
}

// генерация случайных данных
vector<Volume>& GenerateFakeExamples(Network &gan, int randomDim, int batchSize, int generatorEnd) {
	vector<Volume> noise = GenerateNoise(batchSize, randomDim);

	return gan.GetOutputAtLayer(noise, generatorEnd);
}

void NormImage(Volume &volume) {
	int total = volume.Width() * volume.Height() * volume.Deep();

	for (int i = 0; i < total; i++)
		volume[i] = volume[i] / 127.5 - 1;
}

void DenormImage(Volume &volume) {
	int total = volume.Width() * volume.Height() * volume.Deep();

	for (int i = 0; i < total; i++)
		volume[i] = (volume[i] + 1) * 127.5;
}

// сохранение сгенерированных картинок  сети
void SaveExamples(Network &gan, int randomDim, int count, int generatorEnd, int epoch, int w, int h, int d) {
	vector<Volume>& fakeData = GenerateFakeExamples(gan, randomDim, count, generatorEnd);
		
	string path = "epoch" + to_string(epoch);
	system((string("mkdir ") + path).c_str());

	for (int i = 0; i < count; i++) {
		fakeData[i].Reshape(w, h, d);
		DenormImage(fakeData[i]);
		fakeData[i].Save(path + "/" + to_string(i), 2);
	}

	gan.Save(path + "/gan.txt");
}

int main() {
	string dir = "../dataset/"; // путь к папке с файлами
	string train = dir + "mnist_train.csv"; // обучающая выборка
	string test = dir + "mnist_test.csv"; // тестовая выборка
	string labels = dir + "mnist.txt"; // файл с классами

	int width = 28; // ширина изображений
	int height = 28; // высота изображений
	int deep = 1; // количество каналов
	int total = width * height * deep; // общее число пикселей

	int trainCount = 10000; // число обучающих примеров
	double learningRate = 0.0004; // скорость обучения
	int randomDim = 10; // размерность шумового входа

	int epochs = 1000; // количество эпох
	int batchSize = 64;
	int batchCount = trainCount / batchSize; // количество проходов для эпохи

	Network gan(1, 1, randomDim); // создаём сеть

	// генератор
	gan.AddLayer("fc outputs=256");
	gan.AddLayer("leakyrelu alpha=0.2");

	gan.AddLayer("fc outputs=512");
	gan.AddLayer("leakyrelu alpha=0.2");

	gan.AddLayer("fc outputs=" + to_string(total));
	gan.AddLayer("tanh");

	// дискриминатор
	gan.AddLayer("fc outputs=512");
	gan.AddLayer("leakyrelu alpha=0.2");
	gan.AddLayer("dropout p=0.3");

	gan.AddLayer("fc outputs=256");
	gan.AddLayer("leakyrelu alpha=0.2");
	gan.AddLayer("dropout p=0.3");

	gan.AddLayer("fc outputs=1 activation=sigmoid");

	gan.PrintConfig(); // выводим конфигурацию сети

	vector<int> generatorLayers = { 0, 1, 2, 3, 4, 5 };
	vector<int> discriminatorLayers = { 6, 7, 8, 9, 10, 11, 12 };
	int discriminatorStart = discriminatorLayers[0]; // слой, с которого начинается дискриминатор

	LossType loss = LossType::BinaryCrossEntropy; // функция потерь - бинарная перекрёстная энтропия
	Optimizer optimizer = Optimizer::Adam(learningRate, 0, 0.5); // оптимизатор

	DataLoader loader(train, width, height, deep, labels, trainCount, 1); // загружаем обучающие данные

	for (int i = 0; i < trainCount; i++)
		NormImage(loader.trainInputData[i]);

	for (int epoch = 1; epoch <= epochs; epoch++) {
		double genLoss = 0; // ошибка генератора
		double disLoss = 0; // ошибка дискриминатора
		int passed = 0; // количество просмотренных изображений

		for (int batch = 0; batch < batchCount; batch++) {
			vector<Volume> realData = GenerateRealExamples(loader.trainInputData, batchSize); // выбираем случайные реальные картинки
			vector<Volume> fakeData = GenerateFakeExamples(gan, randomDim, batchSize, discriminatorStart - 1); // генерируем картинки из шума

			vector<Volume> realOutputs(batchSize, Volume(1, 1, 1));
			vector<Volume> fakeOutputs(batchSize, Volume(1, 1, 1));

			for (int i = 0; i < batchSize; i++) {
				realOutputs[i][0] = 1; // метка реальных картинок - 1
				fakeOutputs[i][0] = 0; // метка сгенерированных картинок - 0
			}			

			SetTrainable(gan, generatorLayers, false);
			SetTrainable(gan, discriminatorLayers, true); // включаем обучение дискриминатора
			disLoss += gan.TrainOnBatch(realData, realOutputs, optimizer, loss, discriminatorStart); // обучаем дискриминатор на реальных изображениях
			disLoss += gan.TrainOnBatch(fakeData, fakeOutputs, optimizer, loss, discriminatorStart); // обучаем дискриминатор на сгенерированных изображениях

			vector<Volume> noise = GenerateNoise(batchSize, randomDim); // генерируем шум

			SetTrainable(gan, generatorLayers, true);
			SetTrainable(gan, discriminatorLayers, false); // отключаем обучение дискриминатора
			genLoss += gan.TrainOnBatch(noise, realOutputs, optimizer, loss); // обучаем генератор

			passed += batchSize;
			cout << passed << "/" << trainCount << ", dis loss: " << disLoss / (2 * passed) << ", gen loss: " << genLoss / passed << "\r";
		}

		cout << "epoch " << epoch << "    , dis loss: " << disLoss / (2 * passed) << ", gen loss: " << genLoss / passed << endl;

		if (epoch == 1 || epoch % 10 == 0) {
			SaveExamples(gan, randomDim, batchSize, discriminatorStart - 1, epoch, width, height, deep);
		}
	}
}
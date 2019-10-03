#include <iostream>
#include <random>

#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

default_random_engine generator;
normal_distribution<double> distribution(0.0, 1.0);

// генерация шума
vector<Volume> GenerateNoise(int batchSize, int latentDim) {
	vector<Volume> noise(batchSize, Volume(1, 1, latentDim));

	for (int i = 0; i < batchSize; i++)
		for (int j = 0; j < latentDim; j++)
			noise[i][j] = distribution(generator);

	return noise;
}

// создаём кодировщик
int CreateEncoder(Network &network, int latentDim) {
	network.AddLayer("conv fc=16 fs=3 S=2 P=same");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("leakyrelu alpha=0.2");

	network.AddLayer("conv fc=32 fs=3 S=2 P=same");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("leakyrelu alpha=0.2");

	network.AddLayer("conv fc=64 fs=3 S=2 P=same");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("leakyrelu alpha=0.2");

	network.AddLayer("conv fc=128 fs=3 S=2 P=same");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("leakyrelu alpha=0.2");

	network.AddLayer("fc outputs=128 activation=relu");
	network.AddLayer("sampler outputs=" + to_string(latentDim) + " KL=0");

	return network.LayersCount();
}

void CreateDecoder(Network &network, int originalDim) {
	network.AddLayer("fc outputs=2048");
	network.AddLayer("reshape width=4 height=4 deep=128");
	network.AddLayer("leakyrelu alpha=0.2");

	network.AddLayer("convtransposed fc=128 fs=4 S=2 P=same");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("leakyrelu alpha=0.2");

	network.AddLayer("convtransposed fc=64 fs=4 S=2 P=same");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("leakyrelu alpha=0.2");

	network.AddLayer("convtransposed fc=32 fs=4 S=2 P=same");
	network.AddLayer("batchnormalization2D");
	network.AddLayer("leakyrelu alpha=0.2");

	network.AddLayer("convtransposed fc=3 fs=4 S=2 P=same");
	network.AddLayer("tanh");
}

// перевод к пиксельным значениям (* 255) и изменение под заданный размер
void DenormImage(Volume &volume, VolumeSize size) {
	int total = volume.Width() * volume.Height() * volume.Deep();

	for (int i = 0; i < total; i++)
		volume[i] *= 255;

	volume.Reshape(size.width, size.height, size.deep);
}

// получение индекса максимума в объёме длины n
int ArgMax(const Volume &volume, int n) {
	int imax = 0;

	for (int i = 1; i < n; i++)
		if (volume[i] > volume[imax])
			imax = i;

	return imax;
}

// простановка значения объёма коэффициентов из hsv пространства
void SetHsv(Volume& img, int x, int y, double H, double V) {
    while (H < 0)
        H += 360;

    while (H >= 360)
        H -= 360;

    double R, G, B;

    if (V <= 0) {
    	img(0, y, x) = 0;
    	img(1, y, x) = 0;
    	img(2, y, x) = 0;
        return;
    }

    double hf = H / 60.0;

    int i = (int)floor(hf);

    double f = hf - i;
    double pv = 0;
    double qv = V * (1 - f);
    double tv = V * f;

    switch (i) {
        case 0: R = V; G = tv; B = pv; break;
        case 1: R = qv; G = V; B = pv; break;
        case 2: R = pv; G = V; B = tv; break;
        case 3: R = pv; G = qv; B = V; break;
        case 4: R = tv; G = pv; B = V; break;
        case 5: R = V; G = pv; B = qv; break;
        case 6: R = V; G = tv; B = pv; break;
        case -1: R = V; G = pv; B = qv; break;
        default: R = G = B = V; break;
    }

    img(0, y, x) = R * 255;
    img(1, y, x) = G * 255;
    img(2, y, x) = B * 255;
}

// сохранение сгенерированных картинок сети
void SaveExamples(Network &network, int latentDim, const string& path, int decoderStart, VolumeSize size, int count) {
	vector<Volume> noise = GenerateNoise(count, latentDim);
	vector<Volume> results = network.GetOutputFromLayer(noise, decoderStart);

	for (int i = 0; i < count; i++) {
		DenormImage(results[i], size);		
		results[i].Save(path + "/" + to_string(i), 2);
	}
}

// сохранение результатов восстановления
void SavePredictions(Network &network, const string& path, const vector<Volume> &realData, VolumeSize size, int count) {
	vector<Volume> input; 

	for (int i = 0; i < count; i++)
		input.push_back(realData[rand() % realData.size()]);

	vector<Volume> predictions = network.GetOutput(input);

	for (int i = 0; i < count; i++) {
		DenormImage(predictions[i], size);
		DenormImage(input[i], size);
		
		input[i].Save(path + "/pred" + to_string(i) + "_x", 2);
		predictions[i].Save(path + "/pred" + to_string(i) + "_y", 2);
	}
}

int main() {
	string dir = "../dataset/anime-faces/"; // путь к папке с датасетом лиц из аниме
	string train = dir + "anime-faces-train.csv"; // обучающая выборка
	string labels = dir + "anime-faces.txt"; // файл с классами

	VolumeSize size;
	size.width = 64; // ширина изображений
	size.height = 64; // высота изображений
	size.deep = 3; // количество каналов

	int originalDim = size.width * size.height * size.deep; // размер оригинальной картинки
	int latentDim = 128; // размерность латентного пространства

	int trainCount = 2048; // количество обучающих примеров
	int epochs = 50000; // количество эпох
	int batchSize = 64; // размер батча

	int interpolateCount = 30; // количество изображений в ряду для интерполяции
	int interpolateRadius = 4; // радиус интерполяции латентного пространства

	Network network(size.width, size.height, size.deep); // создаём сеть

	int decoderStart = CreateEncoder(network, latentDim); // добавляем кодировщик
	CreateDecoder(network, originalDim); // добавляем декодировщик
	
	network.PrintConfig(); // выводим конфигурацию сети
	
	DataLoader loader(train, size.width, size.height, size.deep, labels, trainCount, 255); // загружаем обучающие данные
	Optimizer optimizer = Optimizer::Adam(0.004); // оптимизатор

	SamplerLayer *sampler = (SamplerLayer*) network.GetLayer(decoderStart - 1);

	double kl = 0.001;
	double dkl = 1.05;

	for (int epoch = 1; epoch <= epochs; epoch++) {
		optimizer.SetEpoch(epoch);
		sampler->SetKL(kl);

		double error = network.Train(loader.trainInputData, loader.trainInputData, batchSize, 1, optimizer, LossFunction::MSE()); // обучаем в течение одной эпохи

		cout << epoch << ": " << error << endl;

		if (epoch % 5 == 0) {			
			string path = "vae_epoch" + to_string(epoch);
			string cmd = "mkdir " + path;
			system(cmd.c_str());

			SaveExamples(network, latentDim, path, decoderStart, size, batchSize * 2);
			SavePredictions(network, path, loader.trainInputData, size, batchSize);

			network.Save(path + "/vae.txt");
		}

		if (epoch < 20)
			optimizer.ChangeLearningRate(epoch < 10 ? 0.8 : 0.9);

		kl *= dkl;

		if (kl > 1)
			kl = 1;
	}
}
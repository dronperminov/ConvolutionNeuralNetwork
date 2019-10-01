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
	network.AddLayer("fc outputs=512 activation=relu");
	network.AddLayer("sampler outputs=" + to_string(latentDim));

	return network.LayersCount();
}

void CreateDecoder(Network &network, int originalDim) {
	network.AddLayer("fc outputs=512 activation=relu");
	network.AddLayer("fc outputs=" + to_string(originalDim) + " activation=sigmoid");
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

// интерполяция в латентном пространстве
void Interpolate(Network &network, int latentDim, const string& path, int decoderStart, VolumeSize size, int n, double radius) {
	if (latentDim != 2) {
		cout << "Unable to interpolate. Latent dimension != 2" << endl;
		return;
	}

	Volume interpolate(size.width * n, size.height * n, size.deep);
	Volume zSample(1, 1, latentDim);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			zSample[0] = (j * 2.0 / n - 1) * radius;
			zSample[1] = (i * 2.0 / n - 1) * radius;

			Volume output = network.GetOutputFromLayer(zSample, decoderStart);
			output.Reshape(size.width, size.height, size.deep);

			for (int x = 0; x < size.width; x++)
				for (int y = 0; y < size.height; y++)
					interpolate(0, i * size.height + y, j * size.width + x) = output(0, y, x) * 255;
		}
	}

	interpolate.Save(path + "/interpolation");
}

// отображение распределения латентного пространства
void ShowLatentDistribution(Network &network, int latentDim, const string &path, int decoderStart, const std::vector<Volume> &testData, const std::vector<Volume> &classData, int width, int height, int classes) {
	if (latentDim != 2) {
		cout << "Unable to show latent distribution. Latent dimension != 2" << endl;
		return;
	}

	int addedHeight = 50; // дополнительная высота для меток классов
	int degrees = 300; // максимальный градус HSV

	double zMin, zMax;
	vector<Volume> z;
	vector<int> nums;

	for (size_t i = 0; i < testData.size(); i++) {
		Volume &out = network.GetOutputAtLayer({ testData[i] }, decoderStart - 1)[0];

		if (i == 0) {
			zMax = max(out[0], out[1]);
			zMin = min(out[0], out[1]);
		}
		else {
			zMin = min(zMin, min(out[0], out[1]));
			zMax = max(zMax, max(out[0], out[1]));
		}

		z.push_back(out);
		nums.push_back(ArgMax(classData[i], classes));
	}
	
	Volume latentDistribution(width, height + addedHeight, 3);

	for (size_t i = 0; i < z.size(); i++) {
		int x = (z[i][0] - zMin) / (zMax - zMin) * width;
		int y = (z[i][1] - zMin) / (zMax - zMin) * width;

		SetHsv(latentDistribution, x, y, degrees * nums[i] / classes, 0.8);
	}

	for (int i = 0; i < classes; i++) {
		int pos = 10 + (width - 20.0) / (classes - 1) * i;

		for (int y = height + 10; y <= height + 40; y++)
			for (int x = pos - 5; x <= pos + 5; x++)
				SetHsv(latentDistribution, x, y, degrees * i / classes, 0.8);
	}

	latentDistribution.Save(path + "/distribution");
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
	string dir = "../dataset/"; // путь к папке с датасетом лиц из аниме
	string train = dir + "mnist_train.csv"; // обучающая выборка
	string labels = dir + "mnist.txt"; // файл с классами

	VolumeSize size;
	size.width = 28; // ширина изображений
	size.height = 28; // высота изображений
	size.deep = 1; // количество каналов

	int originalDim = size.width * size.height * size.deep; // размер оригинальной картинки
	int latentDim = 2; // размерность латентного пространства

	int classesCount = 10; // количество классов
	int trainCount = 50000; // количество обучающих примеров
	int epochs = 50; // количество эпох
	int batchSize = 64; // размер батча

	int interpolateCount = 30; // количество изображений в ряду для интерполяции
	int interpolateRadius = 4; // радиус интерполяции латентного пространства

	Network network(size.width, size.height, size.deep); // создаём сеть

	int decoderStart = CreateEncoder(network, latentDim); // добавляем кодировщик
	CreateDecoder(network, originalDim); // добавляем декодировщик
	
	network.PrintConfig(); // выводим конфигурацию сети
	
	DataLoader loader(train, size.width, size.height, size.deep, labels, trainCount, 255); // загружаем обучающие данные
	Optimizer optimizer = Optimizer::Adam(0.002); // оптимизатор

	for (int epoch = 1; epoch <= epochs; epoch++) {
		optimizer.SetEpoch(epoch);
		double error = network.Train(loader.trainInputData, loader.trainInputData, batchSize, 1, optimizer, LossFunction::MSE()); // обучаем в течение одной эпохи

		cout << epoch << ": " << error << endl;

		if (epoch % 5 == 0) {			
			string path = "vae_epoch" + to_string(epoch);
			string cmd = "mkdir " + path;
			system(cmd.c_str());

			ShowLatentDistribution(network, latentDim, path, decoderStart, loader.trainInputData, loader.trainOutputData, 300, 300, classesCount);
			Interpolate(network, latentDim, path, decoderStart, size, interpolateCount, interpolateRadius);
			SaveExamples(network, latentDim, path, decoderStart, size, batchSize * 2);
			SavePredictions(network, path, loader.trainInputData, size, batchSize);

			network.Save(path + "/vae.txt");
		}
	}
}
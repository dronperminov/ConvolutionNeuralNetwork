#include <iostream>
#include <random>

#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

default_random_engine generator;
normal_distribution<double> distribution(0.0, 1.0);

// управление обучаемостью слоёв сети
void SetTrainable(Network &network, int start, int end, bool trainable) {
	for (int i = start; i < end; i++)
		network.SetLayerLearnable(i, trainable);
}

// генератор
int AddGenerator(Network &gan) {
	gan.AddLayer("fc outputs=2048");
	gan.AddLayer("reshape width=8 height=8 deep=32");
	gan.AddLayer("leakyrelu alpha=0.01");

	gan.AddLayer("convtransposed fc=64 fs=4 P=same S=2");
	gan.AddLayer("leakyrelu alpha=0.01");

	gan.AddLayer("convtransposed fc=128 fs=4 P=same S=2");
	gan.AddLayer("leakyrelu alpha=0.01");

	gan.AddLayer("convtransposed fc=3 fs=4 P=same S=2");
	gan.AddLayer("tanh");

	return gan.LayersCount();
}

// дискриминатор
int AddDiscriminator(Network &gan) {
	gan.AddLayer("conv fc=16 fs=3 S=2 P=same");
	gan.AddLayer("leakyrelu alpha=0.2");

	gan.AddLayer("conv fc=32 fs=3 S=2 P=same");
	gan.AddLayer("leakyrelu alpha=0.2");

	gan.AddLayer("conv fc=32 fs=3 S=2 P=same");
	gan.AddLayer("leakyrelu alpha=0.2");

	gan.AddLayer("conv fc=64 fs=3 S=2 P=same");
	gan.AddLayer("leakyrelu alpha=0.2");

	gan.AddLayer("dropout p=0.4");

	gan.AddLayer("fc outputs=1 activation");
	gan.AddLayer("sigmoid");

	return gan.LayersCount();
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
void SaveExamples(Network &gan, int randomDim, int count, int generatorEnd, int epoch) {
	vector<Volume>& fakeData = GenerateFakeExamples(gan, randomDim, count, generatorEnd);
		
	string path = "epoch" + to_string(epoch);
	system((string("mkdir ") + path).c_str());

	for (int i = 0; i < count; i++) {
		DenormImage(fakeData[i]);
		
		fakeData[i].Save(path + "/" + to_string(i), 2);
	}

	gan.Save(path + "/gan.txt");
}

double Accuracy(vector<Volume> &output, vector<Volume> &targets) {
	double acc = 0;

	for (int i = 0; i < output.size(); i++) {
		double v1 = output[i][0] > 0.5 ? 1 : 0;
		double v2 = targets[i][0] > 0.5 ? 1 : 0;

		acc += v1 == v2;
	}

	return acc / output.size();
}

int main() {
	string dir = "../dataset/"; // путь к папке с датасетом лиц из аниме
	string train = dir + "anime-faces-train.csv"; // обучающая выборка
	string labels = dir + "anime-faces.txt"; // файл с классами

	int width = 64; // ширина изображений
	int height = 64; // высота изображений
	int deep = 3; // количество каналов

	int trainCount = 1024; // число обучающих примеров
	int randomDim = 100; // размерность шумового входа

	int epochs = 1000; // количество эпох
	int batchSize = 128; // размер батча

	int halfBatch = batchSize / 2;
	int batchCount = trainCount / batchSize; // количество проходов для эпохи

	Network gan(1, 1, randomDim); // создаём сеть

	int endGenerator = AddGenerator(gan); // генератор
	int endDiscriminator = AddDiscriminator(gan); // дискриминатор

	gan.PrintConfig(); // выводим конфигурацию сети

	LossType loss = LossType::BinaryCrossEntropy; // функция потерь - бинарная перекрёстная энтропия
	
	Optimizer genOptimizer = Optimizer::Adam(0.004, 0, 0.5); // оптимизатор
	Optimizer disOptimizer = Optimizer::Adam(0.003, 0, 0.5); // оптимизатор

	DataLoader loader(train, width, height, deep, labels, trainCount, 1); // загружаем обучающие данные

	for (int i = 0; i < trainCount; i++)
		NormImage(loader.trainInputData[i]);

	for (int epoch = 1; epoch <= epochs; epoch++) {
		double genLoss = 0; // ошибка генератора
		double disLoss = 0; // ошибка дискриминатора
		int passed = 0; // количество просмотренных изображений

		TimePoint t0 = Time::now();

		for (int batch = 0; batch < batchCount; batch++) {
			vector<Volume> realData = GenerateRealExamples(loader.trainInputData, halfBatch); // выбираем случайные реальные картинки
			vector<Volume> fakeData = GenerateFakeExamples(gan, randomDim, halfBatch, endGenerator - 1); // генерируем картинки из шума
			vector<Volume> noise = GenerateNoise(batchSize, randomDim); // генерируем шум

			vector<Volume> realOutputs(halfBatch, Volume(1, 1, 1));
			vector<Volume> fakeOutputs(halfBatch, Volume(1, 1, 1));
			vector<Volume> genOutputs(batchSize, Volume(1, 1, 1));

			for (int i = 0; i < halfBatch; i++) {
				realOutputs[i][0] = 0.9 + distribution(generator) / 10.0; // метка реальных картинок
				fakeOutputs[i][0] = 0.1 + distribution(generator) / 10.0; // метка сгенерированных картинок
			}

			for (int i = 0; i < batchSize; i++)
				genOutputs[i][0] = 1; // метка изображений генератора

			SetTrainable(gan, 0, endGenerator, false);
			SetTrainable(gan, endGenerator, endDiscriminator, true);
			
			double disLoss1 = gan.TrainOnBatch(realData, realOutputs, disOptimizer, loss, endGenerator) / halfBatch; // обучаем дискриминатор на реальных изображениях
			double realAcc = Accuracy(gan.GetOutput(), realOutputs);

			double disLoss2 = gan.TrainOnBatch(fakeData, fakeOutputs, disOptimizer, loss, endGenerator) / halfBatch; // обучаем дискриминатор на сгенерированных изображениях
			double fakeAcc = Accuracy(gan.GetOutput(), fakeOutputs);

			SetTrainable(gan, 0, endGenerator, true);
			SetTrainable(gan, endGenerator, endDiscriminator, false);
			
			double ganLoss = gan.TrainOnBatch(noise, genOutputs, genOptimizer, loss) / batchSize; // обучаем генератор

			disLoss += (disLoss1 + disLoss2) / 2;
			genLoss += ganLoss;

			passed += batchSize;

			// выводим промежуточную информацию
			ms d = std::chrono::duration_cast<ms>(Time::now() - t0);
			double dt = (double) d.count() / passed;
			double t = (batchSize * batchCount - passed) * dt;

			cout << batch << "/" << batchCount;
			cout << "\tdis (real): " << disLoss1 << " " << realAcc * 100 << "%";
			cout << "\tdis (fake): " << disLoss2 << " " << fakeAcc * 100 << "%";
			cout << "\tgen: " << ganLoss;
			cout << "\tleft: " << TimeSpan(t);
			cout << ", total time: " << TimeSpan(dt * batchSize * batchCount) << "\n";
		}

		cout << "epoch " << epoch << ", dis loss: " << disLoss / batchCount << ", gen loss: " << genLoss / batchCount << endl;

		if (epoch < 10 || epoch % 5 == 0)
			SaveExamples(gan, randomDim, batchSize, endGenerator - 1, epoch);
	}
}
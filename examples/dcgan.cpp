#include <iostream>
#include <random>

#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

default_random_engine generator;
normal_distribution<double> distribution(0.0, 1.0);

// создание генератора
Network CreateGenerator(int latentDim, int width, int height, int deep) {
	Network generator(1, 1, latentDim);

	generator.AddLayer("fc outputs=2048");
	generator.AddLayer("reshape width=8 height=8 deep=32");
	generator.AddLayer("leakyrelu alpha=0.01");

	generator.AddLayer("convtransposed fc=64 fs=4 P=same S=2");
	generator.AddLayer("leakyrelu alpha=0.01");

	generator.AddLayer("convtransposed fc=128 fs=4 P=same S=2");
	generator.AddLayer("leakyrelu alpha=0.01");

	generator.AddLayer("convtransposed fc=3 fs=4 P=same S=2");
	generator.AddLayer("tanh");

	return generator;
}

// создание дискриминатора
Network CreateDiscriminator(int width, int height, int deep) {
	Network discriminator(width, height, deep);

	discriminator.AddLayer("conv fc=16 fs=3 S=2 P=same");
	discriminator.AddLayer("leakyrelu alpha=0.2");

	discriminator.AddLayer("conv fc=32 fs=3 S=2 P=same");
	discriminator.AddLayer("leakyrelu alpha=0.2");

	discriminator.AddLayer("conv fc=32 fs=3 S=2 P=same");
	discriminator.AddLayer("leakyrelu alpha=0.2");

	discriminator.AddLayer("conv fc=64 fs=3 S=2 P=same");
	discriminator.AddLayer("leakyrelu alpha=0.2");

	discriminator.AddLayer("dropout p=0.4");

	discriminator.AddLayer("fc outputs=1 activation");
	discriminator.AddLayer("sigmoid");

	return discriminator;
}

Network CreateGan(int latentDim, Network &generator, Network &discriminator) {
	Network gan(1, 1, latentDim); // создаём gan
	gan.AddNetwork(generator);
	gan.AddNetwork(discriminator);

	// у gan дискриминатор не обучается
	for (int i = generator.LayersCount(); i < gan.LayersCount(); i++)
		gan.SetLayerLearnable(i, false);

	return gan;
}

// генерация реальных данных
vector<Volume> GenerateRealExamples(const vector<Volume> &realData, int batchSize) {
	vector<Volume> real;

	for (int i = 0; i < batchSize; i++)
		real.push_back(realData[rand() % realData.size()]);

	return real;
}

// генерация шума
vector<Volume> GenerateNoise(int batchSize, int latentDim) {
	vector<Volume> noise(batchSize, Volume(1, 1, latentDim));

	for (int i = 0; i < batchSize; i++)
		for (int j = 0; j < latentDim; j++)
			noise[i][j] = distribution(generator);

	return noise;
}

// генерация случайных данных
vector<Volume>& GenerateFakeExamples(Network &generator, int latentDim, int batchSize) {
	vector<Volume> noise = GenerateNoise(batchSize, latentDim);

	return generator.GetOutput(noise);
}

// нормализация изображения
void NormImage(Volume &volume) {
	int total = volume.Width() * volume.Height() * volume.Deep();

	for (int i = 0; i < total; i++)
		volume[i] = volume[i] / 127.5 - 1;
}

// денормализация изображения
void DenormImage(Volume &volume) {
	int total = volume.Width() * volume.Height() * volume.Deep();

	for (int i = 0; i < total; i++)
		volume[i] = (volume[i] + 1) * 127.5;
}

// сохранение сгенерированных картинок  сети
void SaveExamples(Network &generator, Network &discriminator, int latentDim, int count, int epoch) {
	vector<Volume>& fakeData = GenerateFakeExamples(generator, latentDim, count);
		
	string path = "epoch" + to_string(epoch);
	system((string("mkdir ") + path).c_str());

	for (int i = 0; i < count; i++) {
		DenormImage(fakeData[i]);
		fakeData[i].Save(path + "/" + to_string(i), 2);
	}

	generator.Save(path + "/generator.txt");
	discriminator.Save(path + "/discriminator.txt");
}

// точность классификации
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
	string dir = "..dataset/anime-faces/"; // путь к папке с датасетом лиц из аниме
	string train = dir + "anime-faces-train.csv"; // обучающая выборка
	string labels = dir + "anime-faces.txt"; // файл с классами

	int width = 64; // ширина изображений
	int height = 64; // высота изображений
	int deep = 3; // количество каналов

	int trainCount = 2048; // число обучающих примеров
	int latentDim = 128; // размерность шумового входа

	int epochs = 10000; // количество эпох
	int batchSize = 16; // размер батча

	int halfBatch = batchSize / 2;
	int batchCount = trainCount / batchSize; // количество проходов для эпохи

	Network generator = CreateGenerator(latentDim, width, height, deep); // генератор
	Network discriminator = CreateDiscriminator(width, height, deep); // дискриминатор
	Network gan = CreateGan(latentDim, generator, discriminator); // создаём gan

	cout << "GENERATOR:" << endl;
	generator.PrintConfig(); // выводим конфигурацию сети

	cout << "DISCRIMINATOR:" << endl;
	discriminator.PrintConfig(); // выводим конфигурацию сети

	cout << "GAN:" << endl;
	gan.PrintConfig(); // выводим конфигурацию сети

	LossFunction loss = LossFunction::BinaryCrossEntropy(); // функция потерь - бинарная перекрёстная энтропия
	
	Optimizer genOptimizer = Optimizer::Adam(0.004, 0, 0.5); // оптимизатор генератора
	Optimizer disOptimizer = Optimizer::Adam(0.005, 0, 0.5); // оптимизатор дискриминатора

	DataLoader loader(train, width, height, deep, labels, trainCount, 1); // загружаем обучающие данные

	for (int i = 0; i < trainCount; i++)
		NormImage(loader.trainInputData[i]);

	for (int epoch = 1; epoch <= epochs; epoch++) {
		double genLoss = 0; // ошибка генератора
		double disLoss = 0; // ошибка дискриминатора
		int passed = 0; // количество просмотренных изображений

		TimePoint t0 = Time::now();

		cout << "+---------+-----------+-----------+----------+-----------+-----------+-----------+-------------+-------------+" << endl;
		cout << "|  batch  | real loss | fake loss | gen loss | real acc. | fake acc. |  gan acc. |  left time  | total time  |" << endl;
		cout << "+---------+-----------+-----------+----------+-----------+-----------+-----------+-------------+-------------+" << endl;

		for (int batch = 0; batch < batchCount; batch++) {
			vector<Volume> realData = GenerateRealExamples(loader.trainInputData, halfBatch); // выбираем случайные реальные картинки
			vector<Volume> fakeData = GenerateFakeExamples(generator, latentDim, halfBatch); // генерируем картинки из шума
			vector<Volume> noise = GenerateNoise(batchSize, latentDim); // генерируем шум

			vector<Volume> realOutputs(halfBatch, Volume(1, 1, 1));
			vector<Volume> fakeOutputs(halfBatch, Volume(1, 1, 1));
			vector<Volume> genOutputs(batchSize, Volume(1, 1, 1));

			for (int i = 0; i < halfBatch; i++) {
				realOutputs[i][0] = 0.9; // метка реальных картинок
				fakeOutputs[i][0] = 0.1; // метка сгенерированных картинок
			}

			for (int i = 0; i < batchSize; i++)
				genOutputs[i][0] = 1; // метка изображений генератора

			double disLoss1 = discriminator.TrainOnBatch(realData, realOutputs, disOptimizer, loss) / halfBatch; // обучаем дискриминатор на реальных изображениях
			double realAcc = Accuracy(discriminator.GetOutput(), realOutputs);

			double disLoss2 = discriminator.TrainOnBatch(fakeData, fakeOutputs, disOptimizer, loss) / halfBatch; // обучаем дискриминатор на сгенерированных изображениях
			double fakeAcc = Accuracy(discriminator.GetOutput(), fakeOutputs);

			double ganLoss = gan.TrainOnBatch(noise, genOutputs, genOptimizer, loss) / batchSize; // обучаем генератор на градиентах дискриминатора
			double ganAcc = Accuracy(gan.GetOutput(), genOutputs);

			disLoss += (disLoss1 + disLoss2) / 2;
			genLoss += ganLoss;

			passed += batchSize;

			// выводим промежуточную информацию
			ms d = std::chrono::duration_cast<ms>(Time::now() - t0);
			double dt = (double) d.count() / passed;
			double t = (batchSize * batchCount - passed) * dt;

			cout << setfill(' ');
			cout << "| " << setw(3) << (batch + 1) << "/" << setw(3) << left << batchCount << " | ";
			cout << setw(9) << right << disLoss1 << " | ";
			cout << setw(9) << disLoss2 << " | ";
			cout << setw(8) << ganLoss << " | ";
			cout << setw(8) << realAcc * 100 << "% | ";
			cout << setw(8) << fakeAcc * 100 << "% | ";
			cout << setw(8) << ganAcc * 100 << "% | ";
			cout << TimeSpan(t) << " | ";
			cout << TimeSpan(dt * batchSize * batchCount) << " |" << endl;
		}

		cout << "+---------+-----------+-----------+----------+-----------+-----------+-----------+-------------+-------------+" << endl;
		cout << "epoch " << epoch << ", dis loss: " << disLoss / batchCount << ", gen loss: " << genLoss / batchCount << endl;

		if (epoch < 10 || epoch % 5 == 0)
			SaveExamples(generator, discriminator, latentDim, batchSize, epoch);
	}
}
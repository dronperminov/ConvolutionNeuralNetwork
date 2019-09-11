#include <iostream>

#include "../Network.hpp"
#include "../Entities/DataLoader.hpp"

using namespace std;

int main() {
	string dir = "../dataset/"; // путь к папке с файлами
	string train = dir + "cifar10_train.csv"; // обучающая выборка
	string test = dir + "cifar10_test.csv"; // тестовая выборка
	string labels = dir + "cifar10.txt"; // файл с классами

	int width = 32; // ширина изображений
	int height = 32; // высота изображений
	int deep = 3; // количество каналов

	int trainCount = 10; // количество ихображений
	int changesCount = 10; // количество изменений для изображения
	int blockSize = 2; // размер пикселя

	DataLoader loader(train, width, height, deep, labels, trainCount); // загружаем изображения

	DataAugmentation augmentation("rotation=10 shift-x=0.15 shift-y=0.15 flip-x=true");
	string path = "imgs/";

	system("mkdir imgs"); // создаём папку

	for (int img = 0; img < trainCount; img++) {
		string name = path + to_string(img);
		Volume& volume = loader.trainInputData[img];

		volume.Save(name, blockSize); // сохраняем исходную картинку

		// выполняем аугментацию и сохраняем результат
		for (int i = 0; i < changesCount; i++) {
			Volume aug = augmentation.Make(volume);
			aug.Save(name + "_augment_" + to_string(i + 1), blockSize);
		}
	}
}
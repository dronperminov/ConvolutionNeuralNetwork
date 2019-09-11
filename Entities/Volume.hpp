#pragma once

#include <iostream>
#include <vector>
#include <string>

#include "GaussRandom.hpp"
#include "Bitmap.hpp"

// размерность объёма
struct VolumeSize {
	int deep; // глубина
	int height; // высота
	int width; // ширина

	std::string ToString() const {
		return std::to_string(width) + "x" + std::to_string(height) + "x" + std::to_string(deep);
	}

	friend std::ostream& operator<<(std::ostream& os, const VolumeSize &size) {
		return os << size.width << " " << size.height << " " << size.deep;
	}

	friend std::istream& operator>>(std::istream& is, VolumeSize &size) {
		return is >> size.width >> size.height >> size.deep;
	}
};

// объём
class Volume {
	VolumeSize size; // размерность объёма
	std::vector<double> values; // значения объёма

	int whd;
	int dh;
	int dw;

	void Init(int width, int height, int deep);

public:
	Volume(int width, int height, int deep); // создание из размеров
	Volume(VolumeSize size);

	double& At(int d, int i, int j); // индексация
	double At(int d, int i, int j) const; // индексация

	double& operator()(int d, int i, int j); // индексация
	double operator()(int d, int i, int j) const; // индексация

	double& operator[](int i); // индексация
	double operator[](int i) const; // индексация

	int Deep() const; // получение глубины
	int Height() const; // получение высоты
	int Width() const; // получение ширины

	VolumeSize GetSize() const; // получение размера

	void FillRandom(GaussRandom& random, double dev, double mean = 0); // заполнение случайными числами
	void Save(const std::string &path, int blockSize = 1) const; // сохранение в виде картинки

	friend std::ostream& operator<<(std::ostream& os, const Volume &volume);
};

void Volume::Init(int width, int height, int deep) {
	size.width = width;
	size.height = height;
	size.deep = deep;

	whd = width * height * deep;
	dh = deep * height;
	dw = deep * width;

	values = std::vector<double>(deep * height * width, 0);
}

// создание из размеров
Volume::Volume(int width, int height, int deep) {
	Init(width, height, deep);	
}

// создание из размера
Volume::Volume(VolumeSize size) {
	Init(size.width, size.height, size.deep);
}

// индексация
double& Volume::At(int d, int i, int j) {
	return values[i * dw + j * size.deep + d];
}

// индексация
double Volume::At(int d, int i, int j) const {
	return values[i * dw + j * size.deep + d];
}

// индексация
double& Volume::operator()(int d, int i, int j) {
	return values[i * dw + j * size.deep + d];
}

// индексация
double Volume::operator()(int d, int i, int j) const {
	return values[i * dw + j * size.deep + d];
}

// индексация
double& Volume::operator[](int i) {
	return values[i];
}

// индексация
double Volume::operator[](int i) const {
	return values[i];
}

// получение глубины
int Volume::Deep() const {
	return size.deep;
}

// получение высоты
int Volume::Height() const {
	return size.height;
}

// получение ширины
int Volume::Width() const {
	return size.width;
}

// получение размера
VolumeSize Volume::GetSize() const {
	return size;
}

// заполнение случайными числами
void Volume::FillRandom(GaussRandom& random, double dev, double mean) {
	for (int i = 0; i < whd; i++)
		values[i] = random.Next(dev, mean);
}

// сохранение в виде картинки
void Volume::Save(const std::string &path, int blockSize) const {
	double min = values[0];
	double max = values[0];

	for (size_t i = 0; i < values.size(); i++) {
		if (values[i] < min)
			min = values[i];

		if (values[i] > max)
			max = values[i];
	}

	int width = size.width * blockSize;
	int height = size.height * blockSize;

	if (size.deep == 3) {
		BitmapImage image(width, height);

		for (int y = 0; y < size.height; y++) {
			for (int x = 0; x < size.width; x++) {
				int r = (At(0, y, x) - min) / (max - min) * 255;
				int g = (At(1, y, x) - min) / (max - min) * 255;
				int b = (At(2, y, x) - min) / (max - min) * 255;

				for (int i = 0; i < blockSize; i++)
					for (int j = 0; j < blockSize; j++)
						image.set_pixel(x * blockSize + j, y * blockSize + i, r, g, b);
			}
		}
		
		image.save_image(path + ".bmp");
	}
	else {
		for (int d = 0; d < size.deep; d++) {
			BitmapImage image(width, height);

			for (int y = 0; y < size.height; y++) {
				for (int x = 0; x < size.width; x++) {
					int br = (At(d, y, x) - min) / (max - min) * 255;
					
					for (int i = 0; i < blockSize; i++)
						for (int j = 0; j < blockSize; j++)
							image.set_pixel(x * blockSize + j, y * blockSize + i, br, br, br);
				}
			}
			
			if (size.deep == 1) {
				image.save_image(path + "_ch" + std::to_string(d) + ".bmp");
			}
			else {
				image.save_image(path + ".bmp");
			}
		}
	}
}

std::ostream& operator<<(std::ostream& os, const Volume &volume) {
	for (int d = 0; d < volume.size.deep; d++) {
		for (int i = 0; i < volume.size.height; i++) {
			for (int j = 0; j < volume.size.width; j++)
				os << volume.values[i * volume.dw + j * volume.size.deep + d] << " ";
			
			os << std::endl;
		}

		os << std::endl;
	}

	return os;
}
#pragma once

#include <iostream>
#include <vector>
#include <string>

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

	double Min() const; // минимальное значение
	double Max() const; // максимальное значение
	double Mean() const; // среднее значение
	double StdDev() const; // среднеквадратичное отклонение

	VolumeSize GetSize() const; // получение размера
	Volume Resize(int newWidth, int newHeight); // билинейное масштабирование
	Volume ResizeBicubic(int newWidth, int newHeight); // бикубическое масштабирование

	void Save(const std::string &path, int blockSize = 1) const; // сохранение в виде картинки
	void Reshape(int width, int height, int deep); // перераспределение размеров объёма
	void PrintParams() const; // вывод параметров

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

// минимальное значение
double Volume::Min() const {
	double min = values[0];

	for (size_t i = 1; i < values.size(); i++)
		if (values[i] < min)
			min = values[i];

	return min;
}

// максимальное значение
double Volume::Max() const {
	double max = values[0];

	for (size_t i = 1; i < values.size(); i++)
		if (values[i] > max)
			max = values[i];

	return max;
}

// среднее значение
double Volume::Mean() const {
	double sum = 0;

	for (size_t i = 0; i < values.size(); i++)
		sum += values[i];

	return sum / values.size();
}

// среднеквадратичное отклонение
double Volume::StdDev() const {
	double avg = Mean();
	double stddev = 0;

	for (size_t i = 0; i < values.size(); i++)
		stddev += (values[i] - avg) * (values[i] - avg);

	return stddev / values.size();
}

// получение размера
VolumeSize Volume::GetSize() const {
	return size;
}

// билинейное масштабирование
Volume Volume::Resize(int newWidth, int newHeight) {
	Volume result(newWidth, newHeight, size.deep);

	double wscale = (newWidth - 1.0) / (size.width - 1.0);
	double hscale = (newHeight - 1.0) / (size.height - 1.0);

	for (int i = 0; i < newHeight; i++) {
		int y = std::min(int(i / hscale), size.height - 2);
		double dy = (i / hscale) - y;

		for (int j = 0; j < newWidth; j++) {
			int x = std::min(int(j / wscale), size.width - 2);
			double dx = (j / wscale) - x;

			double b0 = (1 - dx) * (1 - dy);
			double b1 = dx * (1 - dy);
			double b2 = (1 - dx) * dy;
			double b3 = dx * dy;

			for (int d = 0; d < size.deep; d++) {
				double sum = 0;

				sum += b0 * At(d, y, x);
				sum += b1 * At(d, y, x + 1);
				sum += b2 * At(d, y + 1, x);
				sum += b3 * At(d, y + 1, x + 1);

				result(d, i, j) = sum;
			}
		}
	}

	return result;
}

// бикубическое масштабирование
Volume Volume::ResizeBicubic(int newWidth, int newHeight) {
	Volume result(newWidth, newHeight, size.deep);

	double wscale = (newWidth - 1.0) / (size.width - 1.0);
	double hscale = (newHeight - 1.0) / (size.height - 1.0);

	for (int i = 0; i < newHeight; i++) {
		int y = std::max(1, std::min(int(i / hscale), size.height - 3));
		double dy = (i / hscale) - y;

		for (int j = 0; j < newWidth; j++) {
			int x = std::max(1, std::min(int(j / wscale), size.width - 3));
			double dx = (j / wscale) - x;

			double b[16];
			b[0] = 1.0 / 4 * (dx - 1) * (dx - 2) * (dx + 1) * (dy - 1) * (dy - 2) * (dy + 1);
			b[1] = -1.0 / 4 * dx * (dx + 1) * (dx - 2) * (dy - 1) * (dy - 2) * (dy + 1);
			b[2] = -1.0 / 4 * dy * (dx - 1) * (dx - 2) * (dx + 1) * (dy + 1) * (dy - 2);
			b[3] = 1.0 / 4 * dx * dy * (dx + 1) * (dx - 2) * (dy + 1) * (dy - 2);
			b[4] = -1.0 / 12 * dx * (dx - 1) * (dx - 2) * (dy - 1) * (dy - 2) * (dy + 1);
			b[5] = -1.0 / 12 * dy * (dx - 1) * (dx - 2) * (dx + 1) * (dy - 1) * (dy - 2);
			b[6] = 1.0 / 12 * dx * dy * (dx - 1) * (dx - 2) * (dy + 1) * (dy - 2);
			b[7] = 1.0 / 12 * dx * dy * (dx + 1) * (dx - 2) * (dy - 1) * (dy - 2);
			b[8] = 1.0 / 12 * dx * (dx - 1) * (dx + 1) * (dy - 1) * (dy - 2) * (dy + 1);
			b[9] = 1.0 / 12 * dy * (dx - 1) * (dx - 2) * (dx + 1) * (dy - 1) * (dy + 1);
			b[10] = 1.0 / 36 * dx * dy * (dx - 1) * (dx - 2) * (dy - 1) * (dy - 2);
			b[11] = -1.0 / 12 * dx * dy * (dx - 1) * (dx + 1) * (dy + 1) * (dy - 2);
			b[12] = -1.0 / 12 * dx * dy * (dx + 1) * (dx - 2) * (dy - 1) * (dy + 1);
			b[13] = -1.0 / 36 * dx * dy * (dx - 1) * (dx + 1) * (dy - 1) * (dy - 2);
			b[14] = -1.0 / 36 * dx * dy * (dx - 1) * (dx - 2) * (dy - 1) * (dy + 1);
			b[15] = 1.0 / 36 * dx * dy * (dx - 1) * (dx + 1) * (dy - 1) * (dy + 1);

			for (int d = 0; d < size.deep; d++) {
				double sum = 0;

				sum += b[0] * At(d, y, x);
				sum += b[1] * At(d, y, x + 1);
				sum += b[2] * At(d, y + 1, x);
				sum += b[3] * At(d, y + 1, x + 1);
				sum += b[4] * At(d, y, x - 1);
				sum += b[5] * At(d, y - 1, x);
				sum += b[6] * At(d, y + 1, x - 1);
				sum += b[7] * At(d, y - 1, x + 1);
				sum += b[8] * At(d, y, x + 2);
				sum += b[9] * At(d, y + 2, x);
				sum += b[10] * At(d, y - 1, x - 1);
				sum += b[11] * At(d, y + 1, x + 2);
				sum += b[12] * At(d, y + 2, x + 1);
				sum += b[13] * At(d, y - 1, x + 2);
				sum += b[14] * At(d, y + 2, x - 1);
				sum += b[15] * At(d, y + 2, x + 2);

				result(d, i, j) = sum;
			}
		}
	}

	return result;
}

// сохранение в виде картинки
void Volume::Save(const std::string &path, int blockSize) const {
	int width = size.width * blockSize;
	int height = size.height * blockSize;

	if (size.width == 1 && size.height == 1) {
		BitmapImage image(blockSize, size.deep * blockSize);

		for (int d = 0; d < size.deep; d++) {
			int value = std::min(255.0, std::max(0.0, values[d]));

			for (int i = 0; i < blockSize; i++)
				for (int j = 0; j < blockSize; j++)
					image.set_pixel(j, d * blockSize + i, 0, value, 0);
		}
		
		image.save_image(path + ".bmp");
	}
	else if (size.deep == 3) {
		BitmapImage image(width, height);

		for (int y = 0; y < size.height; y++) {
			for (int x = 0; x < size.width; x++) {
				int r = std::min(255.0, std::max(0.0, At(0, y, x)));
				int g = std::min(255.0, std::max(0.0, At(1, y, x)));
				int b = std::min(255.0, std::max(0.0, At(2, y, x)));

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
					int br = std::min(255.0, std::max(0.0, At(d, y, x)));
					
					for (int i = 0; i < blockSize; i++)
						for (int j = 0; j < blockSize; j++)
							image.set_pixel(x * blockSize + j, y * blockSize + i, br, br, br);
				}
			}
			
			if (size.deep == 1) {
				image.save_image(path + ".bmp");
			}
			else {
				image.save_image(path + "_ch" + std::to_string(d) + ".bmp");
			}
		}
	}
}

// перераспределение размеров объёма
void Volume::Reshape(int width, int height, int deep) {
	if (width * height * deep != size.width * size.height * size.deep)
		throw std::runtime_error("Unable to reshape");

	size.width = width;
	size.height = height;
	size.deep = deep;

	whd = width * height * deep;
	dh = deep * height;
	dw = deep * width;
}

// вывод параметров
void Volume::PrintParams() const {
	std::cout << "Volume " << size << std::endl;
	std::cout << "Min: " << Min() << std::endl;
	std::cout << "Max: " << Max() << std::endl;
	std::cout << "Mean: " << Mean() << std::endl;
	std::cout << "Std.dev: " << StdDev() << std::endl;
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
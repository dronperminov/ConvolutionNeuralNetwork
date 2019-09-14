#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class UpscaleLayer : public NetworkLayer {
	int scale;

public:
	UpscaleLayer(VolumeSize size, int scale = 2);

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

UpscaleLayer::UpscaleLayer(VolumeSize size, int scale) : NetworkLayer(size, size.width * scale, size.height * scale, size.deep) {
	name = "upscale";
	info = "scale: " + std::to_string(scale);

	this->scale = scale;
}

// прямое распространение
void UpscaleLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(4)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int d = 0; d < inputSize.deep; d++) {
			for (int i = 0; i < inputSize.height; i++) {
				for (int j = 0; j < inputSize.width; j++) {
					double value = X[batchIndex](d, i, j);

					for (int y = 0; y < scale; y++)
						for (int x = 0; x < scale; x++)
							output[batchIndex](d, i * scale + y, j * scale + x) = value;		
				}
			}
		}
	}
}

// обратное распространение
void UpscaleLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(4)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int d = 0; d < inputSize.deep; d++) {
			for (int i = 0; i < inputSize.height; i++) {
				for (int j = 0; j < inputSize.width; j++) {
					double delta = 0;

					for (int y = 0; y < scale; y++)
						for (int x = 0; x < scale; x++)
							delta += dout[batchIndex](d, i * scale + y, j * scale + x);

					dX[batchIndex](d, i, j) = delta;
				}
			}
		}
	}
}

// сохранение слоя в файл
void UpscaleLayer::Save(std::ofstream &f) const {
	f << "upscale " << inputSize << " " << scale << std::endl;
}
#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

#include "NetworkLayer.hpp"

class UpscaleBilinearLayer : public NetworkLayer {
	int scale;

public:
	UpscaleBilinearLayer(VolumeSize size, int scale = 2);

	void Forward(const std::vector<Volume> &X); // прямое распространение
	void Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX); // обратное распространение

	void Save(std::ofstream &f) const; // сохранение слоя в файл
};

UpscaleBilinearLayer::UpscaleBilinearLayer(VolumeSize size, int scale) : NetworkLayer(size, size.width * scale, size.height * scale, size.deep) {
	name = "upscale bilinear";
	info = "scale: " + std::to_string(scale);

	this->scale = scale;
}

// прямое распространение
void UpscaleBilinearLayer::Forward(const std::vector<Volume> &X) {
	#pragma omp parallel for collapse(4)
	for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++) {
		for (int d = 0; d < inputSize.deep; d++) {
			for (int i = 0; i < outputSize.height; i++) {
				for (int j = 0; j < outputSize.width; j++) {
					int y = std::min(i / scale, inputSize.height - 2);
					int x = std::min(j / scale, inputSize.width - 2);

					double dy = (i / (double)scale) - y;
					double dx = (j / (double)scale) - x;

					double p1 = X[batchIndex](d, y, x) * (1 - dx) * (1 - dy);
					double p2 = X[batchIndex](d, y, x + 1) * dx * (1 - dy);
					double p3 = X[batchIndex](d, y + 1, x) * (1 - dx) * dy;
					double p4 = X[batchIndex](d, y + 1, x + 1) * dx * dy;

					output[batchIndex](d, i, j) = p1 + p2 + p3 + p4;		
				}
			}
		}
	}
}

// обратное распространение
void UpscaleBilinearLayer::Backward(const std::vector<Volume> &dout, const std::vector<Volume> &X, bool calc_dX) {
	if (!calc_dX)
		return;

	#pragma omp parallel for collapse(2)
	for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++) {
		for (int d = 0; d < inputSize.deep; d++) {
			for (int i = 0; i < inputSize.height; i++)
				for (int j = 0; j < inputSize.width; j++) 
					dX[batchIndex](d, i, j) = 0;

			for (int i = 0; i < outputSize.height; i++) {
				for (int j = 0; j < outputSize.width; j++) {
					int y = std::min(i / scale, inputSize.height - 2);
					int x = std::min(j / scale, inputSize.width - 2);

					double dy = (i / (double)scale) - y;
					double dx = (j / (double)scale) - x;

					double delta = dout[batchIndex](d, i, j);

					dX[batchIndex](d, y, x) += delta * (1 - dx) * (1 - dy);		
					dX[batchIndex](d, y, x + 1) += delta * dx * (1 - dy);		
					dX[batchIndex](d, y + 1, x) += delta * (1 - dx) * dy;		
					dX[batchIndex](d, y + 1, x + 1) += delta * dx * dy;		
				}
			}
		}
	}
}

// сохранение слоя в файл
void UpscaleBilinearLayer::Save(std::ofstream &f) const {
	f << "upscalebilinear " << inputSize << " " << scale << std::endl;
}
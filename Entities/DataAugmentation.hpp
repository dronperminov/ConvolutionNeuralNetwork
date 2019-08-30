#pragma once

#include "Volume.hpp"

class DataAugmentation {
	double verticalShift;
	double horizontalShift;
	double fillValue;

	double brightnessMin;
	double brightnessMax;

public:
	DataAugmentation(const std::string &config);
	Volume Make(const Volume &volume);
};

DataAugmentation::DataAugmentation(const std::string &config) {
	ArgParser parser(config);

	verticalShift = stod(parser.Get("shift-y", "0"));
	horizontalShift = stod(parser.Get("shift-x", "0"));
	fillValue = stod(parser.Get("fill"));

	brightnessMin = stod(parser.Get("br-min", "1"));
	brightnessMax = stod(parser.Get("br-max", "1"));

	if (verticalShift < 0 || verticalShift > 1)
		throw std::runtime_error("Vertical shift must in [0, 1]");

	if (horizontalShift < 0 || horizontalShift > 1)
		throw std::runtime_error("Horizontal shift must in [0, 1]");

	if (brightnessMin < 0 || brightnessMax < 0 || brightnessMax > 2 || brightnessMin > brightnessMax)
		throw std::runtime_error("Invalid brightness values");
}

Volume DataAugmentation::Make(const Volume &volume) {
	VolumeSize size = volume.GetSize();
	Volume result(size); // создаём результирующий объём

	int shiftVert = volume.Width() * verticalShift * (-1 + rand() * 2.0 / RAND_MAX);
	int shiftHori = volume.Height() * horizontalShift * (-1 + rand() * 2.0 / RAND_MAX);
	
	double brightnessScale = brightnessMin + rand() * (brightnessMax - brightnessMin) / RAND_MAX;

	#pragma omp parallel for
	for (int d = 0; d < size.deep; d++) {
		for (int i = 0; i < size.height; i++) {
			for (int j = 0; j < size.width; j++) {
				int i1 = i + shiftVert;
				int j1 = j + shiftHori;

				if (i1 >= 0 && j1 >= 0 && i1 < size.height && j1 < size.width)
					result(d, i, j) = volume(d, i1, j1) * brightnessScale;
				else if (fillValue != -1)
					result(d, i, j) = fillValue;
				else
					result(d, i, j) = volume(d, i, j) * brightnessScale;
			}
		}
	}

	return result;
}
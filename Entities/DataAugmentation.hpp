#pragma once

#include "Volume.hpp"

class DataAugmentation {
	double verticalShift;
	double horizontalShift;
	double fillValue;

	double brightnessMin;
	double brightnessMax;

	double rotation;

public:
	DataAugmentation(const std::string &config);
	Volume Make(const Volume &volume);
};

DataAugmentation::DataAugmentation(const std::string &config) {
	ArgParser parser(config);

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "shift-y") {
			verticalShift = stod(parser.Get("shift-y", "0"));
		}
		else if (arg == "shift-x") {
			horizontalShift = stod(parser.Get("shift-x", "0"));
		}
		else if (arg == "fill") {
			fillValue = stod(parser.Get("fill", "0"));
		}
		else if (arg == "br-min") {
			brightnessMin = stod(parser.Get("br-min", "1"));
		}
		else if (arg == "br-max") {
			brightnessMax = stod(parser.Get("br-max", "1"));
		}
		else if (arg == "rotate") {
			rotation = stod(parser.Get("rotate", "0")) / 180 * M_PI;
		}
		else
			throw std::runtime_error("Invalid data augmentation argument '" + arg + "'");
	}

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

	int shiftVert = size.width * verticalShift * (-1 + rand() * 2.0 / RAND_MAX);
	int shiftHori = size.height * horizontalShift * (-1 + rand() * 2.0 / RAND_MAX);
	
	double brightnessScale = brightnessMin + rand() * (brightnessMax - brightnessMin) / RAND_MAX;
	double deg = rotation * (-1 + rand() * 2.0 / RAND_MAX);

	double c = cos(deg);
	double s = sin(deg);

	double w = size.width / 2.0;
	double h = size.height / 2.0;

	#pragma omp parallel for collapse(3)
	for (int d = 0; d < size.deep; d++) {
		for (int i = 0; i < size.height; i++) {
			for (int j = 0; j < size.width; j++) {
				int x = j + shiftHori - w;
				int y = i + shiftVert - h;

				int i1 = x * s + y * c + h;
				int j1 = x * c - y * s + w;

				if (i1 >= 0 && j1 >= 0 && i1 < size.height && j1 < size.width) {
					result(d, i, j) = volume(d, i1, j1) * brightnessScale;
				}
				else {
					result(d, i, j) = fillValue;
				}
			}
		}
	}

	return result;
}
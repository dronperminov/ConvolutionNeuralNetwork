#pragma once

#include "Volume.hpp"

class DataAugmentation {
	double verticalShift;
	double horizontalShift;
	double fillValue;

	double brightnessMin;
	double brightnessMax;

	double rotation;

	bool horizontalFlip;
	bool verticalFlip;

public:
	DataAugmentation(const std::string &config);
	Volume Make(const Volume &volume);
};

DataAugmentation::DataAugmentation(const std::string &config) {
	ArgParser parser(config);

	verticalShift = 0;
	horizontalShift = 0;
	fillValue = 0;

	brightnessMin = 1;
	brightnessMax = 1;

	rotation = 0;

	verticalFlip = false;
	horizontalFlip = false;

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "shift-y" || arg == "shift-vert") {
			verticalShift = stod(parser.Get(arg));
		}
		else if (arg == "shift-x" || arg == "shift-hori") {
			horizontalShift = stod(parser.Get(arg));
		}
		else if (arg == "fill") {
			fillValue = stod(parser.Get(arg));
		}
		else if (arg == "br-min" || arg == "min-br") {
			brightnessMin = stod(parser.Get(arg));
		}
		else if (arg == "br-max" || arg == "max-br") {
			brightnessMax = stod(parser.Get(arg));
		}
		else if (arg == "rotate" || arg == "rotation") {
			rotation = stod(parser.Get(arg)) / 180 * M_PI;
		}
		else if (arg == "flip-y" || arg == "flip-vert" || arg == "vert-flip" || arg == "vertical-flip") {
			std::string value = parser.Get(arg);

			if (value == "" || value == "true") {
				verticalFlip = true;
			}
			else if (value == "false") {
				verticalFlip = false;
			}
			else
				throw std::runtime_error("Invalid value for vertical flip");
		}
		else if (arg == "flip-x" || arg == "flip-hori" || arg == "hori-flip" || arg == "horizontal-flip") {
			std::string value = parser.Get(arg);

			if (value == "" || value == "true") {
				horizontalFlip = true;
			}
			else if (value == "false") {
				horizontalFlip = false;
			}
			else
				throw std::runtime_error("Invalid value for horizontal flip");
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

	bool flipVert = verticalFlip && rand() % 2;
	bool flipHori = horizontalFlip && rand() % 2;

	#pragma omp parallel for collapse(3)
	for (int d = 0; d < size.deep; d++) {
		for (int i = 0; i < size.height; i++) {
			for (int j = 0; j < size.width; j++) {
				int x = j + shiftHori - w;
				int y = i + shiftVert - h;

				int i1 = x * s + y * c + h;
				int j1 = x * c - y * s + w;

				y = flipVert ? size.height - 1 - i : i;
				x = flipHori ? size.width - 1 - j : j;

				if (i1 >= 0 && j1 >= 0 && i1 < size.height && j1 < size.width) {
					result(d, y, x) = volume(d, i1, j1) * brightnessScale;
				}
				else {
					result(d, y, x) = fillValue;
				}
			}
		}
	}

	return result;
}
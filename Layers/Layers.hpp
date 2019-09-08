#pragma once

#include "NetworkLayer.hpp"

#include "ConvLayer.hpp"
#include "ConvWithoutStrideLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "AveragePoolingLayer.hpp"
#include "FullyConnectedLayer.hpp"

#include "ResidualLayer.hpp"
#include "InceptionLayer.hpp"
#include "IdentityLayer.hpp"

#include "DropoutLayer.hpp"
#include "GaussDropoutLayer.hpp"
#include "GaussNoiseLayer.hpp"
#include "BatchNormalizationLayer.hpp"
#include "BatchNormalization2DLayer.hpp"

#include "Activations/SigmoidLayer.hpp"
#include "Activations/LogSigmoidLayer.hpp"
#include "Activations/TanhLayer.hpp"
#include "Activations/ReLULayer.hpp"
#include "Activations/ELULayer.hpp"
#include "Activations/ParametricReLULayer.hpp"
#include "Activations/SwishLayer.hpp"
#include "Activations/SoftsignLayer.hpp"
#include "Activations/SoftplusLayer.hpp"
#include "Activations/SoftmaxLayer.hpp"

#include "NetworkBlock.hpp"

#include "../Entities/ArgParser.hpp"

// парсинг свёрточного слоя
NetworkLayer* ParseConvLayer(VolumeSize size, ArgParser &parser) {
	std::string fs = "";
	std::string fc = "";
	std::string S = "1";
	std::string P = "0";

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "filter_size" || arg == "fs") {
			fs = parser.Get(arg);
		}
		else if (arg == "filters" || arg == "fc") {
			fc = parser.Get(arg);
		}
		else if (arg == "S") {
			S = parser.Get(arg);
		}
		else if (arg == "padding" || arg == "P") {
			P = parser.Get(arg);
		}
		else if (arg != "conv" && arg != "convolution")
			throw std::runtime_error("Invalid conv argument '" + arg + "'");
	}

	if (fc == "")
		throw std::runtime_error("Unable to add conv layer. Filters count is not set");

	if (fs == "")
		throw std::runtime_error("Unable to add conv layer. Filters size is not set");

	int pad;

	if (P == "same") {
		if (S != "1")
			throw std::runtime_error("Unable to make same padding for stride > 1");

		pad = (std::stoi(fs) - 1) / 2;
	}
	else if (P == "valid") {
		pad = 0;
	}
	else if (P == "full") {
		pad = std::stoi(fs) - 1;
	}
	else {
		pad = std::stoi(P);
	}

	if (S == "1")
		return new ConvWithoutStrideLayer(size, std::stoi(fc), std::stoi(fs), pad);

	return new ConvLayer(size, std::stoi(fc), std::stoi(fs), pad, std::stoi(S));
}

// парсинг слоя пулинга
NetworkLayer* ParsePoolingLayers(VolumeSize size, ArgParser &parser) {
	std::string scale = "2";

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "scale") {
			scale = parser.Get(arg);
		}
		else if (arg != "maxpool" && arg != "pooling" && arg != "maxpooling" && arg != "avgpool" && arg != "averagepooling" && arg != "avgpooling")
			throw std::runtime_error("Invalid pooling argument '" + arg + "'");
	}

	if (parser["maxpool"] || parser["pooling"] || parser["maxpooling"])
		return new MaxPoolingLayer(size, std::stoi(scale));

	if (parser["avgpool"] || parser["averagepooling"] || parser["avgpooling"])
		return new AveragePoolingLayer(size, std::stoi(scale));

	return nullptr;
}

// парсинг полносвязного слоя
NetworkLayer* ParseFullyConnectedLayer(VolumeSize size, ArgParser &parser) {
	std::string outputs = "";
	std::string type = "none";

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "outputs" || arg == "size") {
			outputs = parser.Get(arg);
		}
		else if (arg == "activation") {
			type = parser.Get(arg);
		}
		else if (arg != "fc" && arg != "fullconnected")
			throw std::runtime_error("Invalid fullyconnected argument '" + arg + "'");
	}

	if (outputs == "")
		throw std::runtime_error("Unable to add full connected layer. Outputs is not set");

	return new FullyConnectedLayer(size, std::stoi(outputs), type);
}

// парсинг слоёв дропаута
NetworkLayer* ParseDropoutLayers(VolumeSize size, ArgParser &parser) {
	std::string p = "0.5";

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "p") {
			p = parser.Get(arg);
		}
		else if (arg != "dropout" && arg != "gaussdropout")
			throw std::runtime_error("Invalid dropout argument '" + arg + "'");
	}

	if (parser["dropout"])
		return new DropoutLayer(size, std::stod(p));

	if (parser["gaussdropout"])
		return new GaussDropoutLayer(size, std::stod(p));

	return nullptr;
}

// парсинг слоёв нормализации
NetworkLayer* ParseNormalizationLayers(VolumeSize size, ArgParser &parser) {
	std::string momentum = "0.9";

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "momentum" || arg == "moment" || arg == "mu") {
			momentum = parser.Get(arg);
		}
		else if (arg != "batchnormalization" && arg != "batchnormalization2D")
			throw std::runtime_error("Invalid normalization argument '" + arg + "'");
	}

	if (parser["batchnormalization"])
		return new BatchNormalizationLayer(size, std::stod(momentum));

	if (parser["batchnormalization2D"])
		return new BatchNormalization2DLayer(size, std::stod(momentum));

	return nullptr;
}

// парсинг слоя шума
NetworkLayer* ParseNoiseLayer(VolumeSize size, ArgParser &parser) {
	std::string stddev = "0.5";

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "stddev" || arg == "dev") {
			stddev = parser.Get(arg);
		}
		else if (arg != "gaussnoise")
			throw std::runtime_error("Invalid gauss noise argument '" + arg + "'");
	}

	return new GaussNoiseLayer(size, std::stod(stddev));
}

// парсинг residual слоя
NetworkLayer* ParseResidualLayer(VolumeSize size, ArgParser &parser) {
	std::string features = "";

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "features" || arg == "k") {
			features = parser.Get(arg);
		}
		else if (arg != "res" && arg != "residual")
			throw std::runtime_error("Invalid residual argument '" + arg + "'");
	}

	if (features == "")
		throw std::runtime_error("Unable to add residual layer. Features are not set");

	return new ResidualLayer(size, std::stod(features));
}

// парсинг inception слоя
NetworkLayer* ParseInceptionLayer(VolumeSize size, ArgParser &parser) {
	std::string fc1 = "";
	std::string fc3 = "";
	std::string fc5 = "";

	for (size_t i = 0; i < parser.size(); i++) {
		std::string arg = parser[i];

		if (arg == "fc1") {
			fc1 = parser.Get(arg);
		}
		else if (arg == "fc3") {
			fc3 = parser.Get(arg);
		}
		else if (arg == "fc5") {
			fc5 = parser.Get(arg);
		}
		else if (arg != "inception")
			throw std::runtime_error("Invalid inception argument '" + arg + "'");
	}

	if (fc1 == "" || fc3 == "" || fc5 == "")
		throw std::runtime_error("Unable to add inception layer. Features are not set");

	return new InceptionLayer(size, std::stod(fc1), std::stod(fc3), std::stod(fc5));
}

// создание слоя по описанию
NetworkLayer* CreateLayer(VolumeSize size, const std::string &layerConf) {
	NetworkLayer *layer = nullptr;
	ArgParser parser(layerConf);

	if (parser["conv"] || parser["convolution"]) {
		layer = ParseConvLayer(size, parser);
	}
	else if (parser["maxpool"] || parser["pooling"] || parser["maxpooling"] || parser["avgpool"] || parser["averagepooling"] || parser["avgpooling"]) {
		layer = ParsePoolingLayers(size, parser);
	}
	else if (parser["fc"] || parser["fullconnected"]) {
		layer = ParseFullyConnectedLayer(size, parser);
	}
	else if (parser["residual"] || parser["res"]) {
		layer = ParseResidualLayer(size, parser);
	}
	else if (parser["inception"]) {
		layer = ParseInceptionLayer(size, parser);
	}
	else if (parser["identity"]) {
		layer = new IdentityLayer(size);
	}
	else if (parser["dropout"] || parser["gaussdropout"]) {
		layer = ParseDropoutLayers(size, parser);
	}
	else if (parser["batchnormalization"] || parser["batchnormalization2D"]) {
		layer = ParseNormalizationLayers(size, parser);
	}
	else if (parser["gaussnoise"]) {
		layer = ParseNoiseLayer(size, parser);
	}
	else if (parser["softmax"]) {
		layer = new SoftmaxLayer(size);
	}
	else if (parser["softsign"]) {
		layer = new SoftsignLayer(size);
	}
	else if (parser["softplus"]) {
		layer = new SoftplusLayer(size);
	}
	else if (parser["sigmoid"]) {
		layer = new SigmoidLayer(size);
	}
	else if (parser["logsigmoid"]) {
		layer = new LogSigmoidLayer(size);
	}
	else if (parser["tanh"]) {
		layer = new TanhLayer(size);
	}
	else if (parser["relu"]) {
		layer = new ReLULayer(size);
	}
	else if (parser["elu"]) {
		std::string alpha = parser.Get("alpha", "1");

		layer = new ELULayer(size, std::stod(alpha));
	}
	else if (parser["prelu"] || parser["parametricrelu"]) {
		layer = new ParametricReLULayer(size);
	}
	else if (parser["swish"]) {
		layer = new SwishLayer(size);
	}
	else {
		throw std::runtime_error("Invalid layer name '" + layerConf + "'");
	}

	return layer;
}

// загрузка слоя из файла
NetworkLayer* LoadLayer(VolumeSize size, const std::string &layerType, std::ifstream &f) {
	NetworkLayer *layer = nullptr;

	if (layerType == "conv" || layerType == "convolution") {
		int fc, fs, P, S;
		f >> fs >> fc >> P >> S;
		
		if (S == 1)
			layer = new ConvWithoutStrideLayer(size, fc, fs, P, f);
		else
			layer = new ConvLayer(size, fc, fs, P, S, f);
	}
	else if (layerType == "maxpool" || layerType == "maxpooling") {
		int scale;
		f >> scale;

		layer = new MaxPoolingLayer(size, scale);
	}
	else if (layerType == "avgpool" || layerType == "avgpooling") {
		int scale;
		f >> scale;

		layer = new AveragePoolingLayer(size, scale);
	}
	else if (layerType == "fc" || layerType == "fullconnected") {
		int outputs;
		std::string type;
		f >> outputs >> type;

		layer = new FullyConnectedLayer(size, outputs, type, f);
	}
	else if (layerType == "residual" || layerType == "res") {
		int features;
		f >> features;

		layer = new ResidualLayer(size, features, f);
	}
	else if (layerType == "inception") {
		int fc1, fc3, fc5;
		f >> fc1 >> fc3 >> fc5;

		layer = new InceptionLayer(size, fc1, fc3, fc5, f);
	}
	else if (layerType == "dropout") {
		double p;
		f >> p;

		layer = new DropoutLayer(size, p);
	}
	else if (layerType == "gaussdropout") {
		double p;
		f >> p;

		layer = new GaussDropoutLayer(size, p);
	}
	else if (layerType == "gaussnoise") {
		double stddev;
		f >> stddev;

		layer = new GaussNoiseLayer(size, stddev);
	}
	else if (layerType == "batchnormalization") {
		double momentum;
		f >> momentum;

		layer = new BatchNormalizationLayer(size, momentum, f);
	}
	else if (layerType == "batchnormalization2D") {
		double momentum;
		f >> momentum;

		layer = new BatchNormalization2DLayer(size, momentum, f);
	}
	else if (layerType == "block") {
		std::string type;
		f >> type;

		layer = new NetworkBlock(size, type, f);
	}
	else if (layerType == "identity") {
		layer = new IdentityLayer(size);
	}
	else if (layerType == "sigmoid") {
		layer = new SigmoidLayer(size);
	}
	else if (layerType == "logsigmoid") {
		layer = new LogSigmoidLayer(size);
	}
	else if (layerType == "tanh") {
		layer = new TanhLayer(size);
	}
	else if (layerType == "relu") {
		layer = new ReLULayer(size);
	}
	else if (layerType == "elu") {
		double alpha;
		f >> alpha;

		layer = new ELULayer(size, alpha);
	}
	else if (layerType == "prelu") {
		layer = new ParametricReLULayer(size, f);
	}
	else if (layerType == "swish") {
		layer = new SwishLayer(size);
	}
	else if (layerType == "softsign") {
		layer = new SoftsignLayer(size);
	}
	else if (layerType == "softplus") {
		layer = new SoftplusLayer(size);
	}
	else if (layerType == "softmax") {
		layer = new SoftmaxLayer(size);
	}
	else
		throw std::runtime_error("Invalid layer type '" + layerType + "'");

	return layer;
}
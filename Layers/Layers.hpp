#pragma once

#include "NetworkLayer.hpp"

#include "ConvLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "AveragePoolingLayer.hpp"
#include "FullyConnectedLayer.hpp"

#include "ResidualLayer.hpp"
#include "InceptionLayer.hpp"

#include "DropoutLayer.hpp"
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

#include "../Entities/ArgParser.hpp"

// создание слоя по описанию
NetworkLayer* CreateLayer(VolumeSize size, const std::string &layerConf) {
	NetworkLayer *layer = nullptr;
	ArgParser parser(layerConf);

	if (parser["conv"] || parser["convolution"]) {
		if (!parser["filters"])
			throw std::runtime_error("Unable to add conv layer. Filters count is not set");

		std::string fs = parser.Get("filter_size", "3");
		std::string fc = parser.Get("filters");

		std::string S = parser.Get("S", "1");
		std::string P = parser.Get("P", "0");

		layer = new ConvLayer(size, std::stoi(fc), std::stoi(fs), std::stoi(P), std::stoi(S));
	}
	else if (parser["maxpool"] || parser["pooling"] || parser["maxpooling"]) {
		std::string scale = parser.Get("scale", "2");

		layer = new MaxPoolingLayer(size, std::stoi(scale));
	}
	else if (parser["avgpool"] || parser["averagepooling"] || parser["avgpooling"]) {
		std::string scale = parser.Get("scale", "2");

		layer = new AveragePoolingLayer(size, std::stoi(scale));
	}
	else if (parser["fc"] || parser["fullconnected"]) {
		if (!parser["outputs"])
			throw std::runtime_error("Unable to add full connected layer. Outputs is not set");

		std::string outputs = parser.Get("outputs");
		std::string type = parser.Get("activation", "none");

		layer = new FullyConnectedLayer(size, std::stoi(outputs), type);
	}
	else if (parser["residual"] || parser["res"]) {
		if (!parser["features"])
			throw std::runtime_error("Unable to add residual layer. Features are not set");

		std::string features = parser.Get("features");

		layer = new ResidualLayer(size, std::stod(features));
	}
	else if (parser["inception"]) {
		if (!parser["fc1"] || !parser["fc3"] || !parser["fc5"])
			throw std::runtime_error("Unable to add inception layer. Features are not set");

		std::string fc1 = parser.Get("fc1");
		std::string fc3 = parser.Get("fc3");
		std::string fc5 = parser.Get("fc5");

		layer = new InceptionLayer(size, std::stod(fc1), std::stod(fc3), std::stod(fc5));
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
	else if (parser["dropout"]) {
		std::string p = parser.Get("p", "0.5");

		layer = new DropoutLayer(size, std::stod(p));
	}
	else if (parser["batchnormalization"]) {
		std::string momentum = parser.Get("momentum", "0.9");

		layer = new BatchNormalizationLayer(size, std::stod(momentum));
	}
	else if (parser["batchnormalization2D"]) {
		std::string momentum = parser.Get("momentum", "0.9");

		layer = new BatchNormalization2DLayer(size, std::stod(momentum));
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
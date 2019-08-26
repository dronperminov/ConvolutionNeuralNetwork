#include <iostream>
#include <cassert>

#include "Layers/ConvLayer.hpp"
#include "Layers/MaxPoolingLayer.hpp"
#include "Layers/AveragePoolingLayer.hpp"
#include "Layers/FullyConnectedLayer.hpp"
#include "Layers/DropoutLayer.hpp"
#include "Layers/BatchNormalizationLayer.hpp"
#include "Network.hpp"

using namespace std;

void FullyConnectedLayerTest() {
	cout << "Full connected tests: ";
	VolumeSize size;
	size.height = 1;
	size.width = 1;
	size.deep = 8;

	FullyConnectedLayer layer(size, 4, "relu");
	layer.SetBatchSize(1);

	double weights[4][8] = {
		{ 1, 2, 3, 4, -1, -2, -3, -4 },
		{ 1, 0, 0, 1, -1, 2, -3, 4 }, 
		{ 1, 1, 1, 1, -3, -4, -5, -6 }, 
		{ 1, 2, 2, 1, 2, 2, -3, -8 }
	};

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 8; j++)
			layer.SetWeight(i, j, weights[i][j]);

		layer.SetBias(i, 0);
	}

	Volume input(1, 1, 8);

	input(0, 0, 0) = 1;
	input(1, 0, 0) = 2;
	input(2, 0, 0) = -3;
	input(3, 0, 0) = 4;
	input(4, 0, 0) = 0;
	input(5, 0, 0) = -7;
	input(6, 0, 0) = 2;
	input(7, 0, 0) = -4;

	layer.Forward({input});

	Volume& output = layer.GetOutput()[0];

	assert(output.Width() == 1);
	assert(output.Height() == 1);
	assert(output.Deep() == 4);

	assert(output(0, 0, 0) == 36);
	assert(output(1, 0, 0) == 0);
	assert(output(2, 0, 0) == 46);
	assert(output(3, 0, 0) == 15);

	Volume deltas(1, 1, 4);

	deltas(0, 0, 0) = -0.5;
	deltas(1, 0, 0) = 0.1;
	deltas(2, 0, 0) = -0.25;
	deltas(3, 0, 0) = 0.7;

	layer.Backward({deltas}, {input}, true);
	Volume &prev = layer.GetDeltas()[0];

	assert(fabs(prev(0, 0, 0) + 0.05) < 1e-15);
	assert(fabs(prev(1, 0, 0) - 0.15) < 1e-15);
	assert(fabs(prev(2, 0, 0) + 0.35) < 1e-15);
	assert(fabs(prev(3, 0, 0) + 1.55) < 1e-15);
	assert(fabs(prev(4, 0, 0) - 2.65) < 1e-15);
	assert(fabs(prev(5, 0, 0) - 3.4) < 1e-15);
	assert(fabs(prev(6, 0, 0) - 0.65) < 1e-15);
	assert(fabs(prev(7, 0, 0) + 2.1) < 1e-15);

	std::cout << "OK" << std::endl;
}

void MaxPoolingLayerTest() {
	cout << "Max pooling tests: ";

	VolumeSize size;
	size.height = 4;
	size.width = 4;
	size.deep = 1;

	MaxPoolingLayer layer(size, 2);
	layer.SetBatchSize(1);
	Volume input(4, 4, 1);

	input(0, 0, 0) = 1;
	input(0, 0, 1) = 9;
	input(0, 0, 2) = 8;
	input(0, 0, 3) = 4;

	input(0, 1, 0) = 4;
	input(0, 1, 1) = 8;
	input(0, 1, 2) = 6;
	input(0, 1, 3) = 7;

	input(0, 2, 0) = 4;
	input(0, 2, 1) = 0;
	input(0, 2, 2) = 5;
	input(0, 2, 3) = 9;

	input(0, 3, 0) = 7;
	input(0, 3, 1) = 3;
	input(0, 3, 2) = 5;
	input(0, 3, 3) = 4;


	layer.Forward({input});
	Volume output = layer.GetOutput()[0];

	assert(output.Width() == 2);
	assert(output.Height() == 2);
	assert(output.Deep() == 1);

	assert(output(0, 0, 0) == 9);
	assert(output(0, 0, 1) == 8);
	assert(output(0, 1, 0) == 7);
	assert(output(0, 1, 1) == 9);

	Volume deltas(2, 2, 1);

	deltas(0, 0, 0) = 1.2;
	deltas(0, 0, 1) = 1.9;
	deltas(0, 1, 0) = 0.9;
	deltas(0, 1, 1) = 0.3;


	layer.Backward({deltas}, {input}, true);

	Volume& deltas2 = layer.GetDeltas()[0];

	assert(deltas2(0, 0, 0) == 0);
	assert(deltas2(0, 0, 1) == 1.2);
	assert(deltas2(0, 0, 2) == 1.9);
	assert(deltas2(0, 0, 3) == 0);

	assert(deltas2(0, 1, 0) == 0);
	assert(deltas2(0, 1, 1) == 0);
	assert(deltas2(0, 1, 2) == 0);
	assert(deltas2(0, 1, 3) == 0);

	assert(deltas2(0, 2, 0) == 0);
	assert(deltas2(0, 2, 1) == 0);
	assert(deltas2(0, 2, 2) == 0);
	assert(deltas2(0, 2, 3) == 0.3);

	assert(deltas2(0, 3, 0) == 0.9);
	assert(deltas2(0, 3, 1) == 0);
	assert(deltas2(0, 3, 2) == 0);
	assert(deltas2(0, 3, 3) == 0);

	cout << "OK" << endl;
}

void AveragePoolingLayerTest() {
	cout << "Average pooling tests: ";

	VolumeSize size;
	size.height = 4;
	size.width = 4;
	size.deep = 1;

	AveragePoolingLayer layer(size, 2);
	layer.SetBatchSize(1);
	Volume input(4, 4, 1);

	input(0, 0, 0) = 1;
	input(0, 0, 1) = 9;
	input(0, 0, 2) = 8;
	input(0, 0, 3) = 4;

	input(0, 1, 0) = 4;
	input(0, 1, 1) = 8;
	input(0, 1, 2) = 6;
	input(0, 1, 3) = 7;

	input(0, 2, 0) = 4;
	input(0, 2, 1) = 0;
	input(0, 2, 2) = 5;
	input(0, 2, 3) = 9;

	input(0, 3, 0) = 7;
	input(0, 3, 1) = 3;
	input(0, 3, 2) = 5;
	input(0, 3, 3) = 4;


	layer.Forward({input});
	Volume output = layer.GetOutput()[0];

	assert(output.Width() == 2);
	assert(output.Height() == 2);
	assert(output.Deep() == 1);

	assert(output(0, 0, 0) == 5.5);
	assert(output(0, 0, 1) == 6.25);
	assert(output(0, 1, 0) == 3.5);
	assert(output(0, 1, 1) == 5.75);

	Volume deltas(2, 2, 1);

	deltas(0, 0, 0) = 1.2;
	deltas(0, 0, 1) = 1.9;
	deltas(0, 1, 0) = 0.9;
	deltas(0, 1, 1) = 0.3;


	layer.Backward({deltas}, {input}, true);

	Volume& deltas2 = layer.GetDeltas()[0];

	assert(deltas2(0, 0, 0) == 1.2 / 4);
	assert(deltas2(0, 0, 1) == 1.2 / 4);
	assert(deltas2(0, 0, 2) == 1.9 / 4);
	assert(deltas2(0, 0, 3) == 1.9 / 4);

	assert(deltas2(0, 1, 0) == 1.2 / 4);
	assert(deltas2(0, 1, 1) == 1.2 / 4);
	assert(deltas2(0, 1, 2) == 1.9 / 4);
	assert(deltas2(0, 1, 3) == 1.9 / 4);

	assert(deltas2(0, 2, 0) == 0.9 / 4);
	assert(deltas2(0, 2, 1) == 0.9 / 4);
	assert(deltas2(0, 2, 2) == 0.3 / 4);
	assert(deltas2(0, 2, 3) == 0.3 / 4);

	assert(deltas2(0, 3, 0) == 0.9 / 4);
	assert(deltas2(0, 3, 1) == 0.9 / 4);
	assert(deltas2(0, 3, 2) == 0.3 / 4);
	assert(deltas2(0, 3, 3) == 0.3 / 4);

	cout << "OK" << endl;
}

void ConvLayerTest() {
	cout << "Conv tests: ";

	VolumeSize size;
	size.height = 5;
	size.width = 5;
	size.deep = 3;

	ConvLayer layer(size, 2, 3, 1, 2);
	layer.SetBatchSize(1);
	Volume input(5, 5, 3);

	double x[75] = { 1, 2, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 2, 0, 2, 2, 2, 2, 0, 1, 2, 0, 1, 0, 1, 1, 2, 2, 1, 2, 0, 2, 2, 0, 2, 1, 2, 2, 1, 1, 2, 2, 0, 1, 0, 2, 2, 1, 0, 0, 0, 2, 0, 1, 1, 2, 0, 2, 1, 1, 2, 0, 1, 2, 1, 0, 1, 0, 1, 2, 1, 0, 1, 2, 1 };
	double f1[27] = { -1, 1, 1, -1, 1, 1, 0, 0, 1, 0, -1, 1, -1, 0, -1, 1, 0, 0, 1, -1, 0, -1, 0, -1, 0, 1, -1 };
	double f2[27] = { 0, 0, -1, 1, 0, 0, 0, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 1, 1, 0, 1, 1, -1, 1, -1, 0, 0 };

	int k = 0;
	for (int d = 0; d < 3; d++)
		for (int i = 0; i < 5; i++)
			for (int j = 0; j < 5; j++)
				input(d, i, j) = x[k++];
	
	int index = 0;
	for (int d = 0; d < 3; d++)
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				layer.SetWeight(0, d, i, j, f1[index]);
				layer.SetWeight(1, d, i, j, f2[index]);
				index++;
			}

	layer.SetBias(0, 1);
	layer.SetBias(1, 0);

	layer.Forward({input});
	Volume output = layer.GetOutput()[0];

	assert(output.Width() == 3);
	assert(output.Height() == 3);
	assert(output.Deep() == 2);

	assert(output(0, 0, 0) == 2);
	assert(output(0, 0, 1) == 0);
	assert(output(0, 0, 2) == 0);
	assert(output(0, 1, 0) == 5);
	assert(output(0, 1, 1) == 0);
	assert(output(0, 1, 2) == 2);
	assert(output(0, 2, 0) == 5);
	assert(output(0, 2, 1) == 0);
	assert(output(0, 2, 2) == 0);

	assert(output(1, 0, 0) == 3);
	assert(output(1, 0, 1) == 5);
	assert(output(1, 0, 2) == 2);
	assert(output(1, 1, 0) == 1);
	assert(output(1, 1, 1) == 0);
	assert(output(1, 1, 2) == 0);
	assert(output(1, 2, 0) == 0);
	assert(output(1, 2, 1) == 4);
	assert(output(1, 2, 2) == 3);

	size.height = 4;
	size.width = 4;
	size.deep = 1;

	ConvLayer layer2(size, 1, 3, 0, 1);
	layer2.SetBatchSize(1);
	layer2.SetBias(0, 0);

	layer2.SetWeight(0, 0, 0, 0, 1);
	layer2.SetWeight(0, 0, 0, 1, 4);
	layer2.SetWeight(0, 0, 0, 2, 1);

	layer2.SetWeight(0, 0, 1, 0, 1);
	layer2.SetWeight(0, 0, 1, 1, 4);
	layer2.SetWeight(0, 0, 1, 2, 3);

	layer2.SetWeight(0, 0, 2, 0, 3);
	layer2.SetWeight(0, 0, 2, 1, 3);
	layer2.SetWeight(0, 0, 2, 2, 1);

	Volume X(4, 4, 1);

	X(0, 0, 0) = 4; X(0, 0, 1) = 5; X(0, 0, 2) = 8; X(0, 0, 3) = 7;
	X(0, 1, 0) = 1; X(0, 1, 1) = 8; X(0, 1, 2) = 8; X(0, 1, 3) = 8;
	X(0, 2, 0) = 3; X(0, 2, 1) = 6; X(0, 2, 2) = 6; X(0, 2, 3) = 4;
	X(0, 3, 0) = 6; X(0, 3, 1) = 5; X(0, 3, 2) = 7; X(0, 3, 3) = 8;

	layer2.Forward({X});
	output = layer2.GetOutput()[0];

	assert(output.Width() == 2);
	assert(output.Height() == 2);
	assert(output.Deep() == 1);

	assert(output(0, 0, 0) == 122);
	assert(output(0, 0, 1) == 148);
	assert(output(0, 1, 0) == 126);
	assert(output(0, 1, 1) == 134);

	Volume deltas(2, 2, 1);

	deltas(0, 0, 0) = 2;
	deltas(0, 0, 1) = 1;
	deltas(0, 1, 0) = 4;
	deltas(0, 1, 1) = 4;

	layer2.Backward({deltas}, {X}, true);

	Volume deltas2 = layer2.GetDeltas()[0];

	assert(fabs(deltas2(0, 0, 0) - 2) < 1e-14);
	assert(fabs(deltas2(0, 0, 1) - 9) < 1e-14);
	assert(fabs(deltas2(0, 0, 2) - 6) < 1e-14);
	assert(fabs(deltas2(0, 0, 3) - 1) < 1e-14);

	assert(fabs(deltas2(0, 1, 0) - 6) < 1e-14);
	assert(fabs(deltas2(0, 1, 1) - 29) < 1e-14);
	assert(fabs(deltas2(0, 1, 2) - 30) < 1e-14);
	assert(fabs(deltas2(0, 1, 3) - 7) < 1e-14);

	assert(fabs(deltas2(0, 2, 0) - 10) < 1e-14);
	assert(fabs(deltas2(0, 2, 1) - 29) < 1e-14);
	assert(fabs(deltas2(0, 2, 2) - 33) < 1e-14);
	assert(fabs(deltas2(0, 2, 3) - 13) < 1e-14);

	assert(fabs(deltas2(0, 3, 0) - 12) < 1e-14);
	assert(fabs(deltas2(0, 3, 1) - 24) < 1e-14);
	assert(fabs(deltas2(0, 3, 2) - 16) < 1e-14);
	assert(fabs(deltas2(0, 3, 3) - 4) < 1e-14);

	cout << "OK" << endl;
}

void DropoutTest() {
	cout << "Dropout tests: ";
	Volume input(1, 1, 10);

	for (int i = 0; i < 10; i++)
		input[i] = 1;

	VolumeSize size;
	size.height = 1;
	size.width = 1;
	size.deep = 10;

	DropoutLayer layer(size, 0.2);
	layer.SetBatchSize(1);

	for (int i = 0; i < 10; i++)
		layer.Forward({input});

	layer.ForwardOutput({input});
	Volume& output = layer.GetOutput()[0];

	double sum = 0;
	for (int j = 0; j < 10; j++)
		sum += output[j];

	assert(fabs(sum - 10) < 1e-14);
	cout << "OK" << endl;
}

void GradientCheckingTest() {
	VolumeSize inputSize;
	VolumeSize outputSize;

	inputSize.width = 28;
	inputSize.height = 28;
	inputSize.deep = 1;

	outputSize.width = 1;
	outputSize.height = 1;
	outputSize.deep = 10;

	int batchSize = 3;

	GaussRandom random;

	vector<Volume> inputs;
	vector<Volume> outputs;

	for (int i = 0; i < batchSize; i++) {
		inputs.push_back(Volume(inputSize));
		outputs.push_back(Volume(outputSize));

		inputs[i].FillRandom(random, 1);
		outputs[i].FillRandom(random, 1);
	}

	Network network(inputSize.width, inputSize.height, inputSize.deep);
	
	network.AddLayer("conv filters=16 filter_size=3 P=1");
	network.AddLayer("maxpool");
	network.AddLayer("conv filters=5 filter_size=3 P=1");
	network.AddLayer("maxpool");
	network.AddLayer("fullconnected outputs=40 activation=none");
	network.AddLayer("batchnormalization");
	network.AddLayer("relu");
	network.AddLayer("fullconnected outputs=20 activation=tanh");
	network.AddLayer("fullconnected outputs=16 activation=sigmoid");
	network.AddLayer("fullconnected outputs=10 activation=none");
	network.AddLayer("softmax");

	network.GradientChecking(inputs, outputs, LossType::BinaryCrossEntropy);
	network.GradientChecking(inputs, outputs, LossType::CrossEntropy);
	network.GradientChecking(inputs, outputs, LossType::MSE);
	network.GradientChecking(inputs, outputs, LossType::MAE);
	network.GradientChecking(inputs, outputs, LossType::Exp);
	network.GradientChecking(inputs, outputs, LossType::Logcosh);
}

int main() {
	ConvLayerTest();
	MaxPoolingLayerTest();
	AveragePoolingLayerTest();
	FullyConnectedLayerTest();
	DropoutTest();
	GradientCheckingTest();
}
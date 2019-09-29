#include <iostream>
#include <cassert>

#include "Layers/ConvLayer.hpp"
#include "Layers/ConvTransposedLayer.hpp"
#include "Layers/UpscaleLayer.hpp"
#include "Layers/UpscaleBilinearLayer.hpp"
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
	assert(output(0, 0, 1) == -3);
	assert(output(0, 0, 2) == -1);
	assert(output(0, 1, 0) == 5);
	assert(output(0, 1, 1) == -7);
	assert(output(0, 1, 2) == 2);
	assert(output(0, 2, 0) == 5);
	assert(output(0, 2, 1) == 0);
	assert(output(0, 2, 2) == 0);

	assert(output(1, 0, 0) == 3);
	assert(output(1, 0, 1) == 5);
	assert(output(1, 0, 2) == 2);
	assert(output(1, 1, 0) == 1);
	assert(output(1, 1, 1) == 0);
	assert(output(1, 1, 2) == -3);
	assert(output(1, 2, 0) == -2);
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

void ConvTransposedLayerTest() {
	cout << "Conv transposed tests: ";

	VolumeSize size;
	size.deep = 3;
	size.height = 5;
	size.width = 5;

	ConvTransposedLayer layer(size, 1, 3, 0, 1);
	layer.SetBatchSize(1);

	Volume input(size);
	
	layer.SetWeight(0, 0, 0, 0, 0);
	layer.SetWeight(0, 0, 0, 1, 1);
	layer.SetWeight(0, 0, 0, 2, 0);

	layer.SetWeight(0, 0, 1, 0, 0);
	layer.SetWeight(0, 0, 1, 1, 0);
	layer.SetWeight(0, 0, 1, 2, 2);

	layer.SetWeight(0, 0, 2, 0, 0);
	layer.SetWeight(0, 0, 2, 1, 1);
	layer.SetWeight(0, 0, 2, 2, 0);


	layer.SetWeight(0, 1, 0, 0, 2);
	layer.SetWeight(0, 1, 0, 1, 1);
	layer.SetWeight(0, 1, 0, 2, 0);

	layer.SetWeight(0, 1, 1, 0, 0);
	layer.SetWeight(0, 1, 1, 1, 0);
	layer.SetWeight(0, 1, 1, 2, 0);

	layer.SetWeight(0, 1, 2, 0, 0);
	layer.SetWeight(0, 1, 2, 1, 3);
	layer.SetWeight(0, 1, 2, 2, 0);


	layer.SetWeight(0, 2, 0, 0, 1);
	layer.SetWeight(0, 2, 0, 1, 0);
	layer.SetWeight(0, 2, 0, 2, 0);

	layer.SetWeight(0, 2, 1, 0, 1);
	layer.SetWeight(0, 2, 1, 1, 0);
	layer.SetWeight(0, 2, 1, 2, 0);

	layer.SetWeight(0, 2, 2, 0, 0);
	layer.SetWeight(0, 2, 2, 1, 0);
	layer.SetWeight(0, 2, 2, 2, 2);
	
	layer.SetBias(0, 0);

	input(0, 0, 0) = 1;
	input(0, 0, 1) = 0;
	input(0, 0, 2) = 1;
	input(0, 0, 3) = 0;
	input(0, 0, 4) = 2;

	input(0, 1, 0) = 1;
	input(0, 1, 1) = 1;
	input(0, 1, 2) = 3;
	input(0, 1, 3) = 2;
	input(0, 1, 4) = 1;

	input(0, 2, 0) = 1;
	input(0, 2, 1) = 1;
	input(0, 2, 2) = 0;
	input(0, 2, 3) = 1;
	input(0, 2, 4) = 1;

	input(0, 3, 0) = 2;
	input(0, 3, 1) = 3;
	input(0, 3, 2) = 2;
	input(0, 3, 3) = 1;
	input(0, 3, 4) = 3;

	input(0, 4, 0) = 0;
	input(0, 4, 1) = 2;
	input(0, 4, 2) = 0;
	input(0, 4, 3) = 1;
	input(0, 4, 4) = 0;


	input(1, 0, 0) = 1;
	input(1, 0, 1) = 0;
	input(1, 0, 2) = 0;
	input(1, 0, 3) = 1;
	input(1, 0, 4) = 0;

	input(1, 1, 0) = 2;
	input(1, 1, 1) = 0;
	input(1, 1, 2) = 1;
	input(1, 1, 3) = 2;
	input(1, 1, 4) = 0;

	input(1, 2, 0) = 3;
	input(1, 2, 1) = 1;
	input(1, 2, 2) = 1;
	input(1, 2, 3) = 3;
	input(1, 2, 4) = 0;

	input(1, 3, 0) = 0;
	input(1, 3, 1) = 3;
	input(1, 3, 2) = 0;
	input(1, 3, 3) = 3;
	input(1, 3, 4) = 2;

	input(1, 4, 0) = 1;
	input(1, 4, 1) = 0;
	input(1, 4, 2) = 3;
	input(1, 4, 3) = 2;
	input(1, 4, 4) = 1;


	input(2, 0, 0) = 2;
	input(2, 0, 1) = 0;
	input(2, 0, 2) = 1;
	input(2, 0, 3) = 2;
	input(2, 0, 4) = 1;

	input(2, 1, 0) = 3;
	input(2, 1, 1) = 3;
	input(2, 1, 2) = 1;
	input(2, 1, 3) = 3;
	input(2, 1, 4) = 2;

	input(2, 2, 0) = 2;
	input(2, 2, 1) = 1;
	input(2, 2, 2) = 1;
	input(2, 2, 3) = 1;
	input(2, 2, 4) = 0;

	input(2, 3, 0) = 3;
	input(2, 3, 1) = 1;
	input(2, 3, 2) = 3;
	input(2, 3, 3) = 2;
	input(2, 3, 4) = 0;

	input(2, 4, 0) = 1;
	input(2, 4, 1) = 1;
	input(2, 4, 2) = 2;
	input(2, 4, 3) = 1;
	input(2, 4, 4) = 1;

	layer.Forward({input});
	Volume output = layer.GetOutput()[0];

	assert(output.Width() == 7);
	assert(output.Height() == 7);
	assert(output.Deep() == 1);

	assert(output(0, 0, 0) == 4);
	assert(output(0, 0, 1) == 2);
	assert(output(0, 0, 2) == 1);
	assert(output(0, 0, 3) == 5);
	assert(output(0, 0, 4) == 2);
	assert(output(0, 0, 5) == 2);
	assert(output(0, 0, 6) == 0);

	assert(output(0, 1, 0) == 9);
	assert(output(0, 1, 1) == 6);
	assert(output(0, 1, 2) == 7);
	assert(output(0, 1, 3) == 13);
	assert(output(0, 1, 4) == 9);
	assert(output(0, 1, 5) == 1);
	assert(output(0, 1, 6) == 4);

	assert(output(0, 2, 0) == 11);
	assert(output(0, 2, 1) == 14);
	assert(output(0, 2, 2) == 12);
	assert(output(0, 2, 3) == 14);
	assert(output(0, 2, 4) == 17);
	assert(output(0, 2, 5) == 11);
	assert(output(0, 2, 6) == 4);

	assert(output(0, 3, 0) == 5);
	assert(output(0, 3, 1) == 17);
	assert(output(0, 3, 2) == 19);
	assert(output(0, 3, 3) == 25);
	assert(output(0, 3, 4) == 18);
	assert(output(0, 3, 5) == 14);
	assert(output(0, 3, 6) == 6);

	assert(output(0, 4, 0) == 6);
	assert(output(0, 4, 1) == 13);
	assert(output(0, 4, 2) == 25);
	assert(output(0, 4, 3) == 21);
	assert(output(0, 4, 4) == 22);
	assert(output(0, 4, 5) == 6);
	assert(output(0, 4, 6) == 6);

	assert(output(0, 5, 0) == 1);
	assert(output(0, 5, 1) == 3);
	assert(output(0, 5, 2) == 20);
	assert(output(0, 5, 3) == 9);
	assert(output(0, 5, 4) == 17);
	assert(output(0, 5, 5) == 15);
	assert(output(0, 5, 6) == 0);

	assert(output(0, 6, 0) == 0);
	assert(output(0, 6, 1) == 3);
	assert(output(0, 6, 2) == 4);
	assert(output(0, 6, 3) == 11);
	assert(output(0, 6, 4) == 11);
	assert(output(0, 6, 5) == 5);
	assert(output(0, 6, 6) == 2);

	cout << "OK" << endl;
}

void UpscaleLayerTest() {
	cout << "Upscale tests: ";

	VolumeSize size;
	size.deep = 1;
	size.height = 2;
	size.width = 2;

	UpscaleLayer layer(size, 2);
	layer.SetBatchSize(1);

	Volume input(size);

	input(0, 0, 0) = 1;
	input(0, 0, 1) = 2;
	input(0, 1, 0) = 3;
	input(0, 1, 1) = 4;

	layer.Forward({input});
	Volume output = layer.GetOutput()[0];

	assert(output.Width() == 4);
	assert(output.Height() == 4);
	assert(output.Deep() == 1);

	assert(output(0, 0, 0) == 1);
	assert(output(0, 0, 1) == 1);
	assert(output(0, 1, 0) == 1);
	assert(output(0, 1, 1) == 1);

	assert(output(0, 0, 2) == 2);
	assert(output(0, 0, 3) == 2);
	assert(output(0, 1, 2) == 2);
	assert(output(0, 1, 3) == 2);

	assert(output(0, 2, 0) == 3);
	assert(output(0, 2, 1) == 3);
	assert(output(0, 3, 0) == 3);
	assert(output(0, 3, 1) == 3);

	assert(output(0, 2, 2) == 4);
	assert(output(0, 2, 3) == 4);
	assert(output(0, 3, 2) == 4);
	assert(output(0, 3, 3) == 4);

	Volume deltas(output.GetSize());

	deltas(0, 0, 0) = 1;
	deltas(0, 0, 1) = 2;
	deltas(0, 1, 0) = 0;
	deltas(0, 1, 1) = -1;

	deltas(0, 0, 2) = -1;
	deltas(0, 0, 3) = 2;
	deltas(0, 1, 2) = 2;
	deltas(0, 1, 3) = 0;

	deltas(0, 2, 0) = 0;
	deltas(0, 2, 1) = 0;
	deltas(0, 3, 0) = 0;
	deltas(0, 3, 1) = 0;

	deltas(0, 2, 2) = -1;
	deltas(0, 2, 3) = -2;
	deltas(0, 3, 2) = -3;
	deltas(0, 3, 3) = -4;

	layer.Backward({deltas}, {input}, true);
	Volume& dX = layer.GetDeltas()[0];

	assert(dX.Width() == 2);
	assert(dX.Height() == 2);
	assert(dX.Deep() == 1);

	assert(dX(0, 0, 0) == 2);
	assert(dX(0, 0, 1) == 3);
	assert(dX(0, 1, 0) == 0);
	assert(dX(0, 1, 1) == -10);

	cout << "OK" << endl;
}

void UpscaleBilinearLayerTest() {
	cout << "Upscale bilinear tests: ";

	VolumeSize size;
	size.deep = 1;
	size.height = 2;
	size.width = 2;

	UpscaleBilinearLayer layer(size, 2);
	layer.SetBatchSize(1);

	Volume input(size);

	input(0, 0, 0) = 1;
	input(0, 0, 1) = 2;
	input(0, 1, 0) = 3;
	input(0, 1, 1) = 4;

	layer.Forward({input});
	Volume output = layer.GetOutput()[0];

	assert(output.Width() == 4);
	assert(output.Height() == 4);
	assert(output.Deep() == 1);

	assert(output(0, 0, 0) == 1);
	assert(output(0, 0, 1) == 1.5);
	assert(output(0, 1, 0) == 2);
	assert(output(0, 1, 1) == 2.5);

	assert(output(0, 0, 2) == 2);
	assert(output(0, 0, 3) == 2.5);
	assert(output(0, 1, 2) == 3);
	assert(output(0, 1, 3) == 3.5);

	assert(output(0, 2, 0) == 3);
	assert(output(0, 2, 1) == 3.5);
	assert(output(0, 3, 0) == 4);
	assert(output(0, 3, 1) == 4.5);

	assert(output(0, 2, 2) == 4);
	assert(output(0, 2, 3) == 4.5);
	assert(output(0, 3, 2) == 5);
	assert(output(0, 3, 3) == 5.5);

	Volume deltas(output.GetSize());

	deltas(0, 0, 0) = 1;
	deltas(0, 0, 1) = 2;
	deltas(0, 1, 0) = 0;
	deltas(0, 1, 1) = -1;

	deltas(0, 0, 2) = -1;
	deltas(0, 0, 3) = 2;
	deltas(0, 1, 2) = 2;
	deltas(0, 1, 3) = 0;

	deltas(0, 2, 0) = 0;
	deltas(0, 2, 1) = 0;
	deltas(0, 3, 0) = 0;
	deltas(0, 3, 1) = 0;

	deltas(0, 2, 2) = -1;
	deltas(0, 2, 3) = -2;
	deltas(0, 3, 2) = -3;
	deltas(0, 3, 3) = -4;

	layer.Backward({deltas}, {input}, true);
	Volume& dX = layer.GetDeltas()[0];

	assert(dX.Width() == 2);
	assert(dX.Height() == 2);
	assert(dX.Deep() == 1);

	assert(dX(0, 0, 0) == -0.25);
	assert(dX(0, 0, 1) == 8.25);
	assert(dX(0, 1, 0) == 3.75);
	assert(dX(0, 1, 1) == -16.75);

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

	default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);

	vector<Volume> inputs;
	vector<Volume> outputs;

	for (int i = 0; i < batchSize; i++) {
		inputs.push_back(Volume(inputSize));
		outputs.push_back(Volume(outputSize));

		for (int j = 0; j < inputSize.width * inputSize.height * inputSize.deep; j++)
			inputs[i][j] = distribution(generator);

		for (int j = 0; j < outputSize.width * outputSize.height * outputSize.deep; j++)
			outputs[i][j] = distribution(generator);
	}

	Network network(inputSize.width, inputSize.height, inputSize.deep);
	
	network.AddLayer("conv filters=16 filter_size=3 P=1");
	network.AddLayer("relu");
	network.AddLayer("maxpool");
	network.AddLayer("conv filters=5 filter_size=3 P=1 S=2");
	network.AddLayer("relu");
	network.AddLayer("convtransposed filters=5 filter_size=2 P=1 S=2");
	network.AddLayer("maxpool");
	network.AddLayer("convtransposed filters=8 filter_size=3 P=1");
	network.AddLayer("maxpool");
	network.AddLayer("fullconnected outputs=40 activation=none");
	network.AddLayer("batchnormalization");
	network.AddLayer("relu");
	network.AddLayer("fullconnected outputs=20 activation=tanh");
	network.AddLayer("fullconnected outputs=16 activation=sigmoid");
	network.AddLayer("fullconnected outputs=10 activation=none");
	network.AddLayer("softmax");

	network.GradientChecking(inputs, outputs, LossFunction::BinaryCrossEntropy());
	network.GradientChecking(inputs, outputs, LossFunction::CrossEntropy());
	network.GradientChecking(inputs, outputs, LossFunction::MSE());
	network.GradientChecking(inputs, outputs, LossFunction::MAE());
	network.GradientChecking(inputs, outputs, LossFunction::Exp());
	network.GradientChecking(inputs, outputs, LossFunction::Logcosh());
}

int main() {
	ConvLayerTest();
	ConvTransposedLayerTest();
	UpscaleLayerTest();
	UpscaleBilinearLayerTest();
	MaxPoolingLayerTest();
	AveragePoolingLayerTest();
	FullyConnectedLayerTest();
	DropoutTest();
	GradientCheckingTest();
}
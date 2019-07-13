#include <iostream>
#include <cassert>

#include "Layers/ConvLayer.hpp"
#include "Layers/MaxPoolingLayer.hpp"
#include "Layers/FlattenLayer.hpp"
#include "Layers/FullConnectedLayer.hpp"

using namespace std;

void FullConnectedLayerTest() {
	cout << "Full connected tests: ";
	FullConnectedLayer layer(8, 4, "relu");
	
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

	layer.Forward(input);

	Volume& output = layer.GetOutput();

	assert(output.Width() == 1);
	assert(output.Height() == 1);
	assert(output.Deep() == 4);

	assert(output(0, 0, 0) == 36);
	assert(output(1, 0, 0) == 0);
	assert(output(2, 0, 0) == 46);
	assert(output(3, 0, 0) == 15);

	Volume prev(1, 1, 8);

	for (int i = 0; i < 8; i++)
		prev(i, 0, 0) = 1;

	layer.GetDeltas()(0, 0, 0) = -0.5;
	layer.GetDeltas()(1, 0, 0) = 0.1;
	layer.GetDeltas()(2, 0, 0) = -0.25;
	layer.GetDeltas()(3, 0, 0) = 0.7;

	layer.Backward(prev);

	assert(fabs(prev(0, 0, 0) - 0.05) < 1e-15);
	assert(fabs(prev(1, 0, 0) - 0.15) < 1e-15);
	assert(fabs(prev(2, 0, 0) + 0.35) < 1e-15);
	assert(fabs(prev(3, 0, 0) + 1.45) < 1e-15);
	assert(fabs(prev(4, 0, 0) - 2.55) < 1e-15);
	assert(fabs(prev(5, 0, 0) - 3.6) < 1e-15);
	assert(fabs(prev(6, 0, 0) - 0.35) < 1e-15);
	assert(fabs(prev(7, 0, 0) + 1.7) < 1e-15);

	std::cout << "OK" << std::endl;
}

void FlattenLayerTest() {
	cout << "Flatten tests: ";

	FlattenLayer layer(2, 3, 2);
	Volume input(2, 3, 2);

	input(0, 0, 0) = 1;
	input(0, 0, 1) = 2;
	input(0, 1, 0) = 3;
	input(0, 1, 1) = 4;
	input(0, 2, 0) = 5;
	input(0, 2, 1) = 6;

	input(1, 0, 0) = 1;
	input(1, 0, 1) = 0;
	input(1, 1, 0) = -1;
	input(1, 1, 1) = 0;
	input(1, 2, 0) = 5;
	input(1, 2, 1) = 0;

	layer.Forward(input);
	Volume output = layer.GetOutput();

	assert(fabs(output(0, 0, 0) - 1) < 1e-15);
	assert(fabs(output(1, 0, 0) - 2) < 1e-15);
	assert(fabs(output(2, 0, 0) - 3) < 1e-15);
	assert(fabs(output(3, 0, 0) - 4) < 1e-15);
	assert(fabs(output(4, 0, 0) - 5) < 1e-15);
	assert(fabs(output(5, 0, 0) - 6) < 1e-15);

	assert(fabs(output(6, 0, 0) - 1) < 1e-15);
	assert(fabs(output(7, 0, 0) - 0) < 1e-15);
	assert(fabs(output(8, 0, 0) + 1) < 1e-15);
	assert(fabs(output(9, 0, 0) - 0) < 1e-15);
	assert(fabs(output(10, 0, 0) - 5) < 1e-15);
	assert(fabs(output(11, 0, 0) - 0) < 1e-15);

	Volume deltas2(2, 3, 2);

	for (int d = 0; d < 2; d++)
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 2; j++)
				deltas2(d, i, j) = 1;

	for (int i = 0; i < 12; i++)
		layer.GetDeltas()(i, 0, 0) = output(i, 0, 0);
	
	layer.Backward(deltas2);

	assert(fabs(deltas2(0, 0, 0) - 1) < 1e-15);
	assert(fabs(deltas2(0, 0, 1) - 2) < 1e-15);
	assert(fabs(deltas2(0, 1, 0) - 3) < 1e-15);
	assert(fabs(deltas2(0, 1, 1) - 4) < 1e-15);
	assert(fabs(deltas2(0, 2, 0) - 5) < 1e-15);
	assert(fabs(deltas2(0, 2, 1) - 6) < 1e-15);

	assert(fabs(deltas2(1, 0, 0) - 1) < 1e-15);
	assert(fabs(deltas2(1, 0, 1) - 0) < 1e-15);
	assert(fabs(deltas2(1, 1, 0) + 1) < 1e-15);
	assert(fabs(deltas2(1, 1, 1) - 0) < 1e-15);
	assert(fabs(deltas2(1, 2, 0) - 5) < 1e-15);
	assert(fabs(deltas2(1, 2, 1) - 0) < 1e-15);

	cout << "OK" << endl;
}

void MaxPoolingLayerTest() {
	cout << "Max pooling tests: ";
	MaxPoolingLayer layer(4, 4, 1, 2);
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


	layer.Forward(input);
	Volume output = layer.GetOutput();

	assert(output.Width() == 2);
	assert(output.Height() == 2);
	assert(output.Deep() == 1);

	assert(output(0, 0, 0) == 9);
	assert(output(0, 0, 1) == 8);
	assert(output(0, 1, 0) == 7);
	assert(output(0, 1, 1) == 9);

	layer.GetDeltas()(0, 0, 0) = 1.2;
	layer.GetDeltas()(0, 0, 1) = 1.9;
	layer.GetDeltas()(0, 1, 0) = 0.9;
	layer.GetDeltas()(0, 1, 1) = 0.3;

	Volume deltas2(4, 4, 1);

	for (int d = 0; d < 1; d++)
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				deltas2(d, i, j) = 1;

	layer.Backward(deltas2);

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

void ConvLayerTest() {
	cout << "Conv tests: ";
	ConvLayer layer(5, 5, 3, 2, 3, 1, 2);
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
				layer.SetFilter(0, d, i, j, f1[index]);
				layer.SetFilter(1, d, i, j, f2[index]);
				index++;
			}

	layer.SetBias(0, 1);
	layer.SetBias(1, 0);

	layer.Forward(input);
	Volume output = layer.GetOutput();

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

	ConvLayer layer2(4, 4, 1, 1, 3);
	layer2.SetBias(0, 0);

	layer2.SetFilter(0, 0, 0, 0, 1);
	layer2.SetFilter(0, 0, 0, 1, 4);
	layer2.SetFilter(0, 0, 0, 2, 1);

	layer2.SetFilter(0, 0, 1, 0, 1);
	layer2.SetFilter(0, 0, 1, 1, 4);
	layer2.SetFilter(0, 0, 1, 2, 3);

	layer2.SetFilter(0, 0, 2, 0, 3);
	layer2.SetFilter(0, 0, 2, 1, 3);
	layer2.SetFilter(0, 0, 2, 2, 1);

	Volume X(4, 4, 1);

	X(0, 0, 0) = 4; X(0, 0, 1) = 5; X(0, 0, 2) = 8; X(0, 0, 3) = 7;
	X(0, 1, 0) = 1; X(0, 1, 1) = 8; X(0, 1, 2) = 8; X(0, 1, 3) = 8;
	X(0, 2, 0) = 3; X(0, 2, 1) = 6; X(0, 2, 2) = 6; X(0, 2, 3) = 4;
	X(0, 3, 0) = 6; X(0, 3, 1) = 5; X(0, 3, 2) = 7; X(0, 3, 3) = 8;

	layer2.Forward(X);
	output = layer2.GetOutput();

	assert(output.Width() == 2);
	assert(output.Height() == 2);
	assert(output.Deep() == 1);

	assert(output(0, 0, 0) == 122);
	assert(output(0, 0, 1) == 148);
	assert(output(0, 1, 0) == 126);
	assert(output(0, 1, 1) == 134);

	layer2.GetDeltas()(0, 0, 0) = 2;
	layer2.GetDeltas()(0, 0, 1) = 1;
	layer2.GetDeltas()(0, 1, 0) = 4;
	layer2.GetDeltas()(0, 1, 1) = 4;

	Volume deltas2(4, 4, 1);

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			deltas2(0, i, j) = 1;

	layer2.Backward(deltas2);

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

int main() {
	ConvLayerTest();
	MaxPoolingLayerTest();
	FlattenLayerTest();
	FullConnectedLayerTest();
}
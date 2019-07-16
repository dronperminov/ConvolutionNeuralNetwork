COMPILER=g++
FLAGS=-O3 -fopenmp

all: mnist cifar10 optimizers

mnist:
	$(COMPILER) $(FLAGS) examples/mnist_cnn.cpp -o examples/mnist_cnn

cifar10:
	$(COMPILER) $(FLAGS) examples/cifar10_cnn.cpp -o examples/cifar10_cnn

optimizers:
	$(COMPILER) $(FLAGS) examples/optimizers_test.cpp -o examples/optimizers_test

tests:
	$(COMPILER) $(FLAGS) tests.cpp -o tests

clean:
	rm -rf *.exe
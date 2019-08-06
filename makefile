COMPILER=g++
FLAGS=-O3 -fopenmp -march=native -mtune=native -ffast-math -mavx2

all: mnist cifar10 optimizers errors

mnist:
	$(COMPILER) $(FLAGS) examples/mnist_cnn.cpp -o examples/mnist_cnn

cifar10:
	$(COMPILER) $(FLAGS) examples/cifar10_cnn.cpp -o examples/cifar10_cnn

optimizers:
	$(COMPILER) $(FLAGS) examples/optimizers_test.cpp -o examples/optimizers

errors:
	$(COMPILER) $(FLAGS) examples/errors_test.cpp -o examples/errors

tests:
	$(COMPILER) $(FLAGS) tests.cpp -o tests

clean:
	rm -rf *.exe
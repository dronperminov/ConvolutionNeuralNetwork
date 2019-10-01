COMPILER=g++
FLAGS=-O3 -fopenmp -march=native -mtune=native -ffast-math -mavx2

all: mnist cifar10 gan dcgan optimizers activations compares losses errors augmentation tests

mnist:
	$(COMPILER) $(FLAGS) examples/mnist_cnn.cpp -o examples/mnist_cnn

cifar10:
	$(COMPILER) $(FLAGS) examples/cifar10_cnn.cpp -o examples/cifar10_cnn

optimizers:
	$(COMPILER) $(FLAGS) examples/optimizers_test.cpp -o examples/optimizers

activations:
	$(COMPILER) $(FLAGS) examples/activations_test.cpp -o examples/activations

compares:
	$(COMPILER) $(FLAGS) examples/compare_test.cpp -o examples/compares

losses:
	$(COMPILER) $(FLAGS) examples/losses_test.cpp -o examples/losses

errors:
	$(COMPILER) $(FLAGS) examples/errors_test.cpp -o examples/errors

augmentation:
	$(COMPILER) $(FLAGS) examples/augmentation_test.cpp -o examples/augmentation

gan:
	$(COMPILER) $(FLAGS) examples/gan.cpp -o examples/gan

dcgan:
	$(COMPILER) $(FLAGS) examples/dcgan.cpp -o examples/dcgan

vae:
	$(COMPILER) $(FLAGS) examples/vae.cpp -o examples/vae

tests:
	$(COMPILER) $(FLAGS) tests.cpp -o tests

clean:
	rm -rf *.exe
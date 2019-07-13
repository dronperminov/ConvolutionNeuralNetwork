COMPILER=g++
FLAGS=-O3 -fopenmp

all:
	$(COMPILER) $(FLAGS) main.cpp -o main

tests:
	$(COMPILER) $(FLAGS) tests.cpp -o tests

clean:
	rm -rf tests main
CC=gcc
CXX=g++
CXXFLAGS=-std=c++0x 
TARGET=out

all: $(TARGET)

clean:
	rm -f *.o
	rm -f $(TARGET)

out:
	cd $(model);$(CXX) $(CXXFLAGS) C_verifier.cpp -o $@


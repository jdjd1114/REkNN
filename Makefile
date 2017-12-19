SRC = ./src
BIN = ./bin

BINGKNN = $(BIN)/Gknn
BINCKNN = $(BIN)/Cknn

TARGET := $(BINGKNN) $(BINCKNN)

CC = nvcc
GCC = g++

$(BINGKNN): $(BIN)/gmain.o
	$(CC) -o $(BINGKNN) $(BIN)/gmain.o -Xlinker -rpath=/opt/MATLAB/R2014a/bin/glnxa64 -L/opt/MATLAB/R2014a/bin/glnxa64 -lmat -lmx -lcublas -I/opt/MATLAB/R2014a/extern/include -DCUDA_VERSION0
	rm -rf $(BIN)/gmain.o

$(BINCKNN): $(BIN)/cmain.o
	$(GCC) -o $(BINCKNN) $(BIN)/cmain.o -Xlinker -rpath=/opt/MATLAB/R2014a/bin/glnxa64 -L/opt/MATLAB/R2014a/bin/glnxa64 -lmat -lmx -I/opt/MATLAB/R2014a/extern/include
	rm -rf $(BIN)/cmain.o

$(BIN)/gmain.o: $(SRC)/knn_tile1.cu
	$(CC) -o $(BIN)/gmain.o -c -arch=sm_35 $(SRC)/knn_tile1.cu -Xlinker -rpath=/opt/MATLAB/R2014a/bin/glnxa64 -L/opt/MATLAB/R2014a/bin/glnxa64 -lmat -lmx -lcublas -I/opt/MATLAB/R2014a/extern/include -DCUDA_VERSION0

$(BIN)/cmain.o: $(SRC)/knnclassifier.cpp
	$(GCC) -o $(BIN)/cmain.o -c $(SRC)/knnclassifier.cpp -Xlinker -rpath=/opt/MATLAB/R2014a/bin/glnxa64 -L/opt/MATLAB/R2014a/bin/glnxa64 -lmat -lmx -I/opt/MATLAB/R2014a/extern/include

all: $(TARGET)
clean:
	-rm -rf $(BIN)/*

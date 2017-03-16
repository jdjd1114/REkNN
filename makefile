SRC = ./src
BIN = ./bin

BINGCN = $(BIN)/knn

CC = nvcc

$(BINGCN): $(BIN)/main.o
	$(CC) -o $(BINGCN) $(BIN)/main.o -Xlinker -rpath=/opt/MATLAB/R2014a/bin/glnxa64 -L/opt/MATLAB/R2014a/bin/glnxa64 -lmat -lmx -lcublas -I/opt/MATLAB/R2014a/extern/include -DCUDA_VERSION0 -Xptxas=-v
	rm -rf $(BIN)/*.o

$(BIN)/main.o: $(SRC)/knn_tile1.cu
	$(CC) -o $(BIN)/main.o -c -arch=sm_35 $(SRC)/knn_tile1.cu -Xlinker -rpath=/opt/MATLAB/R2014a/bin/glnxa64 -L/opt/MATLAB/R2014a/bin/glnxa64 -lmat -lmx -lcublas -I/opt/MATLAB/R2014a/extern/include -DCUDA_VERSION0 -Xptxas=-v

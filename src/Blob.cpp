#include"Blob.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

Blob::Blob() {
  this -> batch = 0;
  this -> channel = 0;
  this -> row = 0;
  this -> col = 0;
  matrixPtr = NULL;
}

Blob::Blob(int batch, int channel, int row, int col, std::string init) {
  matrixPtr = NULL;

  initialize(batch, channel, row, col, init);
}

void Blob::initialize(int batch, int channel, int row, int col, std::string init) {
  if (matrixPtr != NULL) {
    delete []matrixPtr;
  }
  matrixPtr = NULL;

  this -> batch = batch;
  this -> channel = channel;
  this -> row = row;
  this -> col = col;

  matrixPtr = new MatrixXd[batch * channel];

  srand(time(NULL));

  for (int i = 0; i < batch * channel; ++i) {
    if (init == "random") {
      matrixPtr[i].setZero(row, col);

      matrixPtr[i] = matrixPtr[i].unaryExpr([](double item) {
        //std::cout << 2.0 * rand() / RAND_MAX - 1.0 << std::endl;
        return (2.0 * rand() / RAND_MAX - 1.0) / 10.0;
        //return rand() / RAND_MAX / 1000.0;
      });

    } else if (init == "ones") {
      matrixPtr[i].setOnes(row, col);
    } else {
      matrixPtr[i].setZero(row, col);
    }
  }
}

Blob::~Blob() {
  if (matrixPtr != NULL) {
    delete []matrixPtr;
  }

  matrixPtr = NULL;
}

MatrixXd& Blob::getMatrixByIndex(int i, int j) const {
  return matrixPtr[i * channel + j];
}

double& Blob::getValueByIndex(int i, int j, int x, int y) const {
  return (matrixPtr[i * channel + j])(x, y);
}

void Blob::setMatrixByIndex(int i, int j, const MatrixXd& matrix) {
  matrixPtr[i * channel + j] = matrix;
}

void Blob::setValueByIndex(int i, int j, int x, int y, const double& value) {
  (matrixPtr[i * channel + j])(x, y) = value;
}
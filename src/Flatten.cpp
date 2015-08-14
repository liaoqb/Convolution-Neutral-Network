#include"Flatten.h"
#include <iostream>

void Flatten::forwardPropagation(const Blob& input, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < input.getChannel(); ++j) {
      MatrixXd matrix = input.getMatrixByIndex(i, j);

      for (int k = 0; k < matrix.rows() * matrix.cols(); ++k) {
        output.setValueByIndex(i, j * (matrix.rows() * matrix.cols()) + k, 0, 0, matrix(k / matrix.cols(), k % matrix.cols()));
      }
    }
  }
}

void Flatten::backwardPropagation(const Blob& input, const Blob& values, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      MatrixXd matrix = MatrixXd::Zero(output.getRow(), output.getCol());

      for (int k = 0; k < matrix.rows() * matrix.cols(); ++k) {
        matrix(k / matrix.cols(), k % matrix.cols()) = input.getValueByIndex(i, j * (matrix.rows() * matrix.cols()) + k, 0, 0);
      }

      output.setMatrixByIndex(i, j, matrix);
    }
  }
}
#include"Dropout.h"
#include <iostream>

Dropout::Dropout() {
  batch = 0;
  channel = 0;
  row = 0;
  col = 0;
  boxHeight = 0;
  boxWidth = 0;
  step = 0;
}

Dropout::Dropout(int batch, int channel, int row, int col, int boxHeight, int boxWidth, int step) {
  this ->batch = batch;
  this ->channel = channel;
  this ->row = row;
  this ->col = col;
  this ->boxHeight = boxHeight;
  this ->boxWidth = boxWidth;
  this -> step = step;
}

  // The batch is for store the position, take care please
void Dropout::setBatch(int batch) {
  this ->batch = batch;
}

void Dropout::forwardPropagation(const Blob& input, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      MatrixXd inputMatrix = input.getMatrixByIndex(i, j);
      MatrixXd outputMatrix = MatrixXd::Zero(output.getRow(), output.getCol());

      for (int x = 0; x < inputMatrix.rows(); x += step) {
        for (int y = 0; y < inputMatrix.cols(); y += step) {
          MatrixXd box = inputMatrix.block(x, y, boxHeight, boxWidth);

          double index = 0;
          for (int m = 0; m < box.rows(); ++m) {
            for (int n = 0; n < box.cols(); ++n)
              index += box(m, n);
          }

          outputMatrix(x / step, y / step) = index / (box.rows() * box.cols());
        }
      }
      //std::cout << output.getMatrixByIndex(i, j) << std::endl;
      output.setMatrixByIndex(i, j, outputMatrix);
    }
  }
}
  // input is the error, value is the values
void Dropout::backwardPropagation(const Blob& input, const Blob& values, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      MatrixXd matrix = MatrixXd::Zero(output.getRow(), output.getCol());

      for (int x = 0; x < matrix.rows(); ++x) {
        for (int y = 0; y < matrix.cols(); ++y) {
          matrix(x, y) = input.getValueByIndex(i, j, x / step, y / step);
        }
      }

      output.setMatrixByIndex(i, j, matrix);
    }
  }
}
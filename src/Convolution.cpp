#include"Convolution.h"
#include <string>
#include <iostream>

Convolution::Convolution() {
  this ->batch = 0;
  this ->channel = 0;
  this ->row = 0;
  this ->col = 0;
}

Convolution::Convolution(int batch, int channel, int row, int col, double rate) {
  this ->batch = batch;
  this ->channel = channel;
  this ->row = row;
  this ->col = col;
  this ->rate = rate;

  this ->bias.initialize(batch, channel, 1, 1, "random");
  this ->kernels.initialize(batch, channel, row, col, "random");
}

void Convolution::setBatch(int batch) {
  this ->batch = batch;

  //std::cout << kernels.getMatrixByIndex(0, 0) << std::endl;
}

MatrixXd Convolution::correlation(const MatrixXd& matrix, const MatrixXd& kernel, std::string way) {
  MatrixXd newMatrix = matrix;
  if (way == "full") {
    newMatrix = spanMatrix(matrix, kernel.rows() - 1, kernel.cols() - 1);
  }

  MatrixXd result = MatrixXd::Zero(newMatrix.rows() - kernel.rows() + 1, newMatrix.cols() - kernel.cols() + 1);

  for (int i = 0; i < result.rows(); ++i) {
    for (int j = 0; j < result.cols(); ++j) {
      result(i, j) = ((newMatrix.block(i, j, kernel.rows(), kernel.cols())).cwiseProduct(kernel)).sum();
    }
  }

  //std::cout << newMatrix << std::endl;

  return result;
}

MatrixXd Convolution::convolution(const MatrixXd& matrix, const MatrixXd& kernel, std::string way) {
  MatrixXd newKernel = rotateMatrix(kernel);

  return correlation(matrix, newKernel, way);
}

MatrixXd Convolution::spanMatrix(const MatrixXd& matrix, int spanRows, int spanCols) {
  MatrixXd newMatrix = MatrixXd::Zero(matrix.rows() + spanRows * 2, matrix.cols() + spanCols * 2);

  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      newMatrix(i + spanRows, j + spanCols) = matrix(i, j);
    }
  }

  //std::cout << newMatrix << std::endl;

  return newMatrix;
}

MatrixXd Convolution::rotateMatrix(const MatrixXd& matrix) {
  MatrixXd newMatrix(matrix.rows(), matrix.cols());

  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      newMatrix(i, j) = matrix(matrix.rows() - i - 1, matrix.cols() - j - 1);
    }
  }

  //std::cout << newMatrix << std::endl;

  return newMatrix;
}

void Convolution::forwardPropagation(const Blob& input, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      MatrixXd outputMatrix = MatrixXd::Zero(output.getRow(), output.getCol());

        // maybe a bit complex
      for (int k = 0; k < input.getChannel(); ++k) {
        //std::cout << correlation(input.getMatrixByIndex(i, k), kernels.getMatrixByIndex(0, j), "none") << std::endl;
        outputMatrix += correlation(input.getMatrixByIndex(i, k), kernels.getMatrixByIndex(0, j), "none");
        //std::cout << outputMatrix - outputMatrix.unaryExpr([&](double item) {return item + bias.getValueByIndex(0, j, 0, 0);}) << std::endl;
        outputMatrix = outputMatrix.unaryExpr([&](double item) {return item - bias.getValueByIndex(0, j, 0, 0);});
        //std::cout << outputMatrix << std::endl;
      }
      //std::cout << kernels.getMatrixByIndex(0, j) << std::endl;
      //std::cout << outputMatrix << std::endl;

      output.setMatrixByIndex(i, j, outputMatrix);
      //std::cout << output.getMatrixByIndex(i, j) << std::endl;
    }
  }
}

  // I am not quite sure about this function, take care please
void Convolution::backwardPropagation(const Blob& input, const Blob& values, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      MatrixXd matrix = MatrixXd::Zero(output.getRow(), output.getCol());
      
      for (int k = 0; k < kernels.getChannel(); ++k) {
        matrix += convolution(input.getMatrixByIndex(i, k), kernels.getMatrixByIndex(0, k), "full");
      }

      output.setMatrixByIndex(i, j, matrix);
      //std::cout << output.getMatrixByIndex(i, j) << std::endl;
    }
  }

    // update kernels and bias
  Blob updateBias(1, kernels.getChannel(), input.getRow(), input.getCol(), "zero");
  Blob updateKernels(1, kernels.getChannel(), kernels.getRow(), kernels.getCol(), "zero");

  for (int i = 0; i < input.getBatch(); ++i) {
    for (int j = 0; j < input.getChannel(); ++j) {
      updateBias.setMatrixByIndex(0, j, updateBias.getMatrixByIndex(0, j) + input.getMatrixByIndex(i, j));

      for (int k = 0; k < output.getChannel(); ++k) {
        updateKernels.setMatrixByIndex(0, j, updateKernels.getMatrixByIndex(0, j) + correlation(
          values.getMatrixByIndex(i, k), input.getMatrixByIndex(i, j), "none"));
      }
    }
  }

  for (int i = 0; i < kernels.getChannel(); ++i) {
    bias.setValueByIndex(0, i, 0, 0, bias.getValueByIndex(0, i, 0, 0) +
      rate * (updateBias.getMatrixByIndex(0, i) / input.getBatch()).sum());

    //std::cout << rate * updateKernels.getMatrixByIndex(0, i) / input.getBatch() << std::endl;

    kernels.setMatrixByIndex(0, i, kernels.getMatrixByIndex(0, i) + rate * updateKernels.getMatrixByIndex(0, i) / input.getBatch());

    //std::cout << bias.getMatrixByIndex(0, i) << std::endl;
    //std::cout << kernels.getMatrixByIndex(0, i) << std::endl;
  }
}
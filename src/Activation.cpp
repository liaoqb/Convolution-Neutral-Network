#include"Activation.h"
#include <iostream>
#include <cmath>

void Activation::forwardPropagation(const Blob& input, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      output.setMatrixByIndex(i, j, input.getMatrixByIndex(i, j).unaryExpr([](double item) {
        return 1.0 / (1.0 + exp(-item));
      }));
    }
  }
}
  // input is the error, value is the values
void Activation::backwardPropagation(const Blob& input, const Blob& values, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      //std::cout << input.getChannel() << std::endl;
      //std::cout << output.getChannel() << std::endl;
      //std::cout << output.getMatrixByIndex(i, j).size() << std::endl;
      //std::cout << input.getMatrixByIndex(i, j).size() << std::endl;
      output.setMatrixByIndex(i, j, input.getMatrixByIndex(i, j).cwiseProduct(
        values.getMatrixByIndex(i, j).cwiseProduct(
        values.getMatrixByIndex(i, j).unaryExpr([](double values) {
        return 1.0 - values;
      }))));
    }
  }
}
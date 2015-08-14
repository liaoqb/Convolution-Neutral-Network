#include"Connection.h"
#include <iostream>

Connection::Connection() {
  this ->batch = 0;
  this ->in = 0;
  this ->out = 0;
}

  // column vector, channel is output
Connection::Connection(int batch, int in, int out, double rate) {
  this ->batch = batch;
  this ->in = in;
  this ->out = out;
  this ->rate = rate;

  this ->weights.initialize(batch, out, in, 1 , "random");
  this ->thresholds.initialize(batch, out, 1, 1, "random");
}

  // channel = output, row = input, the batch is always 1, take care of the weights
void Connection::setBatch(int batch) {
  this ->batch = batch;
}

void Connection::forwardPropagation(const Blob& input, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      double value = 0.0;
      //std::cout << input.getChannel() << ' ' << weights.getRow() << std::endl;
      for (int k = 0; k < weights.getRow(); ++k) {
        value += input.getValueByIndex(i, k, 0, 0) * weights.getValueByIndex(0, j, k, 0);
      }
      //std::cout << output.getValueByIndex(i, j, 0, 0) << std::endl;
      output.setValueByIndex(i, j, 0, 0, value - thresholds.getValueByIndex(0, j, 0, 0));
    }
  }
}
  
  // input is the error, values is the value, I am not quite sure about this function
void Connection::backwardPropagation(const Blob& input, const Blob& values, Blob& output) {
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      double value = 0.0;

      for (int k = 0; k < weights.getChannel(); ++k) {
        value += input.getValueByIndex(i, k, 0, 0) * weights.getValueByIndex(0, k, j, 0);
      }

      output.setValueByIndex(i, j, 0, 0, value);
    }
  }

    // update weights and thresholds
  MatrixXd updateError = MatrixXd::Zero(1, input.getChannel());
  MatrixXd updateWeight = MatrixXd::Zero(weights.getChannel(), weights.getRow());

  for (int i = 0; i < input.getBatch(); ++i) {
    for (int j = 0; j < input.getChannel(); ++j) {
      //std::cout << input.getValueByIndex(i, j, 0, 0) << std::endl;
      updateError(0, j) += input.getValueByIndex(i, j, 0, 0);

      for (int k = 0; k < weights.getRow(); ++k) {
        //std::cout << i << ' ' << j << ' ' << k << std::endl;
        //std::cout << updateWeight.size() << std::endl;
        //std::cout << values.getMatrixByIndex(i, k).size() << std::endl;
        //std::cout << input.getMatrixByIndex(i, j).size() << std::endl;
        updateWeight(j, k) += values.getValueByIndex(i, k, 0, 0) * input.getValueByIndex(i, j, 0, 0);
      }
    }
  }

  for (int i = 0; i < input.getChannel(); ++i) {
    //std::cout << thresholds.getValueByIndex(0, i, 0, 0) << std::endl;
    thresholds.setValueByIndex(0, i, 0, 0, thresholds.getValueByIndex(0, i, 0, 0) + rate * updateError(0, i) / input.getBatch());
    //std::cout << thresholds.getValueByIndex(0, i, 0, 0) - thresholds.getValueByIndex(0, i, 0, 0)<< std::endl;

    for (int j = 0; j < weights.getRow(); ++j) {
      weights.setValueByIndex(0, i, j, 0, weights.getValueByIndex(0, i, j, 0) + rate * updateWeight(i, j) / input.getBatch());
    }
    //std::cout << weights.getMatrixByIndex(0, i) << std::endl;
  }

}
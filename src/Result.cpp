#include"Result.h"
#include <iostream>
#include <cmath>

  // input is the result, values is the output wanted
void Result::backwardPropagation(const Blob& input, const Blob& values, Blob& output) {
  error = 0.0;
  for (int i = 0; i < output.getBatch(); ++i) {
    for (int j = 0; j < output.getChannel(); ++j) {
      output.setValueByIndex(i, j, 0, 0, (values.getValueByIndex(i, j, 0, 0) - input.getValueByIndex(i, j, 0, 0)) *
        input.getValueByIndex(i, j, 0, 0) * (1.0 - input.getValueByIndex(i, j, 0, 0)));
      //std::cout << input.getValueByIndex(i, j, 0, 0) << std::endl;
      //std::cout << j  << ' ' << (values.getValueByIndex(i, j, 0, 0) - input.getValueByIndex(i, j, 0, 0)) *
      //  input.getValueByIndex(i, j, 0, 0) * (1.0 - input.getValueByIndex(i, j, 0, 0)) << std::endl;
      //std::cout << values.getValueByIndex(i, j, 0, 0) << std::endl;
      //std::cout << input.getValueByIndex(i, j, 0, 0) << std::endl;
      //std::cout << "\n\n\n";
      //std::cout << values.getValueByIndex(i, j, 0, 0) << ' ' << input.getValueByIndex(i, j, 0, 0) << ' ' << j << std::endl;
      error += pow((values.getValueByIndex(i, j, 0, 0) - input.getValueByIndex(i, j, 0, 0)), 2);
    }
    //std::cout << std::endl;
  }

  //std::cout << error / 2.0 << std::endl;
}
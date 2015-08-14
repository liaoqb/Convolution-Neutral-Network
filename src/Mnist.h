// <Copyright liaoqb>  [Copyright 2015.07.08]

#ifndef MNIST_H_
#define MNIST_H_

#include"Blob.h"
#include <string>
#include <Eigen/Eigen>

using namespace Eigen;

class Mnist {
public:
  Mnist(std::string imageFileName, std::string labelFileName) :
    imageFileName(imageFileName), labelFileName(labelFileName) {
    readData();
  }

  ~Mnist() {}
  Blob& getImages() {return images;}
  Blob& getLabels() {return labels;}

private:
  std::string imageFileName;
  std::string labelFileName;
  void readData();
  int reverseToInt(int number);

  Blob images;
  Blob labels;
};

#endif
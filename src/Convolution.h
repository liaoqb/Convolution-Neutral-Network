#ifndef CONCOLUTION_H
#define CONCOLUTION_H

#include"Blob.h"
#include"Layer.h"
#include <Eigen/Eigen>

using namespace Eigen;

class Convolution : public Layer {
public:
  Convolution();
  Convolution(int batch, int channel, int row, int col, double rate = 0.01);
  ~Convolution() {}

  void setBatch(int batch);
  void setRate(double rate) {this ->rate = rate;}

  int getBatch() const {return batch;}
  int getChannel() const {return channel;}
  int getRow() const {return row;}
  int getCol() const {return col;}
  double getRate() const {return rate;}
  Blob& getKernels() {return kernels;}
  Blob& getBias() {return bias;}

  std::string getName() {return "Convolution";}

  void forwardPropagation(const Blob& input, Blob& output);
  void backwardPropagation(const Blob& input, const Blob& values, Blob& output);

  MatrixXd convolution(const MatrixXd& matrix, const MatrixXd& kernel, std::string way);
  MatrixXd correlation(const MatrixXd& matrix, const MatrixXd& kernel, std::string way);

private:
  double rate;
  int batch;
  int channel;
  int row;
  int col;
  Blob kernels;
  Blob bias;

  MatrixXd spanMatrix(const MatrixXd& matrix, int spanRows, int spanCols);
  MatrixXd rotateMatrix(const MatrixXd& matrix);
};

#endif
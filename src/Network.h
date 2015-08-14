#ifndef NETWORK_H
#define NETWORK_H

#include"Blob.h"
#include"Convolution.h"
#include"Connection.h"
#include"Dropout.h"
#include"Flatten.h"
#include"Mnist.h"
#include"Result.h"
#include <vector>
#include <string>
#include <Eigen/Eigen>

using namespace Eigen;

class Network {
public:
  Network(std::string imageName, std::string labelName);
  ~Network();

  std::vector<Layer*>& getLayers() {return layers;}
  std::vector<Blob*>& getOutputs() {return outputs;}
  std::vector<Blob*>& getErrors() {return errors;}

  void train(int times = 5, double rate = 0.01, int batch = 1, std::string in = "result.txt", std::string out = "result.txt");
  void predicate(const Blob& images, const Blob& labels);

private:
  std::vector<Layer*> layers;
  std::vector<Blob*> outputs;
  std::vector<Blob*> errors;
  Mnist* mnist;

  void readData(std::string in);
  void saveData(std::string out);
};
 
#endif
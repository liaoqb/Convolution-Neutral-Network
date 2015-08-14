#include <iostream>
#include"Blob.h"
#include"Convolution.h"
#include"Dropout.h"
#include"Activation.h"
#include"Flatten.h"
#include"Connection.h"
#include"Result.h"
#include"Network.h"
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
  int train = 1;
  double rate = 0.0002;
  int batch = 1;

  for (int i = 1; i < argc; ++i) {
    string t = "-t";
    string r = "-r";
    string b = "-b";

    if (strcmp(argv[i], t.c_str()) == 0) {
      train = atoi(argv[i + 1]);
    }

    if (strcmp(argv[i], r.c_str()) == 0) {
      rate = atof(argv[i + 1]);
    }

    if (strcmp(argv[i], b.c_str()) == 0) {
      batch = atoi(argv[i + 1]);
    }
  }

  Network network("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

  network.train(train, rate, batch, "data/result.txt", "data/result.txt");

  Mnist* mnist = new Mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

  network.predicate(mnist ->getImages(), mnist ->getLabels());
           
  delete mnist;

  return 0;
}
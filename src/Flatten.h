#ifndef FLATTEN_H
#define FLATTEN_H

#include"Layer.h"

class Flatten : public Layer {
public:
  Flatten() {}
  ~Flatten() {}

  std::string getName() {return "Flatten";}

  void forwardPropagation(const Blob& input, Blob& output);
  void backwardPropagation(const Blob& input, const Blob& values, Blob& output);
};

#endif
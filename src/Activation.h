#ifndef ACTIVATION_H
#define ACTIVATION_H

#include"Layer.h"

class Activation : public Layer {
public:
  Activation() {}
  ~Activation() {}         

  void forwardPropagation(const Blob& input, Blob& output);
  void backwardPropagation(const Blob& input, const Blob& values, Blob& output);

  std::string getName() {return "Activation";}
};

#endif
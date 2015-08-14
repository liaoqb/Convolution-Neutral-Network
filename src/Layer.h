#ifndef LAYER_H
#define LAYER_H

#include"Blob.h"

class Layer {
public:
  Layer() {};
  virtual void forwardPropagation(const Blob& input, Blob& output) = 0;
  virtual void backwardPropagation(const Blob& input, const Blob& values, Blob& output) = 0;
  virtual std::string getName() = 0;
  virtual ~Layer() {}
};

#endif
#ifndef RESULT_H
#define RESULT_H

#include"Activation.h"

class Result : public Activation {
public:
  Result() {}
  ~Result() {}

  void backwardPropagation(const Blob& input, const Blob& values, Blob& output);

  double getError() const {return error;}

  std::string getName() {return "Result";}

private:
  double error;
};

#endif
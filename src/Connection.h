#ifndef CONNECTION_H
#define CONNECTION_H

#include"Layer.h"

class Connection : public Layer {
public:
  Connection();
  Connection(int batch, int in, int out, double rate = 0.01);
  ~Connection() {}

  void setBatch(int batch);
  void setRate(double rate) {this ->rate = rate;}

  void forwardPropagation(const Blob& input, Blob& output);
  void backwardPropagation(const Blob& input, const Blob& values, Blob& output);

  int getBatch() const {return batch;}
  int getIn() const {return in;}
  int getOut() const {return out;}
  double getRate() const {return rate;}
  Blob& getWeights() {return weights;}
  Blob& getThresholds() {return thresholds;}

  std::string getName() {return "Connection";}

private:
  double rate;
  int batch;
  int in;
  int out;
  Blob weights;
  Blob thresholds;
};

#endif
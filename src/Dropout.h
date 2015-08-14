#ifndef DROPOUT_H
#define DROPOUT_H

#include"Layer.h"
#include <iostream>

class Dropout : public Layer {
public:
  Dropout();
  Dropout(int batch, int channel, int row, int col, int boxHeight, int boxWidth, int step);
  ~Dropout() {}

  void setBatch(int batch);

  int getRow() const {return row;}
  int getCol() const {return col;}
  int getBoxHeight() const {return boxHeight;}
  int getBoxWidth() const {return boxWidth;}
  int getStep() const {return step;}
  int getBatch() const {return batch;}

  std::string getName() {return "Dropout";}

  void forwardPropagation(const Blob& input, Blob& output);
  void backwardPropagation(const Blob& input, const Blob& values, Blob& output);

private:
  int batch;
  int channel;
  int row;
  int col;
  int boxHeight;
  int boxWidth;
  int step;
};

#endif
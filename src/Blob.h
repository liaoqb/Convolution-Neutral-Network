#ifndef BLOB_H
#define BLOB_H

#include <Eigen/Eigen>
#include <string>

using namespace Eigen;

class Blob {
public:
  Blob();
  Blob(int batch, int channel, int row, int col, std::string init);
  ~Blob();

  void initialize(int batch, int channel, int row, int col, std::string init);

  MatrixXd& getMatrixByIndex(int i, int j) const;
  double& getValueByIndex(int i, int j, int x, int y) const;
  void setMatrixByIndex(int i, int j, const MatrixXd& matrix);
  void setValueByIndex(int i, int j, int x, int y, const double& value);

  int getBatch() const {return batch;}
  int getChannel() const {return channel;}
  int getRow() const {return row;}
  int getCol() const {return col;}

private:
  int batch;
  int channel;
  int row;
  int col;
  MatrixXd* matrixPtr;
};

#endif
#include "Mnist.h"
#include <fstream>
#include <iostream>

int Mnist::reverseToInt(int number) {
  unsigned char c1 = number & 0xff;
  unsigned char c2 = (number >> 8) & 0xff;
  unsigned char c3 = (number >> 16) & 0xff;
  unsigned char c4 = (number >> 24) & 0xff;

  return int(c1 << 24) + int(c2 << 16) + int(c3 << 8) + int(c4);
}

void Mnist::readData() {
  std::ifstream imageFile(imageFileName, std::ifstream::binary);
  std::ifstream labelFile(labelFileName, std::ifstream::binary);

  if (imageFile.is_open()) {
    int magicNumber;
    int numberOfImage;
    int row;
    int col;
    int nRows;
    int nCols;

    imageFile.read((char*)&magicNumber, sizeof(magicNumber));
    imageFile.read((char*)&numberOfImage, sizeof(numberOfImage));
    row = reverseToInt(numberOfImage);
    imageFile.read((char*)&nRows, sizeof(nRows));
    nRows = reverseToInt(nRows);
    imageFile.read((char*)&nCols, sizeof(nCols));
    nCols = reverseToInt(nCols);

    // take care of this
    //row = 6;

    images.initialize(row, 1, nRows, nCols, "zero");

    // read all the image data file and normalization
    for (int k = 0; k < row; ++k) {
      for (int i = 0; i < nRows; ++i) {
        for (int j = 0 ; j < nCols; ++j) {
          unsigned char temp = 0;
          imageFile.read((char*)&temp, sizeof(temp));
          images.setValueByIndex(k, 0, i, j, (temp == 0 ? 0 : 1));
          //std::cout << images(k, i * nRows + j) << std::endl;
        }
      }
    }

    imageFile.close();
  }

  if (labelFile.is_open()) {
    int magicNumber;
    int items;

    labelFile.read((char*)&magicNumber, sizeof(magicNumber));
    labelFile.read((char*)&items, sizeof(items));

    items = reverseToInt(items);

    // take care of this
    //items = 6;

    labels.initialize(items, 1, 1, 1, "zero");

    for (int i = 0; i < items; ++i) {
      unsigned char temp = 0;
      labelFile.read((char*)&temp, sizeof(temp));
      labels.setValueByIndex(i, 0, 0, 0, temp);
    }

    labelFile.close();
  }
}
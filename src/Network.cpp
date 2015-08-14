#include"Network.h"
#include <fstream>
#include <iostream>

Network::Network(std::string imageName, std::string labelName) {
  mnist = new Mnist(imageName, labelName);
}

Network::~Network() {

  for (int i = 0; i < layers.size(); ++i) {
    delete layers[i];
    delete outputs[i];
    delete errors[i];
  }

  delete mnist;

}

void Network::train(int times, double rate, int batch, std::string in, std::string out) {
  layers.push_back(new Convolution(1, 6, 5, 5, rate));
  outputs.push_back(new Blob(batch, 6, 24, 24, "zero"));
  errors.push_back(new Blob(batch, 1, 28, 28, "zero"));

  layers.push_back(new Dropout(batch, 6, 12, 12, 2, 2, 2));
  outputs.push_back(new Blob(batch, 6, 12, 12, "zero"));
  errors.push_back(new Blob(batch, 6, 24, 24, "zero"));
  /*
  layers.push_back(new Activation());
  outputs.push_back(new Blob(batch, 20, 12, 12, "zero"));
  errors.push_back(new Blob(batch, 20, 12, 12, "zero"));
  */
  layers.push_back(new Convolution(1, 15, 5, 5, rate));
  outputs.push_back(new Blob(batch, 15, 8, 8, "zero"));
  errors.push_back(new Blob(batch, 6, 12, 12, "zero"));

  layers.push_back(new Dropout(batch, 15, 4, 4, 2, 2, 2));
  outputs.push_back(new Blob(batch, 15, 4, 4, "zero"));
  errors.push_back(new Blob(batch, 15, 8, 8, "zero"));
  /*
  layers.push_back(new Activation());
  outputs.push_back(new Blob(batch, 50, 4, 4, "zero"));
  errors.push_back(new Blob(batch, 50, 4, 4, "zero"));
  */
  layers.push_back(new Flatten());
  outputs.push_back(new Blob(batch, 240, 1, 1, "zero"));
  errors.push_back(new Blob(batch, 15, 4, 4, "zero"));

  layers.push_back(new Connection(1, 240, 75));
  outputs.push_back(new Blob(batch, 75, 1, 1, "zero"));
  errors.push_back(new Blob(batch, 240, 1, 1, "zero"));

  layers.push_back(new Activation());
  outputs.push_back(new Blob(batch, 75, 1, 1, "zero"));
  errors.push_back(new Blob(batch, 75, 1, 1, "zero"));

  layers.push_back(new Connection(1, 75, 10));
  outputs.push_back(new Blob(batch, 10, 1, 1, "zero"));
  errors.push_back(new Blob(batch, 75, 1, 1, "zero"));

  layers.push_back(new Result());
  outputs.push_back(new Blob(batch, 10, 1, 1, "zero"));
  errors.push_back(new Blob(batch, 10, 1, 1, "zero"));

  readData(in);
  //std::cout << (*(Convolution*)layers[0]).getKernels().getMatrixByIndex(0, 0) << std::endl;
  //std::cout << (*(Convolution*)layers[0]).getKernels().getMatrixByIndex(0, 1) << std::endl;
  for (int z = 0; z < times; ++z) {
    double error = 0.0;
    for (int k = 0; k < mnist ->getImages().getBatch() / batch; ++k) {
      Blob batchImage(batch, 1, 28, 28, "zero");
      Blob batchLabel(batch, 10, 1, 1, "zero");

      for (int i = k * batch; i < (k + 1) * batch; ++i) {
        //for (int z = 0; z < 784; ++z)
          batchImage.setMatrixByIndex(i - k * batch, 0, mnist ->getImages().getMatrixByIndex(i, 0));
        batchLabel.setValueByIndex(i - k * batch, mnist ->getLabels().getValueByIndex(i, 0, 0, 0), 0, 0, 1);
      }

      for (int i = 0; i < layers.size(); ++i) {
        if (i == 0) {
          layers[i] ->forwardPropagation(batchImage, *outputs[0]);
          //std::cout << outputs[0] ->getMatrixByIndex(0, 0) << std::endl;
        } else {
          layers[i] ->forwardPropagation(*outputs[i - 1], *outputs[i]);
        }
      }

      for (int i = layers.size() - 1; i >= 0; --i) {
        if (i == layers.size() - 1) {
          layers[i] ->backwardPropagation(*outputs[layers.size() - 1], batchLabel,
            *errors[layers.size() - 1]);
          //std::cout << errors[layers.size() - 1] ->getMatrixByIndex(0, 0) << std::endl;
        } else if (i > 0) {
          //std::cout << errors[layers.size() - 1] ->getMatrixByIndex(0, 0) << std::endl;
          //layers[i] ->backwardPropagation(*errors[i + 1], *outputs[i - 1], *errors[i]);

          if (i == 6) {
            layers[i] ->backwardPropagation(*errors[i + 1], *outputs[i], *errors[i]);
          } else {
            layers[i] ->backwardPropagation(*errors[i + 1], *outputs[i - 1], *errors[i]);
          }

        } else {
          layers[i] ->backwardPropagation(*errors[i + 1], batchImage, *errors[i]);
        }
      }

      error += (*(Result*)(layers[layers.size() - 1])).getError();
    }

    std::cout << "all error: " << error / 2.0 << std::endl;
  }

  saveData(out);
}

void Network::readData(std::string in) {
  std::ifstream fin;

  fin.open(in, std::ios::in);

  if (fin.is_open()) {
	for (int i = 0; i < layers.size(); ++i) {
	  if (layers[i] ->getName() == "Convolution") {
		Blob& kernels = (*(Convolution*)layers[i]).getKernels();
		Blob& bias = (*(Convolution*)layers[i]).getBias();

		double temp;
		
		for (int x = 0; x < kernels.getChannel(); ++x) {
		  for (int y = 0; y < kernels.getRow(); ++y) {
			for (int z = 0; z < kernels.getCol(); ++z) {
			  fin >> temp;
			  kernels.setValueByIndex(0, x, y, z, temp);
			}
		  }
		}

		for (int x = 0; x < bias.getChannel(); ++x) {
		  fin >> temp;

		  bias.setValueByIndex(0, x, 0, 0, temp);
		}
	  } else if (layers[i] ->getName() == "Connection") {
		Blob& weights = (*(Connection*)layers[i]).getWeights();
		Blob& thresholds = (*(Connection*)layers[i]).getThresholds();
		double temp;

		for (int x = 0; x < weights.getChannel(); ++x) {
		  for (int y = 0; y < weights.getRow(); ++y) {
			fin >> temp;
			weights.setValueByIndex(0, x, y, 0, temp);
		  }
		}

		for (int x = 0; x < thresholds.getChannel(); ++x) {
		  fin >> temp;

		  thresholds.setValueByIndex(0, x, 0, 0, temp);
		}
	  }
	}

	fin.close();

	std::cout << "Read data success!\n";
  } else {
    std::cout << "Read data failed!\n";
  }
}

void Network::saveData(std::string out) {
  std::ofstream fout;

  fout.open(out, std::ios::out);

  if (fout.is_open()) {
	for (int i = 0; i < layers.size(); ++i) {
	  if (layers[i] ->getName() == "Convolution") {
		Blob& kernels = (*(Convolution*)layers[i]).getKernels();
		Blob& bias = (*(Convolution*)layers[i]).getBias();

		for (int x = 0; x < kernels.getChannel(); ++x) {
		  for (int y = 0; y < kernels.getRow(); ++y) {
			for (int z = 0; z < kernels.getCol(); ++z) {
			  if (z) {
			    fout << ' ' << kernels.getValueByIndex(0, x, y, z);
			  } else {
			    fout << kernels.getValueByIndex(0, x, y, z);
			  }
			}
			fout << std::endl;
		  }
		}

		for (int x = 0; x < bias.getChannel(); ++x) {
		  if (x) {
		    fout << ' ' << bias.getValueByIndex(0, x, 0, 0);
		  } else {
		    fout << bias.getValueByIndex(0, x, 0, 0);
		  }
		}

		fout << std::endl;
	  } else if (layers[i] ->getName() == "Connection") {
		Blob& weights = (*(Connection*)layers[i]).getWeights();
		Blob& thresholds = (*(Connection*)layers[i]).getThresholds();

		for (int x = 0; x < weights.getChannel(); ++x) {
		  for (int y = 0; y < weights.getRow(); ++y) {
			if (y) {
			  fout << ' ' << weights.getValueByIndex(0, x, y, 0);
			} else {
			  fout << weights.getValueByIndex(0, x, y, 0);
			}
		  }
		  fout << std::endl;
		}

		for (int x = 0; x < thresholds.getChannel(); ++x) {
		  if (x) {
		    fout << ' ' << thresholds.getValueByIndex(0, x, 0, 0);
		  } else {
		    fout << thresholds.getValueByIndex(0, x, 0, 0);
		  }
		}

		fout << std::endl;
	  }
	}

	fout.close();

	std::cout << "Save data success!\n";
  } else {
    std::cout << "Save data fialed!\n";
  }
}

void Network::predicate(const Blob& images, const Blob& labels) {
  int batch = (*(Dropout*)layers[1]).getBatch();
  int count = 0;

  std::ofstream fout;

  fout.open("data/label.txt", std::ios::out);

  //std::cout << batch << std::endl;

  for (int i = 0; i < labels.getBatch() / batch; ++i) {
    Blob batchImage(batch, 1, 28, 28, "zero");
    Blob batchLabel(batch, 1, 1, 1, "zero");

    for (int k = i * batch; k < (i + 1) * batch; ++k) {
      //for (int z = 0; z < 784; ++z)
        batchImage.setMatrixByIndex(k - i * batch, 0, images.getMatrixByIndex(k, 0));
      //std::cout << images.getMatrixByIndex(k, 0) << std::endl;
      batchLabel.setValueByIndex(k - i * batch, 0, 0, 0, labels.getValueByIndex(k, 0, 0, 0));
      //std::cout << labels.getValueByIndex(k, 0, 0, 0) << std::endl;
    }

    for (int j = 0; j < layers.size(); ++j) {
      if (j == 0) {
        layers[j] ->forwardPropagation(batchImage, *outputs[0]);
        //std::cout << outputs[0] ->getMatrixByIndex(0, 0) << std::endl;
      } else {
        layers[j] ->forwardPropagation(*outputs[j - 1], *outputs[j]);
        //std::cout << outputs[j] ->getMatrixByIndex(0, 0) << std::endl;
      }
    }
    /*
    std::cout << "Result:" << std::endl;
    for (int j = 0; j < 10; ++j) {
      std::cout << outputs[3] ->getMatrixByIndex(0, j) - outputs[3] ->getMatrixByIndex(1, j) << std::endl;
    }
    std::cout << "Result:" << std::endl;
    */
    for (int j = 0; j < batch; ++j) {
      int index = 0;
      double value = outputs[layers.size() - 1] ->getValueByIndex(j, 0, 0, 0);

      for (int k = 0; k < 10; ++k) {
        //std::cout << outputs[layers.size() - 1] ->getValueByIndex(j, k, 0, 0) << std::endl;
        if (outputs[layers.size() - 1] ->getValueByIndex(j, k, 0, 0) > value) {
          value = outputs[layers.size() - 1] ->getValueByIndex(j, k, 0, 0);
          index = k;
        }
      }

      //std::cout << index << ' ' << batchLabel.getValueByIndex(j, 0, 0, 0) << std::endl;

	  fout << "predicate " << index << ' ' << "real " << batchLabel.getValueByIndex(j, 0, 0, 0) << std::endl;

      if (index == batchLabel.getValueByIndex(j, 0, 0, 0)) {
        count++;
      }
    }
  }

  std::cout << "Rate: " << count / double(labels.getBatch()) << std::endl;

  fout.close();
}
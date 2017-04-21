//
// Created by ken.yu on 17-3-27.
//

#include <iostream>
#include <getopt.h>
#include <opencv2/opencv.hpp>
#include <DeepMARCaffe.hpp>
#include <omp.h>

using namespace cripac;
using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  int gpuID = -1;
  char *proto_path = NULL;
  char *model_path = NULL;
  char *img_path = NULL;

  int opt;

  while ((opt = getopt(argc, argv, "g:p:m:i:")) != -1)
    switch (opt) {
      case 'g':gpuID = atoi(optarg);
        break;
      case 'p':proto_path = optarg;
        break;
      case 'm':model_path = optarg;
        break;
      case 'i':img_path = optarg;
        break;
      case '?':
        switch (optopt) {
          case 'g':
          case 'p':
          case 'm':
          case 'i':fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            break;
          default:
            if (isprint(optopt))
              fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
              fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            break;
        }
      default:abort();
    }

  DeepMAR *recognizer = new DeepMAR();
  recognizer->initialize(proto_path,
                         model_path,
                         gpuID);

  const int INPUT_SIZE = 227;

  Mat img = imread(img_path);
  if (img.empty()) {
    fprintf(stderr, "Cannot find image for test at \"%s\"", img_path);
    return -1;
  }
  resize(img, img, Size(INPUT_SIZE, INPUT_SIZE));
  img.convertTo(img, CV_32FC3);

  Mat channels[3];
  split(img, channels);
  float input[INPUT_SIZE * INPUT_SIZE * 3];
  for (int i = 0; i < 3; ++i)
    memmove(input + i * INPUT_SIZE * INPUT_SIZE, channels[i].data, sizeof(float) * INPUT_SIZE * INPUT_SIZE);

#pragma omp parallel for
  for (int i = 0; i < INPUT_SIZE * INPUT_SIZE * 3; ++i)
    input[i] = (input[i] - 128) / 256.f;

  const float* fc8;

  double start = omp_get_wtime();
  for (int i = 0; i < 100; ++i) fc8 = recognizer->recognize(input);
  double end = omp_get_wtime();

  for (int i = 0; i < 1024; ++i)
    cout << fc8[i] << ' ';
  cout << endl;

  cout << (end - start) << "ms per round." << endl;

  return 0;
}
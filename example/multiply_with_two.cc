#include <iostream>

#include <tensorflow/c/c_api.h>
#include <tensorflow_cc_inference/Inference.h>

using tensorflow_cc_inference::Inference;

int main(int argc, char** argv) {

  // instanciate the library
  auto CNN = Inference("../test/graphs/x_times_two.pb", "x", "y");

  // create an input tensor
  int64_t dims[] = {1, };
  TF_Tensor* in  = TF_AllocateTensor(TF_FLOAT, dims, 1, 1*sizeof(float));
  float* in_data = (float*)(TF_TensorData(in));
  in_data[0] = 3.14;

  // run the tensor through the graph
  TF_Tensor* out = CNN(in);

  // get and display the data from the output tensor
  float* out_data = (float*)(TF_TensorData(out));
  std::cout << "3.14 * 2 = " << out_data[0] << std::endl;
  TF_DeleteTensor(in);
  TF_DeleteTensor(out);
}

// To compile and run this file:
// create a build directory in the repository root
//   git clone git@github.com:adriankoering/tensorflow-cc-inference.git
//   cd tensorflow-cc-inference && mkdir build && cd build
//   cmake .. && make
// then run the build example via
//   ./multiply_with_two

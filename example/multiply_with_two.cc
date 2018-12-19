#include <iostream>

#include <tensorflow/c/c_api.h>
#include <tensorflow_cc_inference/Inference.h>

using tensorflow_cc_inference::Inference;
using tensorflow_cc_inference::Tensor;

void TestDeallocator(void* /*data*/, size_t /*len*/, void* /*arg*/) {
	std::cout << "deallocating data" << '\n';
}

int main(int /*argc*/, char** /*argv*/) {

  // instanciate the library
  auto CNN = Inference("../test/graphs/x_times_two.pb", "x", "y");

  // Raw TF_Tensor example:

  // create an input tensor
  int64_t dims[] = {1, };
  // test raw TF_Tensor* interface
  TF_Tensor* in  = TF_AllocateTensor(TF_FLOAT, dims, 1, 1*sizeof(float));
  float* in_data = (float*)(TF_TensorData(in));
  in_data[0] = 3.14f;

  // run the tensor through the graph
  TF_Tensor* out = CNN(in);

  // get and display the data from the output tensor
  float* out_data = (float*)(TF_TensorData(out));
  std::cout << "3.14 * 2 = " << out_data[0] << std::endl;
  TF_DeleteTensor(in);
  TF_DeleteTensor(out);



  // C++ Tensor<float> wrapper example: 

  std::cout << "Tensor<float>(dims2, 3) example:" << std::endl;
  int64_t dims2[] = {2,2,2};
  for (int i =0; i<5;++i)
  {
	  auto in = Tensor<float>(dims2, 3);
	  for(int j = 0; j < 8; j++)
		in.Data()[j] = 3.14f+j;
	  auto out = CNN.Run<float>(in);
	  for (int j = 0; j < 8; j++)
		  std::cout << "(3.14+i) * 2 = " << out.Data()[j] << std::endl;
  }



  // C++ Tensor<float> wrapper example with initialization: 

  std::cout << "Tensor<float>(dims2, 3, pi_data ...) example:" << std::endl;
  for (int i = 0; i < 5; ++i)
  {
	  float pi_data[] = {3.14f,13.14f,23.14f,33.14f,43.14f,53.14f,63.14f,73.14f};
	  auto in = Tensor<float>(dims2, 3, pi_data, NULL /* or &TestDeallocator */, NULL);
	  auto out = CNN.Run<float>(in);
	  for (int j = 0; j < 8; j++)
		  std::cout << "(3.14+10*i) * 2 = " << out.Data()[j] << std::endl;
  }



  // C++ Tensor<char> wrapper example with initialization: 

  std::cout << "Tensor<uint8_t>(dims3, 2, pi_data ...) example:" << std::endl;
  auto CNN_uint8 = Inference("../test/graphs/x_times_two_uint8.pb", "x", "y");
  int64_t dims3[] = {2,3};
  for (int i = 0; i < 5; ++i)
  {
	  uint8_t pi_data[] = {3,4,5,6,7,8};
	  auto in = Tensor<uint8_t>(dims3, 2, pi_data, NULL /* or &TestDeallocator */, NULL);
	  auto out = CNN_uint8.Run<uint8_t>(in);
	  for (int j = 0; j < 6; j++)
		  std::cout << "(3+i) * 2 = " << (int)(out.Data()[j]) << std::endl;
  }
}

// To compile and run this file:
// create a build directory in the repository root
//   git clone git@github.com:adriankoering/tensorflow-cc-inference.git
//   cd tensorflow-cc-inference && mkdir build && cd build
//   cmake .. && make
// then run the build example via
//   ./multiply_with_two

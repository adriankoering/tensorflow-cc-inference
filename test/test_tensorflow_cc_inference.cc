#include <stdint.h>
#include "tensorflow_cc_inference/Inference.h"
#include <gtest/gtest.h>
#include <string>
using tensorflow_cc_inference::Inference;
using tensorflow_cc_inference::Tensor;

TEST(TestSimplegraph, DoesItRun) {

  auto M = Inference("test/graphs/x_times_two.pb", "x", "y");

  int64_t dims[1] = {1l, };
  TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT,
                                              dims, 1,
                                              sizeof(float));
  static_cast<float*>(TF_TensorData(input_tensor))[0] = 1.f;

  TF_Tensor* out = M(input_tensor);

  ASSERT_EQ(TF_TensorType(out), TF_FLOAT);
  ASSERT_EQ(TF_NumDims(out), 1);
  ASSERT_EQ(TF_Dim(out, 0), 1);

  float* out_data = static_cast<float*>(TF_TensorData(out));

  ASSERT_DOUBLE_EQ(out_data[0], 2);
}

TEST(TestSimplegraph, CppFloatNoInit)
{
	auto M = Inference("test/graphs/x_times_two.pb", "x", "y");

	int64_t dims[3] = {2,2,2};
	auto in = Tensor<float>(dims, 3);
	for (int i = 0; i < 8; i++)
		in.Data()[i] = 3.14f + i;
	auto out = M.Run<float>(in);

	auto out_shape = out.Shape();

	for (int i = 0; i < 8; ++i)
		ASSERT_DOUBLE_EQ(out.Data()[i], 2 * (3.14f + i));

	ASSERT_EQ(out_shape.size(), 3);
	ASSERT_EQ(out_shape[0], 2);
	ASSERT_EQ(out_shape[1], 2);
	ASSERT_EQ(out_shape[2], 2);
}

TEST(TestSimplegraph, CppFloatWithInit)
{
	auto M = Inference("test/graphs/x_times_two.pb", "x", "y");

	int64_t dims[2] = {2,3};
	float pi_data[] = {3.14f,13.14f,23.14f,33.14f,43.14f,53.14f};

	auto in = Tensor<float>(dims, 2, pi_data, NULL, NULL);
	auto out = M.Run<float>(in);

	for (int i = 0; i < 6; ++i)
		ASSERT_DOUBLE_EQ(out.Data()[i], 2 * (3.14f + i*10));
}

TEST(TestSimplegraph, CppTypeCheck)
{
	auto M = Inference("test/graphs/x_times_two.pb", "x", "y");

	int64_t dims[1] = {2};
	float pi_data[] = {3,4};

	auto in = Tensor<float>(dims, 1, pi_data, NULL, NULL);
	ASSERT_THROW(
		auto out = M.Run<double>(in), // incorrect output type
		std::runtime_error
	);
}

int main(int argc, char** argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

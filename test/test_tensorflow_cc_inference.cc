
#include "tensorflow_inference/tensorflow_inference.h"
#include <gtest/gtest.h>

using tensorflow_inference::TensorflowInference;

TEST(TestOpenFile, DoesItRun) {
  auto M = TensorflowInference("../test/graphs/x_times_2.pb", "x", "y");


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

int main(int argc, char** argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

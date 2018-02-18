#include "tensorflow/c/c_api.h"
#include "opencv2/opencv.hpp"

void /*TF_Tensor*/ float_to_tensor(float* in)
{

}

void /*float**/ tensor_to_float(TF_Tensor** in)
{

}

TF_Tensor* image_to_tensor(const cv::Mat& in)
{
  int64_t dims[4] = {1, in.rows, in.cols, in.channels()};
  size_t num_elements = in.rows*in.cols*in.channels();

  TF_Tensor* out = TF_AllocateTensor(TF_FLOAT, dims, 4, sizeof(float)*num_elements);
  // assuming uint8 input image
  if(in.isContinuous())
  {
    float* data = (float*) TF_TensorData(out);
    for (size_t i = 0; i < num_elements; i++)
    {
      data[i] = in.data[i];
    }
  }
  else
  {
    throw std::runtime_error("Input Image must be Continuous");
  }
  return out;
}

cv::Mat tensor_to_image(TF_Tensor* in)
{
  float* data = (float*) TF_TensorData(in);

  cv::Mat out = cv::Mat(TF_Dim(in, 1), TF_Dim(in, 2), CV_8UC1);

  for (int i = 0, n = out.total(); i < n; ++i)
  {
    // saturated cast to uint8
    out.data[i] = cv::saturate_cast<uchar>(data[i]);
  }

  return out;
}

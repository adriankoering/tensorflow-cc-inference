#pragma once

#include <string>
#include <vector>
#include "tensorflow/c/c_api.h"

namespace tensorflow_cc_inference
{

/**
* Class providing type-specific access to TF_Tensor
* and auto deleting TF_Tensor in destructor
*/

template<typename T>
class Tensor
{
private:
	static void StubDeallocator(void* /*data*/, size_t /*len*/, void* /*arg*/) {}

	inline TF_DataType TFDataType();
	size_t CalcDataLen(const int64_t* dims, int num_dims)
	{
		size_t len = sizeof(T);
		for (int i = 0; i < num_dims; ++i)
			len *= dims[i];
		return len;
	}

	TF_Tensor* tensor;
public:
	// Creates Tensor that holds previously created TF_Tensor and deletes it in destructor
	Tensor(TF_Tensor* atensor)
	{
		if (TFDataType() != TF_TensorType(atensor))
			throw std::runtime_error("Inconsistent TF_Tensor* and Tensor<T> data types: " +
									std::to_string(TFDataType()) + " vs. " + std::to_string(TF_TensorType(atensor)));
		tensor = atensor;
	}

	// Creates tensor that holds data in it's own memory
	// Data should be filled using pointer returned by Data()
	Tensor(const int64_t* dims, int num_dims)
	{
		tensor = TF_AllocateTensor(TFDataType(), dims, num_dims, CalcDataLen(dims, num_dims));
	}

	// Creates tensor that holds external data pointed by data[0,len-1].
	// If deallocator is passed, it will be called when tensor is deallocated to 
	// deallocate underlying data.
	Tensor(const int64_t* dims, int num_dims, T* data,
		void(*adeallocator)(void* data, size_t len, void* arg),
		void* deallocator_arg)
	{
		auto deallocator = adeallocator ? adeallocator : &StubDeallocator;
		tensor = TF_NewTensor(TFDataType(), dims, num_dims, data, CalcDataLen(dims, num_dims),
			deallocator, deallocator_arg);
	}

	/**
	* Clean up all pointer-members using the dedicated tensorflor api functions
	*/
	~Tensor()
	{
		if (tensor)
			TF_DeleteTensor(tensor);
		tensor = nullptr;
	}

	TF_Tensor* TFTensor() { return tensor; }
	T* Data() { return (T*)(TF_TensorData(tensor)); }
	std::vector<int64_t> Shape()
	{
		int ndims = TF_NumDims(tensor);
		std::vector<int64_t> shape;
		for (int i = 0; i < ndims; ++i)
			shape.push_back(TF_Dim(tensor, i));
		return shape;
	}
};

template<> TF_DataType Tensor<float>::TFDataType() { return TF_FLOAT; }
template<> TF_DataType Tensor<double>::TFDataType() { return TF_DOUBLE; }
template<> TF_DataType Tensor<int32_t>::TFDataType() { return TF_INT32; }
template<> TF_DataType Tensor<uint8_t>::TFDataType() { return TF_UINT8; }
template<> TF_DataType Tensor<int16_t>::TFDataType() { return TF_INT16; }
template<> TF_DataType Tensor<int8_t>::TFDataType() { return TF_INT8; }
template<> TF_DataType Tensor<int64_t>::TFDataType() { return TF_INT64; }
template<> TF_DataType Tensor<uint64_t>::TFDataType() { return TF_UINT64; }
// TODO add other types when required


class Inference {

private:
	TF_Graph*   graph;
	TF_Session* session;

	TF_Operation* input_op;
	TF_Output 		input;

	TF_Operation* output_op;
	TF_Output 		output;

	/**
	 * Load a protobuf buffer from disk,
	 * recreate the tensorflow graph and
	 * provide it for inference.
	 */
	TF_Buffer* ReadBinaryProto(const std::string& fname) const;

	/**
	 * Tensorflow does not throw errors but manages runtime information
	 *   in a _Status_ object containing error codes and a failure message.
	 *
	 * AssertOk throws a runtime_error if Tensorflow communicates an
	 *   exceptional status.
	 *
	 */
	void AssertOk(const TF_Status* status) const ;

public:
	/**
	 * binary_graphdef_protobuf_filename: only binary protobuffers
	 *   seem to be supported via the tensorflow C api.
	 * input_node_name: the name of the node that should be feed with the
	 *   input tensor
	 * output_node_name: the node from which the output tensor should be
	 *   retrieved
	 */
	Inference(const std::string& binary_graphdef_protobuf_filename,
						const std::string& input_node_name,
						const std::string& output_node_name);

	/**
	 * Clean up all pointer-members using the dedicated tensorflor api functions
	 */
	~Inference();

	/**
	 * Run the graph on some input data.
	 *
	 * Provide the input and output tensor.
	 */
	 TF_Tensor* operator()(TF_Tensor* input_tensor) const;

	 template<typename OutputType, typename InputType>
	 Tensor<OutputType> Run(Tensor<InputType>& input_tensor) const
	 {
		 return Tensor<OutputType>((*this)(input_tensor.TFTensor()));
	 }
};

} // namespace tensorflow_cc_inference

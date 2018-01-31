#pragma once

#include <string>
#include "tensorflow/c/c_api.h"

namespace tensorflow_cc_inference{

class TensorflowCCInference {

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
	TensorflowCCInference(const std::string&,
												const std::string&,
												const std::string&);

	/**
	 * Clean up all pointer-members using the dedicated tensorflor api functions
	 */
	~TensorflowCCInference();

	/**
	 * Run the graph on some input data.
	 *
	 * Provide the input and output tensor.
	 */
	 TF_Tensor* operator()(TF_Tensor* input_tensor) const;

};

} // namespace tensorflow_cc_inference

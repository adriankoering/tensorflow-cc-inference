#include <fstream>
#include <sstream>
#include <exception>

#include "tensorflow_cc_inference/Inference.h"

using tensorflow_cc_inference::Inference;

/**
 * Read a binary protobuf (.pb) buffer into a TF_Buffer object.
 *
 * Non-binary protobuffers are not supported by the C api.
 * The caller is responsible for freeing the returned TF_Buffer.
 */
TF_Buffer* Inference::ReadBinaryProto(const std::string& fname) const
{
	std::ostringstream content;
	std::ifstream in(fname, std::ios::in | std::ios::binary); // | std::ios::binary ?

	if(!in.is_open())
	{
		throw std::runtime_error("Unable to open file: " + std::string(fname));
	}

	// convert the whole filebuffer into a string
	content << in.rdbuf();
	std::string data = content.str();

	return TF_NewBufferFromString(data.c_str(), data.length());
}

/**
 * Tensorflow does not throw errors but manages runtime information
 *   in a _Status_ object containing error codes and a failure message.
 *
 * AssertOk throws a runtime_error if Tensorflow communicates an
 *   exceptional status.
 *
 */
void Inference::AssertOk(const TF_Status* status) const
{
	if(TF_GetCode(status) != TF_OK)
	{
		throw std::runtime_error(TF_Message(status));
	}
}


/**
 * Load a protobuf buffer from disk,
 * recreate the tensorflow graph and
 * provide it for inference.
 */
Inference::Inference(
		const std::string& binary_graphdef_protobuffer_filename,
		const std::string& input_node_name,
 	  const std::string& output_node_name)
{
  // init the 'trival' members
	TF_Status* status = TF_NewStatus();
	graph = TF_NewGraph();

  // create a bunch of objects we need to init graph and session
	TF_Buffer* graph_def = ReadBinaryProto(binary_graphdef_protobuffer_filename);
	TF_ImportGraphDefOptions* opts  = TF_NewImportGraphDefOptions();
  TF_SessionOptions* session_opts = TF_NewSessionOptions();

  // import graph
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
	AssertOk(status);
  // and create session
	session = TF_NewSession(graph, session_opts, status);
	AssertOk(status);

	// prepare the constants for inference
	// input
	input_op 	= TF_GraphOperationByName(graph, input_node_name.c_str());
	input 		= {input_op, 0};

	// output
	output_op = TF_GraphOperationByName(graph, output_node_name.c_str());
	output		= {output_op, 0};

  // Clean Up all temporary objects
  TF_DeleteBuffer(graph_def);
  TF_DeleteImportGraphDefOptions(opts);
  TF_DeleteSessionOptions(session_opts);

	TF_DeleteStatus(status);
}

Inference::~Inference()
{
	TF_Status* status = TF_NewStatus();
  // Clean up all the members
	TF_CloseSession(session, status);
	TF_DeleteGraph(graph);
	//TF_DeleteSession(session); // TODO: delete session?

	TF_DeleteStatus(status);
	// input_op & output_op are delete by deleting the graph
}

TF_Tensor* Inference::operator()(TF_Tensor* input_tensor) const
{
	TF_Status* status = TF_NewStatus();
	TF_Tensor* output_tensor;
	TF_SessionRun(session, nullptr,
                &input,  &input_tensor,  1,
                &output, &output_tensor, 1,
                &output_op, 1,
                nullptr, status);
  AssertOk(status);
	TF_DeleteStatus(status);

	return output_tensor;
}

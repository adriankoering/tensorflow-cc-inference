# Tensorflow CC Inference

For the moment Tensorflow only provides a C-API that is easy to deploy and can be installed from pre-build binaries. This library aims to take away a lot of the overhead inflicted by the C-API and provide an easier-to-use interface that allows to execute trained tensorflow neural networks from C++.

It still is a little involved to produce a neural-network graph in the suitable format and to work with Tensorflow's C-API version of tensors. [This](https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f) great post by Jim Fleming might help to get started exporting the graph-definition into a binary-Protobuf format.


## Usage

``` C++
#include <tensorflow/c/c_api.h>
#include "tensorflow_cc_inference/tensorflow_inference.h"
using tensorflow_cc_inference::Inference;

auto CNN = Inference("path/to/graph", "input_node_name", "output_node_name");

TF_Tensor* in  = TF_AllocateTensor(/*Allocate and fill tensor*/);
TF_Tensor* out = CNN(in);

float* data = static_cast<float*>(TF_TensorData(out));
```
For a more detailed example on how to perform inference in C++ and information on how to export graphs from Python please check out the example directory.


## Installation

This library depends on tensorflow and its C-API. Installation is straightforward following the official documentation: [Tensorflow Install C](https://tensorflow.org/install/install_c)


## Build

To compile this library start with cloning the sources and creating a build directory in the repository root.

1. git clone git@github.com:adriankoering/tensorflow-cc-inference.git
2. cd tensorflow-cc-inference && mkdir build && cd build

    Then compile it via
3. cmake .. && make

     and optionally run the example:
4. ./multiply_with_two


## Glossary

**Frozen Graph** During training a neural network's weights are stored within variables, because they can be changed during training. For deployment and inference those weights are fixed and instead stored in constants. Replacing the variables in a graph with constants is called "freezing" the graph and produces a frozen graph.

**Graph** Neural Network

**Graph Definition** A Graph Definition describes the network architecture as tensorflow operations.

**Operation** A neural network in Tensorflow is constructed from operations. Every operations performs a calculation on the given input tensor and returns an output tensor.

**Protobuf** Dataformat that takes objects, serializes them to make them storable on disk. It is used by Tensorflow to save graph definitions and network weights in a file.

**Tensor** A tensor generalizes a matrix to arbitrary dimensions. Where a matrix is a two-dimensional (rows and columns) array of numbers, a tensor is an array of numbers with higher dimensions.

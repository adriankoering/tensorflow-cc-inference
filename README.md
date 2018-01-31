# tensorflow-cc-inference
Use trained tensorflow models for inference in a C++ application.

## Installation
 This library depends on tensorflow and its c-api.
 Copy the shared object file (libtensorflow.so) from here (TODO) into
 usr/local/lib and maybe the header from here (TODO) into /usr/local/tensorflow/c/.

 Then enter "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" to make
 the shared library available at runtime.

TF_TYPE="cpu"
OS="linux"
TARGET_DIRECTORY= $PWD"/lib/"
curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.5.0.tar.gz" | tar -C $TARGET_DIRECTORY -xz

 export LIBRARY_PATH=$LIBRARY_PATH:$TARGET_DIRECTORY
 export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TARGET_DIRECTORY

import os
path = os.getcwd()


# A trivial example for a start
import tensorflow as tf

# This placeholder is where the inference in C++ starts.
x = tf.keras.layers.Input((784, ))
y = tf.keras.layers.Dense(10, activation="softmax")(x)

M = tf.keras.models.Model(x, y)

# train the graph
# M.fit(...)

sess      = tf.keras.backend.get_session()
graph_def = sess.graph.as_graph_def()
print(graph_def) # get the input and output node name from here

# 'freeze' the graph (convert variables into constants): allows to have both,
# weights and the graph architecture in the same file
frozen_graphdef = tf.graph_util.convert_variables_to_constants(
    sess, graph_def, "Dense/Softmax")

# save the graph
tf.train.write_graph(graph_or_graph_def = frozen_graphdef,
                     logdir = path, # requires absolute path
                     name = "mnist.pb",
                     as_text = False)
# as_text must be false, because the C-API can only operate with binary graphs

# continue with 'multipy_with_two.cc' to see a graph used in C++

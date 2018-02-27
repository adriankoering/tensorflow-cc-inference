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

graph_def = tf.keras.backend.get_session().graph_def
print(graph_def) # get the input and output node name from here

# save the graph
tf.train.write_graph(graph_or_graph_def = graph_def,
                     logdir = path, # absolute path
                     name = "mnist.pb",
                     as_text = False)
# as_text must be false, because the C-API can only operate with binary graphs

# continue with 'multipy_with_two.cc' to see a graph used in C++

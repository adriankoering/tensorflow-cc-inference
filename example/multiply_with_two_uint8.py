import os
path = os.getcwd()


# A trivial example for a start
import tensorflow as tf
sess = tf.InteractiveSession()

# This placeholder is where the inference in C++ starts. Therefore the name
# set here will be used later as 'input_node_name'
x   = tf.placeholder(dtype=tf.uint8, shape=(1, ), name="x")
two = tf.constant(2, dtype=tf.uint8, shape=(1, ), name="two")
y   = tf.multiply(x, two, name="y")
# 'y' is where the inference ends: output_node_name = 'y'

# try the graph
print("3 * 2 = ", sess.run(y, feed_dict={x:(3, )}))

# save the graph
tf.train.write_graph(graph_or_graph_def = sess.graph_def,
                     logdir = path, # absolute path
                     name = "x_times_two_uint8.pb",
                     as_text = False)
# as_text must be false, because the C-API can only operate with binary graphs

# continue with 'multipy_with_two.cc' to see this graph used in C++

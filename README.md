# Eager-Execution-TensorFlow

**TensorFlow's eager execution** is an anvironment that evaluates operations immediately, without building graphs: operations return concrete values instead of constructing a computational graph to run later.

**tf.GradientTape** is used to train and/or compute gradients in eager. Since different operations can occur during each call, all forward-pass operations get recorded to a "tape".

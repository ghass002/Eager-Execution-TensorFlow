import tensorflow as tf

#enable Eager_execution
tf.enable_eager_execution()
tfe = tf.contrib.eager

# designing a model and a loss function
class Model():
	def __init__(self, input_shape, output_shape):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.W = tfe.Variable(tf.random_normal([self.input_shape, self.output_shape]))
		self.B = tfe.Variable( tf.random_normal([self.output_shape]))
		self.variables = [self.W, self.B]

	def frwrd_pass(self, X_train):
		out = tf.matmul(X_train, self.W) + self.B
		return out

	def loss(predicted_y, desired_y):
		return tf.reduce_mean(tf.square(predicted_y - desired_y))

model = Model()

#Training loop
def train(X_train, Y_train, epochs):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)

	for i in range(epochs):
		with tf.GradientTape() as tape:
			predicted = model.frwrd_pass(X_train)
			curr_loss = loss(predicted, Y_train)

		grads = tape.gradient(curr_loss, model.variables)
		optimizer.apply_gradients(zip(grads, model.variables), global_step = tf.train.get_or_create_global_step())


		print("Loss at step {:d}: {:.3f}".format(i,loss(model.frwrd_pass(X_train), Y_train)))




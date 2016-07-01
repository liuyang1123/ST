from soft_constraints_model import SoftConstraintsModel
import tensorflow as tf

# TODO Figure out how an event can be encoded in vector.
# TODO Test it using the DMN model.
#   - Whats the duration?
#   - What's the probability of attendance?
#   - Figure out if that works, how user's specific preferences can be added.
#       - For example: Only schedule meetings in the morning.


##################
# TODO
# Test if this works properly
# Update it to use save-restore
# Convert it to SoftConstraintsModel
##################


class MLP(object):

    def __init__(self):
        # Parameters
        self.lr = 0.05
        self.training_epochs = 25
        self.batch_size = 10
        self.display_step = 1

        # Network Parameters
        self.n_hidden_1 = 10  # 1st layer number of features
        self.n_hidden_2 = 20  # 2nd layer number of features
        self.n_input = 5  # Data input (type, timeslot, day, duration, place)
        self.n_classes = 2  # Participant decision classes (0-1 digit)

        self._build_graph()

    def _build_graph(self):
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.get_variable("w_h1", shape=(self.n_input,
                                                 self.n_hidden_1)),
            'h2': tf.get_variable("w_h2", shape=(self.n_hidden_1,
                                                 self.n_hidden_2)),
            'out': tf.get_variable("w_out", shape=(self.n_hidden_2,
                                                   self.n_classes))}
        biases = {
            'b1': tf.get_variable("b1", shape=(self.n_hidden_1)),
            'b2': tf.get_variable("b2", shape=(self.n_hidden_2)),
            'out': tf.get_variable("b_out", shape=(self.n_classes))
        }

        # Create model
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self.x, weigths['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weigths['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        self.pred = tf.matmul(layer_2, weights['out']) + biases['out']  # Model
        # Define loss and optimizer
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.pred, self.y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr).minimize(self.cost)

    def do_epoch(self, events, labels):
        avg_cost = 0.
        total_batch = int(events.train.num_examples / self.batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = events.train.next_batch(self.batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([self.optimizer, self.cost], feed_dict={
                            self.x: batch_x, self.y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost)

    def train(self, events, labels):
        # Initializing the variables
        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(self.training_epochs):
                self.do_epoch(events, labels)

            print "Optimization Finished!"

            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1),
                                          tf.argmax(self.y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print "Accuracy:", accuracy.eval({x: events.test.data, y: events.test.labels})


# class MLPModel(SoftConstraintsModel):
#     model_type = "MultiLayerPerceptron"
#
#     def _load(self):
#         """
#         Loads the weights if exists
#         """
#         pass
#
#     def _build(self, args):
#         """
#
#         """
#         pass
#
#     def predict(self, args):
#         pass
#
#     def score_event(self, event):
#         # TODO Armar un diccionario con los datos del evento
#
#         pass
#
#     def train(self, data, labels):
#         pass
#
#     def save(self):
#         pass
#
#     def _create_default(self):
#         """
#
#         """
#         return {}
#
#     def _get_distribution_values(self):
#         return

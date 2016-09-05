'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
import tensorflow as tf
from networkmodel import NetworkModel

class MultiLayerPerceptronModel(NetworkModel):
    '''
    '''
    network_type = ""

    def train(self, dataset):
        """
        Start the learning algorithm
        """
        if not self.model:
            self._build()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(self._init)
            # Training cycle
            for epoch in range(self.config.max_epochs):
                avg_cost = 0.
                total_batch = int(dataset.number_of_examples_train()/self.config.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = dataset.next_batch(
                        self.config.batch_size, "train")
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self._optimizer, self._cost],
                                    feed_dict={self._x: batch_x,
                                               self._y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % self.config.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                          "{:.9f}".format(avg_cost))
            print("Optimization Finished!")
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.model, 1),
                                          tf.argmax(self._y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            x_test, y_test = dataset.next_batch(dataset.number_of_examples_test,
                                                "test")
            print("Accuracy:", accuracy.eval({self._x: x_test,
                                              self._y: y_test}))

            # Save model weights to disk
            save_path = self._saver.save(sess, self.config.model_path)
            print("Model saved in file: %s" % save_path)

    def save(self):
        """
        Allows to save the training results, in order to restore them for later use
        """
        return

    def predict(self, query):
        if not self.model:
            self._build()
        result = None
        # Running a new session
        print("Starting 2nd session...")
        with tf.Session() as sess:
            # Initialize variables
            sess.run(self._init)

            # Restore model weights from previously saved model
            self._saver.restore(sess, self.config.model_path)
            print("Model restored from file: %s" % self.config.model_path)

            # Predict
            # _p = sess.run([self.model], feed_dict={x: np.array(query['x']),
            #                                        y: np.array(query['y'])})
            _p = tf.argmax(self.model, 1)
            result = _p.eval({self._x: np.array(query['x']),
                              self._y: np.array(query['y'])})
            print("Result:", result)

        return result

    def _load(self):
        return

    def _process_dataset(self, dataset):
        return

    def _build(self):
        """
        Constructs the model
        """
        # Input
        self._x = tf.placeholder(tf.float32, [None, self.config.n_input])
        self._y = tf.placeholder(tf.float32, [None, self.config.n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([self.config.n_input,
                                                self.config.hidden1_dim])),
            'h2': tf.Variable(tf.random_normal([self.config.hidden1_dim,
                                                self.config.hidden2_dim])),
            'out': tf.Variable(tf.random_normal([self.config.hidden2_dim,
                                                 self.config.n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.config.hidden1_dim])),
            'b2': tf.Variable(tf.random_normal([self.config.hidden2_dim])),
            'out': tf.Variable(tf.random_normal([self.config.n_classes]))
        }
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(self._x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        self.model = tf.matmul(layer_2, weights['out']) + biases['out']

        ## Define loss and optimizer
        self._cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.model, self._y))
        self._optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.lr).minimize(self._cost)

        # Initializing the variables
        self._init = tf.initialize_all_variables()

        # 'Saver' op to save and restore all the variables
        self._saver = tf.train.Saver()

    def _default(self):
        return

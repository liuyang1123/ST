"""
Most tasks in NLP can be cast in QA problems over language input.
This model (DMN) is a unified neural network framework which processes
input sequences and questions, forms semantic and episodic memories,
and generates relevant answers.
Questions trigger an iterative attention process with allow the model
to condition its attention on the result of previous iterations.
This results are then reasoned over in the hierarchical recurrent sequence
model to generate answers

The model relies exclusively on trained word vector representations and
requires no string matching or manually engineered features.


TODO
    Code organization:
        - Organize as MemN2N from carpedm20 (github) and similar TF projects.
                - Separate data processing, config

    Data processing:
        - Remove this from DMNClass
        - Support Batches
        - Shuffle data - Already supported in utils.py
        - Check that embeddings is done efficiently (compare to a lookup)

    Improve graph:
        - Implement changes in the episodic memory - avoid using gather!
        - Calculate batch accuracy
        - Implement DMN+ improvements
                - Regularization
                - Input Fusion layer
                - Untied weights
        - Add methods for interactive prediction (give context, and question, return an answer)
                - Used later in the API
        - Save and load results - tensorflow
        - Gradient clipping

    Visualization:
    - TensorBoard
    - API UI:   - Show episodes - probs., predict, feed examples, start learning, etc.
                - Different corpus - NER, POS, etc.
                    - Slot filling - ATIS corpus
                    - Sentiment analysis

    New Features:
            - VQA
"""
import tensorflow as tf
from networkmodel import NetworkModel

class DynamicMemoryNetworkModel(NetworkModel):
    network_type = "DynamicMemoryNetwork"

    def train(self, dataset):
        """
        Start the learning algorithm
        """
        print("Init finished!")

        max_sent_length = 0

        # Train over multiple epochs
        with tf.Session() as sess:
            best_loss = float('inf')
            best_val_epoch = 0
            sess.run(init)

            # train until we reach the maximum number of epochs
            for epoch in range(self.args.max_epochs):
                total_training_loss = 0
                num_correct = 0
                prev_prediction = 0

                print(" ")
                print('Epoch {}'.format(epoch))
#                 start = time.time()

                for i in range(len(self.train_input)):
                    ans = np.zeros((1, self.vocab_size))
                    ans[0][self.train_answer[i]] = 1

                    # For debugging:
                    # Input module: _input_tensor - self.input_only_for_testing
                    # Question module: _question_representation - self.question_representation
                    # Episode module: _e_i - self.e_i / _e_m_s - self.episodic_memory_state
                    loss, _, pred_prob, _projections = sess.run(
                        [cost, optimizer, self.prediction, self.projections],
                        feed_dict={self.input_placeholder: [self.train_input[i]],
                                   self.input_length_placeholder: [len(self.train_input[i])],
                                   self.end_of_sentences_placeholder: [self.train_input_mask[i]],
                                   self.question_placeholder: [self.train_q[i]],
                                   self.question_length_placeholder: [len(self.train_q[i])],
                                   self.labels_placeholder: ans,
                                   self.gate_placeholder: [float(self.train_gate[i])]})

                    total_training_loss += loss

                    if np.argmax(pred_prob) == np.argmax(ans):
                        num_correct += 1

                    if i % self.args.update_length == 0:
                        print "Current average training loss: {}".format(total_training_loss / (i + 1))
                        print "Current training accuracy: {}".format(float(num_correct) / (i + 1))
                        print("Ans: " + str(self.ivocab[np.argmax(ans)]))
                        print("Pred: " + str(self.ivocab[np.argmax(pred_prob)]))

    def save(self):
        """
        Allows to save the training results, in order to restore them for later use
        """
        return

    def predict(self, query):
        return

    def _load(self):
        return

    def _process_dataset(self, dataset):
        return

    def _add_placeholders(self):
        """
        Generate placeholder variables to represent the input tensors
        These placeholders are used as inputs by the rest of the model building
        code and will be fed data during training.  Note that when "None" is in a
        placeholder's shape, it's flexible.
        """

        # 1 = Batch Size
        self.input_placeholder = tf.placeholder(
            tf.float32, shape=[self.args.batch_size, None,
                               self.args.word_vector_length])
        self.input_length_placeholder = tf.placeholder(tf.int32, shape=[self.args.batch_size])
        self.end_of_sentences_placeholder = tf.placeholder(tf.int32, shape=[self.args.batch_size, None])
        self.question_placeholder = tf.placeholder(tf.float32, shape=[self.args.batch_size, None, self.args.word_vector_length])
        self.question_length_placeholder = tf.placeholder(tf.int32, shape=[self.args.batch_size])
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[self.args.batch_size, self.vocab_size])
        self.gate_placeholder = tf.placeholder(tf.float32, shape=[self.args.batch_size])

    # Takes in an input matrix of size (number of words in input) x (WORD_VECTOR_LENGTH) with the input word vectors
    # and a tensor the length of the number of sentences in the input with the index of the word that ends each
    # sentence and returns a list of the states after the end of each sentence to be fed to other modules.
    def _input_module(self):
        """
        Computes a useful representation of the inputs such that the relevant
        facts can be retrieved later.

        - Responsible for computing representations of (audio, visual or) textual
        inputs, such that they can be retrieved when needed later.
        - Assume a temporal sequence indexable by a time stamp.
        - For written language we have a sequence of words (v1, v2, ..., vTw).
        - Recurrent Neural Network (RNN) computation for context states -> GRU

        TODO:
            - Make this more abstract for "lego" purposes
            - Support visual input (DMN+). OOP-tensorflow?
            - The new class should provide methods for visualization.
                Word vectors, GRU results.
        """
        # Compute the number of words in the input
        self.cell = tf.nn.rnn_cell.GRUCell(num_units=self.args.hidden_size,
                                           input_size=self.args.word_vector_length)

        # Get output after every word.
        # outputs: (BATCH_SIZE, n_words, HIDDEN_SIZE)
        # state: The last output (1, HIDDEN_SIZE)
        outputs, state = tf.nn.dynamic_rnn(self.cell,
                                           self.input_placeholder,
                                           self.input_length_placeholder,
                                           dtype='float32')

        context_rnn_out = tf.reshape(outputs, [-1, self.args.hidden_size]) # Now the shape = [n_words, n_units]

        # Only project the state at the end of each sentence
        sentence_representations_mat = tf.gather(context_rnn_out,
                                                 self.end_of_sentences_placeholder) # (1, n_sentences, HIDDEN_SIZE)

        # The code below it necessary to split the Tensor(?,HIDDEN_SIZE) in a list of tensors with shape (1,HIDDEN_SIZE)

        # Reshape `X` as a vector. -1 means "set this dimension automatically".
        sentences_as_vector = tf.reshape(sentence_representations_mat, [-1])

        # Create another vector containing zeroes to pad `X` to (MAX_INPUT_LENGTH * WORD_VECTOR_LENGTH) elements.
        zero_padding = tf.zeros(
            [self.args.max_input_sentences * self.args.hidden_size] - tf.shape(sentences_as_vector),
            dtype=sentences_as_vector.dtype)

        # Concatenate `X_as_vector` with the padding.
        sentences_padded_as_vector = tf.concat(0, [sentences_as_vector, zero_padding])

        # Reshape the padded vector to the desired shape.
        sentences_padded = tf.reshape(sentences_padded_as_vector, [self.args.max_input_sentences,
                                                                   self.args.hidden_size])

        # Split X into a list of tensors of length MAX_INPUT_LENGTH where each tensor is a 1xHIDDEN_SIZE vector
        # of the word vectors
        self.sentence_representations = tf.split(0, self.args.max_input_sentences, sentences_padded)

        self.number_of_sentences = tf.shape(sentence_representations_mat)[1]

        self.input_only_for_testing = tf.concat(0, self.sentence_representations)

        # Result of this module:
        #                        - List of sentences. Each sentence is a tensor of shape (1, HIDDEN_UNITS). 1=Batch_size
        #                        - The number of sentences.

    def _semantic_memory_module(self):
        """
        Consists of:
        1 - Stored word concepts
        2 - Facts about them

        Initialized with GloVe vectos.

        This module could include gazeteers or other forms of explicit
        knowledge bases.


        Semantic Memory --> Episodic Memory
        Semantic Memory <-> Input Text Sequences
        """
        return

    def _question_module(self):
        """
        Simple GRU over question word vectors.

        qt = GRU(vt, qt-1)
        """
        # Get output after every word.
        # outputs: (1, n_words, HIDDEN_SIZE)
        # state: The last output (1, HIDDEN_SIZE)
        with tf.variable_scope("", reuse=True):
            outputs, state = tf.nn.dynamic_rnn(self.cell,
                                               self.question_placeholder,
                                               self.question_length_placeholder,
                                               dtype='float32')
        self.question_representation = state # BATCH_SIZE, HIDDEN_SIZE


    def _episodic_memory_module(self):
        """
        Retrieves facts from the input module conditioned on the question.
        It then reasons over those facts to produce a final representation that
        the answer module will use to generate an answer (memory).
        Each pass produces an episode, an these episodes are then summarized into
        the memory.

        - Combines the input and question modules in order to reason over them
        and give the resulting knowledge to the answer module.
        - Dynamically retrieves the necessary information over the sequence of
        words or sentences.
        - If necessary to retrieve additional facts -> iterate over inputs.
        - Needed for transitive inference.


        TODO:
            - Refactor: - Remove (1). Using operations tile over the variables q and m.
                                The resulting shape of z should be MAX_INPUT_SENTENCES, 1
                                To get the tensor in for sentence t, do reshape [-1], and
                                then use gather op. No need to do the gather in the input
                                module.
                                c : N_sent, H
                                q : 1, H --tile--> N_sent, H
                                z : N_sent, 1
                                g : softmax(z) -> gt = softmax[t]
                        - Extract from (2), the calculation for g. This should be a
                                tf.nn.softmax(z)
                        - Is it necessary to use the same weigths in the GRU?
        """
        W_mem_res_in = tf.get_variable("W_mem_res_in", shape=(self.args.hidden_size,
                                                              self.args.hidden_size))
        W_mem_res_hid = tf.get_variable("W_mem_res_hid", shape=(self.args.hidden_size,
                                                                self.args.hidden_size))
        b_mem_res = tf.get_variable("b_mem_res", shape=(self.args.hidden_size,))

        W_mem_upd_in = tf.get_variable("W_mem_upd_in", shape=(self.args.hidden_size,
                                                              self.args.hidden_size))
        W_mem_upd_hid = tf.get_variable("W_mem_upd_hid", shape=(self.args.hidden_size,
                                                                self.args.hidden_size))
        b_mem_upd = tf.get_variable("b_mem_upd", shape=(self.args.hidden_size,))

        W_mem_hid_in = tf.get_variable("W_mem_hid_in", shape=(self.args.hidden_size,
                                                              self.args.hidden_size))
        W_mem_hid_hid = tf.get_variable("W_mem_hid_hid", shape=(self.args.hidden_size,
                                                                self.args.hidden_size))
        b_mem_hid = tf.get_variable("b_mem_hid", shape=(self.args.hidden_size,))

        memory_states = [self.question_representation]

        with tf.variable_scope("memory"):
            gru_cell = tf.nn.rnn_cell.GRUCell(num_units = self.args.hidden_size)
            for i in range(self.args.max_episodes):
                z_vector = []
                g_vector = []
                e_vector = []
                m_prev = memory_states[-1]
                for t in range(self.args.max_input_sentences): # (1)
                    s = self.sentence_representations[t]
                    q = self.question_representation

                    with tf.variable_scope("episode0", reuse=True if (t > 0 or i > 0) else None):
                        W_b = tf.get_variable("W_b", shape=(self.args.hidden_size, self.args.hidden_size))

                        W_1 = tf.get_variable("W_1", shape=(4 * self.args.hidden_size,
                                                            self.args.attention_gate_hidden_size))
                        b_1 = tf.get_variable("b_1", shape=(1, self.args.attention_gate_hidden_size))
                        W_2 = tf.get_variable("W_2", shape=(self.args.attention_gate_hidden_size, 1))
                        b_2 = tf.get_variable("b_2", shape=(1, 1))

                        # Compute z
                        # z(s, m, q) is BATCH_SIZE x (7 * HIDDEN_SIZE + 2)
                        z_t = tf.concat(1, [tf.mul(s, q),
                                            tf.mul(s, m_prev),
                                            tf.abs(tf.sub(s, q)), # tf.pow(tf.sub(s, q), 2),
                                            tf.abs(tf.sub(s, m_prev)) # tf.pow(tf.sub(s, m_prev), 2)
                                           ])

                        z_t = tf.add(tf.matmul(tf.tanh(tf.add(tf.matmul(z_t, W_1), b_1)), W_2), b_2)
                        z_t = tf.exp(z_t)
#                         print("z_t")
#                         print(z_t)
                        self.z = tf.reduce_sum(z_t)
                        z_vector.append(self.z) # This should be an scalar


                # Gating function as the attention mechanism
                # g = G(s, m_prev, q)
                # G returns a single scalar.
                h_prev = tf.zeros((1, self.args.hidden_size))
                for t in range(self.args.max_input_sentences): # (2)
                    s = self.sentence_representations[t]
                    g = z_vector[t] / tf.add_n(z_vector) # Softmax
                    # TODO:
                    #     - Add training capabilities - CE g
                    #     - Add end of passes representation to the facts, and stop the
                    #       iterative attention process if this representation is chosen by the
                    #       gate function.
                    self.g = tf.reduce_sum(g)
                    g = self.g

#                     print("g")
#                     print(g)
                    g_vector.append(g) # This should be an scalar

                    # TODO: Figure out if it's necessary to share the weights?

                    # Compute the episode for pass i, using a modified GRU.
                    r = tf.sigmoid(tf.add(tf.add(tf.matmul(s, W_mem_res_in),
                                                 tf.matmul(h_prev, W_mem_res_hid)),
                                          b_mem_res))
                    _h = tf.nn.tanh(tf.add(tf.matmul(s, W_mem_hid_in),
                                           tf.add(tf.mul(r, tf.matmul(h_prev, W_mem_hid_hid)),
                                                  b_mem_hid)))
                    h = tf.add(tf.mul(g, _h), tf.mul(tf.sub(1., g), h_prev))

                    h_prev = h

                    # TODO Replace the modified GRU using: gt * GRU(ct, hi-1) + (1-gt) * h_t-1
                    # Look at other implementations

                e = h_prev # e^i = h_tc -> Final state of the modified GRU.
                # In case of a sequence modeling
                # e^i = h_t -> for word t the episode is the state at t.
                # And each word's unique m is sent independently to the answer module.

                with tf.variable_scope("", reuse=True if i > 0 else None):
                    # Summarize the episodes e^i into a memory, using the same GRU that
                    # updates the attention mechanism's state.
#                     _output, new_mem_state = gru_cell(e, m_prev)
                    gru_h_prev = m_prev
                    gru_x = e
                    # TODO Figure out a way to use the same weigths using TF built in GRU

                    mem_u = tf.sigmoid(tf.add(tf.add(tf.matmul(gru_x, W_mem_upd_in),
                                                     tf.matmul(gru_h_prev, W_mem_upd_hid)),
                                              b_mem_upd))
                    mem_r = tf.sigmoid(tf.add(tf.add(tf.matmul(gru_x, W_mem_res_in),
                                                     tf.matmul(gru_h_prev, W_mem_res_hid)),
                                              b_mem_res))
                    mem_h_hat = tf.nn.tanh(tf.add(tf.matmul(gru_x, W_mem_hid_in),
                                                  tf.add(tf.mul(mem_r,
                                                                tf.matmul(gru_h_prev, W_mem_hid_hid)),
                                                         b_mem_hid)))
                    new_mem_state = tf.add(tf.mul(mem_u, mem_h_hat), tf.mul(tf.sub(1., mem_u), gru_h_prev))

                    memory_states.append(new_mem_state)

            # Return final memory state
            self.episodic_memory_state = memory_states[-1]


    def _answer_module(self):
        """
        This module decodes the memory into a sequence of words representing the
        answer.

        - Simple GRU to produce an output at each of its time steps.
        - Allow to predict end of sentence and stop.
        """
        # a_0 = m
        # a_t = GRU([y_t-1, q], a_t-1), y_t = softmax(W^a * a_t)

#         q = self.question_representation
#         a_0 = self.episodic_memory_state
#         y_t_1 = tf.nn.softmax(tf.matmul(a_0, W_a)) # 1 x vocab
#         concat = tf.concat(1, [y_t_1, q])

#         with tf.variable_scope("answer_module_gru"):
#             gru_cell_memory = tf.nn.rnn_cell.GRUCell(num_units=HIDDEN_SIZE)
#             output, a = gru_cell_memory(concat, a_0)

#         # Using GRU
#         self.projections = tf.matmul(a, W_a)
#         self.prediction = tf.nn.softmax(self.projections)

        # Without GRU
#         self.projections = tf.matmul(a_0, W_out)
#         self.prediction = tf.nn.softmax(self.projections)

        with tf.variable_scope("answer_module"):
            W_out = tf.get_variable("W_out", shape=(self.args.hidden_size, self.vocab_size))
            b_out = tf.get_variable("b_out", shape=(1, self.vocab_size))

        self.projections = tf.matmul(self.episodic_memory_state, W_out) + b_out
        self.prediction = tf.nn.softmax(self.projections)

    def _build(self):
        """
        Constructs the model
        """
        """
        The semantic memory module (SMM) gives the words vectors (GloVe) to
        the input module (IM), which computes the representation for each
        sentence. The question module (QM) computes a representation of the
        question too. The output of this two modules are used in the episodic
        memory module (EMM) and reason over them. The output is then given to
        the answer module (AM).

        For bAbI tasks the Mean Accuracy (%) should be 93.6.

        TODO:
            - Cross entropy training of the gates.
                bAbI posee en el dataset esta informacion, tratar de
                generalizar la clase para soportar otros datasets
                J = a * CE(Gates) + b * CE(Answers)
        """
        # max_sent = 0
        # for i in range(len(self.train_input)):
        #     if max_sent < len(self.train_input_mask[i]):
        #         max_sent = len(self.train_input_mask[i])
        #         print(max_sent)
        # return

        self._add_placeholders()
        self._input_module()
        self._question_module()
        self._episodic_memory_module()
        self._answer_module()

        global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
        #                                            global_step,
        #                                            1000, 0.1, staircase=True) # Every epoch

        # Compute loss
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.projections, self.labels_placeholder))

        # Add optimizer
        #         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(cost)

        # correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels_placeholder, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Initialize all variables
        init = tf.initialize_all_variables()
        # saver = tf.train.Saver()

    def _default(self):
        return

import numpy as np, tensorflow as tf, time, os
from collections import deque
from networkmodel import NetworkModel


class CFBaselineModel(NetworkModel):
    network_type = "CFBaseline"

    def _clip(self, x):
        return np.clip(x, 1.0, 5.0)

    def train(self, dataset):
        """
        Start the learning algorithm
        """
        if not self.model:
            self._build()

        samples_per_batch = dataset.number_of_examples_train() // self.config.batch_size

        with tf.Session() as sess:
            sess.run(self.init)
            print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
            errors = deque(maxlen=samples_per_batch)
            start = time.time()
            for i in range(self.config.max_epochs * samples_per_batch):
                tr_elems, rates = dataset.next_batch(samples_per_batch)

                _, pred_batch = sess.run([self.train_op, self.rui_hat],
                                         feed_dict={self.user_batch: tr_elems[0],
                                                    self.item_batch: tr_elems[1],
                                                    self.rui_batch: rates})
                pred_batch = self._clip(pred_batch)
                errors.append(np.power(pred_batch - rates, 2))
                if i % samples_per_batch == 0:
                    train_err = np.sqrt(np.mean(errors))
                    test_err2 = np.array([])

                    test_elems, rates = dataset.next_batch(
                        dataset.number_of_examples_test(), "test")
                    pred_batch = sess.run(self.rui_hat,
                                          feed_dict={self.user_batch: test_elems[0],
                                                     self.item_batch: test_elems[1]})
                    pred_batch = self._clip(pred_batch)
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))

                    end = time.time()
                    print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)),
                                                           end - start))
                    start = end

            # output_graph_def = tf.python.framework.graph_util.extract_sub_graph(sess.graph.as_graph_def(),
                                                                                #  ["svd_inference", "svd_regularizer"])
            # tf.train.SummaryWriter(logdir="./tmp/svd", graph=sess.graph)

            # Save model weights to disk
            save_path = self._saver.save(sess, self.config.model_path)
            # print("Model saved in file: %s" % save_path)

    def save(self):
        """
        Allows to save the training results, in order to restore them for later use
        """
        return

    def predict(self, query):
        if not self.model:
            self._build()

        result = np.array([])
        with tf.Session() as sess:
            # Initialize variables
            sess.run(self.init)

            # Restore model weights from previously saved model
            self._saver.restore(sess, self.config.model_path)
            print("Model restored from file: %s" % self.config.model_path)

            # Predict
            _p = sess.run([self.model],
                          feed_dict={self.user_batch: np.array(query["users"]),
                                     self.item_batch: np.array(query["items"])})
            result = np.array(_p)

        return {"prediction": result.tolist(),
                "message": "Prediction contains the estimated rating."}

    def _load(self):
        return

    def _process_dataset(self, dataset):
        return

    def _build(self):
        """
        Constructs the model
        """
        save_dir = '/'.join(self.config.model_path.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Simplified way - all together
        self.user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
        self.item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
        self.rui_batch = tf.placeholder(tf.float32, shape=[None]) # The rating for each pair of user (u, i)

        average_rating = tf.get_variable("mu", []) # (1)
        embedding_user = tf.get_variable("bu", [self.config.m_users]) # M_Users, 1 Column vector
        embedding_item = tf.get_variable("bi", [self.config.n_items]) # 1, N_Items Row vector
        deviation_user = tf.nn.embedding_lookup(embedding_user,
                                                self.user_batch, name="bu_u")
        deviation_item = tf.nn.embedding_lookup(embedding_item,
                                                self.item_batch, name="bi_i")

        # https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf
        # Without bias regularization
        Pu = tf.get_variable("Pu", [self.config.m_users, self.config.dim])
        Qi = tf.get_variable("Qi", [self.config.n_items, self.config.dim])

        # reviewed = tf.get_variable("Nu", [M_Users, N_Items])
        # Xi = tf.get_variable("Xi", [N_Items, DIM])


        emb_pu = tf.nn.embedding_lookup(Pu, self.user_batch,
                                        name="embedding_user")
        emb_qi = tf.nn.embedding_lookup(Qi, self.item_batch,
                                        name="embedding_item")

        # The mul term has a shape of (batch_size, DIM)
        # And then because we need a column vector, we sum the columns
        self.rui_hat = average_rating
        self.rui_hat = tf.add(self.rui_hat, deviation_user)
        self.rui_hat = tf.add(self.rui_hat, deviation_item)
        # tf.reduce_sum(tf.matmul(emb_pu, tf.transpose(emb_qi)), 0)
        self.rui_hat = tf.add(self.rui_hat, tf.reduce_sum(tf.mul(emb_pu, emb_qi), 1),
                         name="svd_inference")
        self.model = self.rui_hat
        # Original regularization term from the paper:
        reg_term = tf.add(tf.nn.l2_loss(emb_pu),
                          tf.nn.l2_loss(emb_qi),
                          name="svd_regularizer")
        # t1 = tf.add(tf.nn.l2_loss(deviation_user),
        #             tf.nn.l2_loss(deviation_item),
        #             name="svd_regularizer_term1")
        # t2 = tf.add(tf.nn.l2_loss(emb_pu),
        #             tf.nn.l2_loss(emb_qi),
        #             name="svd_regularizer_term2")
        # reg_term = tf.add(t1, t2, name="svd_regularizer")

        # Objective function definition

        # l2_cost = tf.nn.l2_loss(tf.sub(rui_hat, rui_batch))
        l2_cost = tf.reduce_sum(tf.square(tf.sub(self.rui_hat, self.rui_batch)))
        penalty = tf.constant(self.config.reg_lambda, dtype=tf.float32, shape=[], name="l2")
        self.cost = tf.add(l2_cost, tf.mul(reg_term, penalty))
        self.train_op = tf.train.FtrlOptimizer(self.config.lr).minimize(self.cost)

        self.init = tf.initialize_all_variables()

        # 'Saver' op to save and restore all the variables
        self._saver = tf.train.Saver()

    def _default(self):
        return


# NSVD Model - by Paterek
# Reemplazar Pu por: (SUM j->R(u) (Xj)) / |R(u)|^(1/2), donde R(u) es el conjunto de items que el usuario u puntuo
# Xj al igual que Pu -> N x f
# R(u) -> [N, 1] donde en cada celda hay un 0 o 1, dependiendo si puntuo ese item o no
# Esto es implicit feedback que todos los datasets lo tienen

# TODO Implement SVD++
# http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf
# http://dparra.sitios.ing.uc.cl/classes/recsys-2015-2/student_ppts/CRojas_SVDpp-PMF.pdf

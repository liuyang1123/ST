# SVD - Singular Value Decomposition

BATCH_SIZE = 100
MAX_EPOCHS = 100

def svd(number_of_users, number_of_items, batch_users, batch_items, dimension=10):
    # All the biases: mu, b_u, b_i
    average_rating = tf.get_variable("mu", shape=[]) # This is one number. AKA global bias
    embedding_user_bias = tf.get_variable("b_u", shape=[number_of_users])
    embedding_item_bias = tf.get_variable("b_i", shape=[number_of_items])

    # Get only the rows needed for the batch (b_u, b_i for this batch)
    bias_users = tf.nn.embedding_lookup(embedding_user_bias, batch_users,
                                        name="bias_user")
    bias_items = tf.nn.embedding_lookup(embedding_item_bias, batch_items,
                                        name="bias_item")
    # P matrix : N x d. d is the features length.
    embedding_p_user = tf.get_variable("embedding_p_u", shape=[number_of_users,
                                                               dimension])
    p_u = tf.nn.embedding_lookup(embedding_p_user, batch_users, name="p_u")
    # Q matrix : M x d
    embedding_q_item = tf.get_variable("embedding_q_i", shape=[number_of_items,
                                                               dimension])
    q_i = tf.nn.embedding_lookup(embedding_q_item, batch_items, name="q_i")

    # EstimateRating_ui = average_rating + bias_users + bias_items + q_iT * p_u
    infer = tf.reduce_sum(tf.matmul(q_i, p_u), 1) # We need the column vector
    infer = tf.add(infer, average_rating)
    infer = tf.add(infer, bias_users)
    infer = tf.add(infer, bias_items, name="svd_inference")

    # Regularization
    regularization = tf.add(tf.nn.l2_loss(q_i), tf.nn.l2_loss(p_u), name="svd_regularizer")

    return infer, regularization

def opt(infer, regularizer, rate_batch, lr=0.05, reg=0.5):
    l2_cost = tf.nn.l2_loss(tf.sub(infer, rate_batch))
    penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
    cost = tf.add(l2_cost, tf.mul(regularizer, penalty))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    return cost, train_op

def cf(df_train, df_test):
    datasets, total_users, total_items = read_data_sets_cf()
    samples_per_batch = datasets.train.num_examples() // BATCH_SIZE

    # The list of user ids, and item ids in this batch
    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    # The rating for each pair of user (u, i)
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = svd(total_users, total_items, user_batch, item_batch)
    _, train_op = opt(infer, regularizer, rate_batch, lr=0.1, reg=0.1)

    init = tf.initializer_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(MAX_EPOCHS * samples_per_batch):
            values, rates = datasets.train.next_batch(BATCH_SIZE)
            users = values[0]
            items = values[1]
            _, pred_batch = sess.run([train_op, infer],
                                     feed_dict={user_batch: users,
                                                item_batch: items,
                                                rate_batch: rates})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                # TODO
                # for users, items, rates in iter_test:
                #     pred_batch = sess.run(infer, feed_dict={user_batch: users,
                #                                             item_batch: items})
                #     pred_batch = clip(pred_batch)
                #     test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2)),
                                                       end - start))
                start = end
        output_graph_def = tf.python.client.graph_util.extract_sub_graph(sess.graph.as_graph_def(),
        tf.train.SummaryWriter(logdir="/tmp/svd", graph_def=output_graph_def)                          ["svd_inference", "svd_regularizer"])

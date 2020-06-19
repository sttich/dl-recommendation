# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import os
from datetime import datetime as dt

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from gpu_ops import gpu_diag_wide
from tensorflow.python.ops import rnn_cell


class GRU4Rec:

    def __init__(self, sess, args):
        self.sess = sess
        self.is_training = args.is_training

        self.layers = args.layers
        self.rnn_size = args.rnn_size
        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.dropout_p_hidden = args.dropout_p_hidden
        self.learning_rate = args.learning_rate
        self.decay = args.decay
        self.decay_steps = args.decay_steps
        self.sigma = args.sigma
        self.init_as_normal = args.init_as_normal
        self.reset_after_session = args.reset_after_session
        self.session_key = args.session_key
        self.item_key = args.item_key
        self.time_key = args.time_key
        self.user_key = args.user_key
        self.grad_cap = args.grad_cap
        self.n_items = args.n_items
        self.dwell_time = args.dwell_time
        self.user_data = args.user_data  # 0: no user in data, 1: taobao user data
        self.optimizer = args.optimizer
        self.boosting_memory = args.boosting_memory
        self.bpr_max_lambda = args.bpr_max_lambda
        self.n_samples = args.n_samples
        self.sample_alpha = args.sample_alpha

        # set hidden activation function
        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        # set loss function
        if args.loss == 'cross-entropy' or args.loss == 'cross_entropy':
            if args.final_act == 'tanh':
                self.final_activation = self.softmaxth
            else:
                self.final_activation = self.softmax
            self.loss_function = self.cross_entropy

        elif args.loss == 'bpr':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activation = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr

        elif args.loss == 'top1':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1

        elif args.loss == 'top1_max' or args.loss == 'top1-max':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.top1_max

        elif args.loss == 'bpr_max' or args.loss == 'bpr-max':
            if args.final_act == 'linear':
                self.final_activation = self.linear
            elif args.final_act == 'relu':
                self.final_activatin = self.relu
            else:
                self.final_activation = self.tanh
            self.loss_function = self.bpr_max
        else:
            raise NotImplementedError

        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            raise Exception("[!] Checkpoint Dir not found")

        if self.user_data != 0:
            self.n_users = args.n_users
        self.build_model()  # This function will distinguish whether to use user information
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

        if self.is_training:
            return

        # use self.predict_state to hold hidden states during prediction.
        self.predict_state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if self.user_data:
                self.saver.restore(sess, '{}/gru-model_user-{}'.format(self.checkpoint_dir, args.test_model))
            else:
                self.saver.restore(sess, '{}/gru-model-{}'.format(self.checkpoint_dir, args.test_model))

    # Activation Function
    def linear(self, X):
        return X

    def tanh(self, X):
        return tf.nn.tanh(X)

    def softmax(self, X):
        return tf.nn.softmax(X)

    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))

    def relu(self, X):
        return tf.nn.relu(X)

    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    def softmax_neg(self, X):  # need more testing
        hm = 1.0 - tf.eye(num_rows=X.get_shape().as_list()[0], num_columns=X.get_shape().as_list()[1])
        X = X * hm
        e_x = tf.exp(X - tf.expand_dims(tf.reduce_max(X, axis=1), 1)) * hm
        return e_x / tf.expand_dims(tf.reduce_sum(e_x, axis=1), 1)

        # Loss Function

    def cross_entropy(self, yhat):
        return tf.reduce_mean(-tf.log(gpu_diag_wide(yhat) + 1e-24))

    def bpr(self, yhat):
        yhatT = tf.transpose(yhat)
        return tf.reduce_mean(-tf.log(tf.nn.sigmoid(gpu_diag_wide(yhat) - yhatT)))

    def top1(self, yhat):
        yhatT = tf.transpose(yhat)
        term1 = tf.reduce_mean(tf.nn.sigmoid(-gpu_diag_wide(yhat) + yhatT) + tf.nn.sigmoid(yhatT ** 2), axis=0)
        term2 = tf.nn.sigmoid(gpu_diag_wide(yhat) ** 2) / (self.batch_size + self.n_samples)
        return tf.reduce_mean(term1 - term2)

    def top1_max(self, yhat):
        term1 = self.softmax_neg(yhat)
        term2 = tf.sigmoid(-tf.expand_dims(gpu_diag_wide(yhat), 1) + yhat) + tf.sigmoid(yhat ** 2)
        return tf.reduce_mean(tf.reduce_sum(term1 * term2, axis=1))

    def bpr_max(self, yhat):

        softmax_scores = self.softmax_neg(yhat)
        ###tf.expand_dims(gpu_diag_wide(yhat),1)  [y1,y1,y1,...,y1][y2,...]
        term1 = - tf.log(
            tf.reduce_sum(tf.sigmoid(tf.expand_dims(gpu_diag_wide(yhat),1) - yhat) * softmax_scores, axis=1) + 1e-24)
        term2 = self.bpr_max_lambda * tf.reduce_sum((yhat ** 2) * softmax_scores, axis=1)
        return tf.reduce_mean(term1 + term2)

        # addtional negative sampling

    def generate_neg_samples(self, pop, length):
        if self.sample_alpha:
            sample = np.searchsorted(pop, np.random.rand(self.n_samples * length))
        else:
            sample = np.random.choice(self.n_items, size=self.n_samples * length)
        if length > 1:
            sample = sample.reshape((length, self.n_samples))
        return sample

    def build_model(self):
        """
        neural network architecture
        """
        if self.user_data:
            self.U = tf.placeholder(tf.int32, [self.batch_size], name='user_id')
        self.X = tf.placeholder(tf.int32, [self.batch_size], name='input')
        self.Y = tf.placeholder(tf.int32, [self.batch_size + self.n_samples], name='output')
        self.state = [tf.placeholder(tf.float32, [self.batch_size, self.rnn_size], name='rnn_state') for _ in
                      range(self.layers)]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('gru_layer'):

            # parameter initialization
            sigma = self.sigma if self.sigma != 0 else np.sqrt(6.0 / (self.n_items + self.rnn_size))
            if self.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_W = tf.get_variable('softmax_w', [self.n_items, self.rnn_size], initializer=initializer)
            softmax_b = tf.get_variable('softmax_b', [self.n_items], initializer=tf.constant_initializer(0.0))
            if self.user_data:
                user_embedding = tf.get_variable('user_embedding', [self.n_users, self.rnn_size],
                                                 initializer=initializer)

            cells = []
            for i in range(self.layers):
                cell = rnn_cell.GRUCell(num_units=self.rnn_size, activation=self.hidden_act)
                # GRU cell in the hidden layer
                cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)
                cells.append(cell)

            multi_cell = rnn_cell.MultiRNNCell(cells)  # multiple GRU cells in hidden layers with dropout

            # input and output
            inputs = tf.nn.embedding_lookup(embedding, self.X)  # self.X is input
            output, state = multi_cell(inputs, tuple(self.state))
            if self.user_data:
                user_output = tf.nn.embedding_lookup(user_embedding, self.U)
                user_concatenated = tf.concat([output, user_output], axis=1)
                output = tf.layers.dense(user_concatenated, self.rnn_size,
                                                           activation='tanh')
            self.final_state = state

        if self.is_training:
            '''
            Use other examples of the minibatch as negative samples.
            '''

            sampled_W = tf.nn.embedding_lookup(softmax_W, self.Y)
            sampled_b = tf.nn.embedding_lookup(softmax_b, self.Y)
            logits = tf.matmul(output, sampled_W, transpose_b=True) + sampled_b
            # sampled_W is transposed before multiplication
            self.yhat = self.final_activation(logits)
            self.cost = self.loss_function(self.yhat)
        else:  # if not training
            logits = tf.matmul(output, softmax_W, transpose_b=True) + softmax_b
            self.yhat = self.final_activation(logits)


        if not self.is_training:
            return

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                              self.decay, staircase=True))

        # set optimizer
        if self.optimizer == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif self.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)

        tvars = tf.trainable_variables()
        gvs = self.optimizer.compute_gradients(self.cost, tvars)
        if self.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

    def init(self, data):
        """
        :param data: csv file
        :return offset_sessions: the offset of each session in data
        :return data: input data after item boosting processing (if dwell time is incorporated)
        """

        data.sort_values([self.session_key, self.time_key], inplace=True)  # Sort data by session and time
        data = self.item_boosting(data)  # item boosting: the number of occurrence is dwell_time_i / threshold_time

        offset_sessions = np.zeros(data[self.session_key].nunique() + 1, dtype=np.int32)
        # nunique: Return the number of unique elements in the object.
        offset_sessions[1:] = data.groupby(self.session_key).size().cumsum()
        # cumsum: Return cumulative sum over a DataFrame or Series axis.
        return offset_sessions, data

    def init_user_data(self, data):
        data.sort_values([self.user_key, self.session_key, self.time_key], inplace=True)
        data.reset_index(drop=True, inplace=True)
        data = self.item_boosting(data)

        offset_sessions = np.r_[
            0, data.groupby([self.user_key, self.session_key], sort=False).size().cumsum().iloc[:-1]]
        user_indptr = np.r_[0, data.groupby(self.user_key, sort=False)[self.session_key].nunique().cumsum().iloc[:-1]]

        return user_indptr, offset_sessions, data

    def item_boosting(self, data):
        if self.dwell_time == 0:  # important!
            return data
        print("[{}] calculating item boosting...".format(dt.now()))
        boosting_count = 0
        data_t = pd.DataFrame(pd.np.empty((self.boosting_memory * len(data), len(data.columns))) * pd.np.nan,
                              columns=data.columns)
        data_values = data.values  # use numpy to accelerate
        data_t_values = data_t.values
        for i in range(0, (len(data_values) - 1)):
            boosting_times = 1
            line1 = data.iloc[i]
            line2 = data.iloc[i + 1]
            if line2[self.session_key] == line1[self.session_key]:
                boosting_times = int(
                    (line2[self.time_key] - line1[self.time_key]) / self.dwell_time) + 1
            for j in range(0, boosting_times):
                data_t_values[boosting_count] = data_values[i]
                boosting_count += 1
        data_t_values[boosting_count] = data_values[-1]
        data_t.loc[0] = data.loc[0]
        data = data_t[:boosting_count + 1]
        print("[{}] item boosting completed, boosting_count: {} (starting from 0), data length: {}".format(dt.now(),
                                                                                                           boosting_count,
                                                                                                           len(data)))
        return data

    def fit(self, data, sample_store=10000000):
        """
        session-parallel mini-batch is used in model fitting
        :param data: csv file
        :return void
        """
        self.error_during_train = False
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)  # the number of items
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                        on=self.item_key, how='inner')

        offset_sessions, data = self.init(data)  # the offset of each session in data, and the data with item boosting

        if self.n_samples:  # additional negative sampling
            pop = data.groupby(self.item_key).size()  # item popularity
            pop = pop[self.itemidmap.index.values].values ** self.sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            if sample_store:
                generate_length = sample_store // self.n_samples
                if generate_length <= 1:
                    sample_store = 0
                    print('No example store was used')
                else:
                    neg_samples = self.generate_neg_samples(pop, generate_length)
                    sample_pointer = 0
            else:
                print('No example store was used')

        print('fitting model...')
        for epoch in range(self.n_epochs):
            epoch_cost = []
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]
            session_idx_arr = np.arange(len(offset_sessions) - 1)
            # arange: Return evenly spaced values within a given interval.
            # renumber each session, starting at 0
            iters = np.arange(self.batch_size)
            maxiter = iters.max()
            start = offset_sessions[session_idx_arr[iters]]  # start index of each session in this batch
            end = offset_sessions[session_idx_arr[iters] + 1]  # end index of each session in this batch
            # session i: [start[i], end[i])
            finished = False  # whether all the sessions has been finished
            while not finished:
                minlen = (end - start).min()  # the minimum length of all sessions
                out_idx = data.ItemIdx.values[start]
                for i in range(minlen - 1):
                    in_idx = out_idx
                    out_idx = data.ItemIdx.values[start + i + 1]

                    # additional negative sampling
                    if self.n_samples:
                        if sample_store:
                            if sample_pointer == generate_length:
                                neg_samples = self.generate_neg_samples(pop, generate_length)
                                sample_pointer = 0
                            sample = neg_samples[sample_pointer]
                            sample_pointer += 1
                        else:
                            sample = self.generate_neg_samples(pop, 1)
                        y = np.hstack([out_idx, sample])
                    else:  # if self.n_samples == 0
                        y = out_idx

                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_idx, self.Y: y}
                    for j in range(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))
                start = start + minlen - 1
                mask = np.arange(len(iters))[(end - start) <= 1]
                for idx in mask:
                    maxiter += 1
                    if maxiter >= len(offset_sessions) - 1:
                        finished = True
                        break
                    iters[idx] = maxiter
                    start[idx] = offset_sessions[session_idx_arr[maxiter]]
                    end[idx] = offset_sessions[session_idx_arr[maxiter] + 1]
                if len(mask) and self.reset_after_session:
                    for i in range(self.layers):
                        state[i][mask] = 0

            avgc = np.mean(epoch_cost)  # average cost
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            self.saver.save(self.sess, '{}/gru-model'.format(self.checkpoint_dir), global_step=epoch)
            # save variables
        print('model fitting complete')

    def fit_user(self, data, sample_store=10000000):
        self.error_during_train = False

        # build data.ItemIdx
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)  # the number of items
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}),
                        on=self.item_key, how='inner')

        # build data.UserIdx
        userids = data[self.user_key].unique()
        self.n_users = len(userids)  # the number of users
        useridmap = pd.Series(data=np.arange(self.n_users), index=userids)
        data = pd.merge(data, pd.DataFrame({self.user_key: userids, 'UserIdx': useridmap[userids].values}),
                        on=self.user_key, how='inner')

        user_indptr, offset_sessions, data = self.init_user_data(data)
        # user index pointer, the offset of each session in data, and the data with item boosting

        if self.n_samples:  # additional negative sampling
            pop = data.groupby(self.item_key).size()  # item popularity
            pop = pop[self.itemidmap.index.values].values ** self.sample_alpha
            pop = pop.cumsum() / pop.sum()
            pop[-1] = 1
            if sample_store:
                generate_length = sample_store // self.n_samples
                if generate_length <= 1:
                    sample_store = 0
                    print('No example store was used')
                else:
                    neg_samples = self.generate_neg_samples(pop, generate_length)
                    sample_pointer = 0
            else:
                print('No example store was used')

        print('fitting model...')
        for epoch in range(self.n_epochs):
            epoch_cost = []
            state = [np.zeros([self.batch_size, self.rnn_size], dtype=np.float32) for _ in range(self.layers)]

            # variables to manage iterations over users
            offset_users = offset_sessions[user_indptr]
            user_idx_arr = np.arange(self.n_users - 1)
            user_iters = np.arange(self.batch_size)
            user_maxiter = user_iters.max()
            user_start = offset_users[user_idx_arr[user_iters]]
            user_end = offset_users[user_idx_arr[user_iters] + 1]

            # variables to manage iterations over sessions
            session_iters = user_indptr[user_iters]
            session_start = offset_sessions[session_iters]
            session_end = offset_sessions[session_iters + 1]

            finished = False
            while not finished:
                session_minlen = (session_end - session_start).min()
                out_item_idx = data.ItemIdx.values[session_start]
                user_idx = data.UserIdx.values[session_start]
                for i in range(session_minlen - 1):
                    in_item_idx = out_item_idx
                    out_item_idx = data.ItemIdx.values[session_start + i + 1]

                    # additional negative sampling
                    if self.n_samples:
                        if sample_store:
                            if sample_pointer == generate_length:
                                neg_samples = self.generate_neg_samples(pop, generate_length)
                                sample_pointer = 0
                            sample = neg_samples[sample_pointer]
                            sample_pointer += 1
                        else:
                            sample = self.generate_neg_samples(pop, 1)
                        y = np.hstack([out_item_idx, sample])
                    else:  # if self.n_samples == 0
                        y = out_item_idx

                    # prepare inputs, targeted outputs and hidden states
                    fetches = [self.cost, self.final_state, self.global_step, self.lr, self.train_op]
                    feed_dict = {self.X: in_item_idx, self.Y: y, self.U: user_idx}
                    for j in range(self.layers):
                        feed_dict[self.state[j]] = state[j]

                    cost, state, step, lr, _ = self.sess.run(fetches, feed_dict)
                    epoch_cost.append(cost)
                    if np.isnan(cost):
                        print(str(epoch) + ':Nan error!')
                        self.error_during_train = True
                        return
                    if step == 1 or step % self.decay_steps == 0:
                        avgc = np.mean(epoch_cost)
                        print('Epoch {}\tStep {}\tlr: {:.6f}\tloss: {:.6f}'.format(epoch, step, lr, avgc))
                session_start = session_start + session_minlen - 1
                session_start_mask = np.arange(len(session_iters))[(session_end - session_start) <= 1]
                for idx in session_start_mask:
                    user_maxiter += 1
                    if user_maxiter >= len(offset_sessions) - 1:
                        finished = True
                        break
                    session_iters[idx] = user_maxiter
                    session_start[idx] = offset_sessions[session_iters[idx]]
                    session_end[idx] = offset_sessions[session_iters[idx] + 1]
                if len(session_start_mask) and self.reset_after_session:
                    for i in range(self.layers):
                        state[i][session_start_mask] = 0

                user_change_mask = np.arange(len(user_iters))[(user_end - session_start <= 0)]
                for idx in user_change_mask:
                    user_maxiter += 1
                    if user_maxiter + 1 >= len(offset_users):
                        finished = True
                        break
                    user_iters[idx] = user_maxiter
                    user_start[idx] = offset_users[user_maxiter]
                    user_end[idx] = offset_users[user_maxiter + 1]
                    session_iters[idx] = user_indptr[user_maxiter]
                    session_start[idx] = offset_sessions[session_iters[idx]]
                    session_end[idx] = offset_sessions[session_iters[idx] + 1]

            avgc = np.mean(epoch_cost)  # average cost
            if np.isnan(avgc):
                print('Epoch {}: Nan error!'.format(epoch, avgc))
                self.error_during_train = True
                return
            self.saver.save(self.sess, '{}/gru-model_user'.format(self.checkpoint_dir), global_step=epoch)
            # save variables
        print('model fitting complete')

    def predict_next_batch(self, session_ids, input_item_ids, itemidmap, batch=50):
        '''
        Gives predicton scores for a selected set of items. Can be used in batch mode to predict for multiple independent events (i.e. events of different sessions) at once and thus speed up evaluation.
        If the session ID at a given coordinate of the session_ids parameter remains the same during subsequent calls of the function, the corresponding hidden state of the network will be kept intact (i.e. that's how one can predict an item to a session).
        If it changes, the hidden state of the network is reset to zeros.
        Parameters
        --------
        session_ids : 1D array
            Contains the session IDs of the events of the batch. Its length must equal to the prediction batch size (batch param).
        input_item_ids : 1D array
            Contains the item IDs of the events of the batch. Every item ID must be must be in the training data of the network. Its length must equal to the prediction batch size (batch param).
        batch : int
            Prediction batch size.
        Returns
        --------
        out : pandas.DataFrame
            Prediction scores for selected items for every event of the batch.
            Columns: events of the batch; rows: items. Rows are indexed by the item IDs.
        '''
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))
        if not self.predict:
            self.current_session = np.ones(batch) * -1
            self.predict = True

        session_change = np.arange(batch)[session_ids != self.current_session]
        if len(session_change) > 0:  # change internal states with session changes
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session = session_ids.copy()

        in_idxs = itemidmap[input_item_ids]
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: in_idxs}
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T  # convert a list into an array
        return pd.DataFrame(data=preds, index=itemidmap.index)

    def predict_next_batch_user(self, session_ids, input_item_ids, input_user_ids, itemidmap, useridmap, batch=50):
        if batch != self.batch_size:
            raise Exception('Predict batch size({}) must match train batch size({})'.format(batch, self.batch_size))

        if self.predict is None or self.predict_batch != batch:
            self.current_session = np.ones(batch) * -1
            self.current_users = np.ones(batch) * -1
            self.predict_batch = batch

        session_change = session_ids != self.current_session
        if len(session_change) > 0:  # change internal states with session changes
            for i in range(self.layers):
                self.predict_state[i][session_change] = 0.0
            self.current_session = session_ids.copy()
            self.current_users = input_user_ids.copy()

        in_idxs = itemidmap[input_item_ids]
        user_idxs = useridmap[input_user_ids]
        fetches = [self.yhat, self.final_state]
        feed_dict = {self.X: in_idxs, self.U: user_idxs}
        for i in range(self.layers):
            feed_dict[self.state[i]] = self.predict_state[i]
        preds, self.predict_state = self.sess.run(fetches, feed_dict)
        preds = np.asarray(preds).T  # convert a list into an array
        return pd.DataFrame(data=preds, index=itemidmap.index)
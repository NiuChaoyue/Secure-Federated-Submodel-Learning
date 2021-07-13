import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from utils import *
from Dice import dice


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, opt_alg='sgd'):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.opt_alg = opt_alg

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)    # [B, H]

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)    # [B, H]
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)    # [B, L, H]

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)    # [B, H]
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)    # [B, L, H]

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)     # [B, 2*H]
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)     # [B, L, 2*H]
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)       # [B, 2*H]

    def build_fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')     # [B, 9*H]
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')    # [B, 200]
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')    # [B, 80]
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')     # [B, 2]
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001   # [B, 2]

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            tf.summary.scalar('loss', self.loss)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            if self.opt_alg == 'adam':
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
                # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                # gradients = self.optimizer.compute_gradients(self.loss)
                # self.train_op = self.optimizer.apply_gradients(gradients, global_step=self.global_step)
            else: # default is sgd
                trainable_params = tf.trainable_variables()
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
                self.train_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def train(self, sess, inps):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.train_op], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cat_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cat_his_batch_ph: inps[4],
            self.mask: inps[5],
            self.target_ph: inps[6],
            self.seq_len_ph: inps[7],
            self.lr: inps[8],
        })
        return loss, accuracy

    def calculate(self, sess, inps):
        probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cat_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cat_his_batch_ph: inps[4],
            self.mask: inps[5],
            self.target_ph: inps[6],
            self.seq_len_ph: inps[7]
        })
        return probs, loss, accuracy

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, self.mask)     # [B, 1, 2*H]
            att_fea = tf.reduce_sum(attention_output, 1)    # [B, 2*H]
            tf.summary.histogram('att_fea', att_fea)
        # [B, H] + [B, 2*H] + [B, 2*H] + [B, 2*H]*[B, 2*H] + [B, 2*H] = [B, 9*H]
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)

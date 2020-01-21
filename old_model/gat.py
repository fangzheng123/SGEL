# encoding: utf-8

import time
import tensorflow as tf
import os
import json
import config_util
import data_util

class GAT(object):
    """
    Graph Attention Network
    """

    def __init__(self, config):
        self.config = config

        self.fea_in = tf.placeholder(dtype=tf.float32, shape=(None, self.config.node_num, self.config.fea_size))
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=(None, self.config.node_num, self.config.node_num))
        self.label_in = tf.placeholder(dtype=tf.int32, shape=(None, self.config.node_num, self.config.class_num))
        self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.is_train = tf.placeholder(dtype=tf.bool, shape=())

        self.create_model()

    def create_model(self):
        """
        train old_model
        :return:
        """
        # shape=(B,N,L)
        self.logits = self.predict(self.attn_drop, self.ffd_drop, hid_units=self.config.hid_units,
                                     n_heads=self.config.n_heads, activation=tf.nn.elu,
                                     residual=self.config.residual)

        # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_in)
        self.loss = self.masked_softmax_cross_entropy(self.logits, self.label_in)

        self.group_acc = self.group_accuracy()

        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.config.l2_coef
        # optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss + lossL2)

    def attn_head(self, fea_in, adj_in, out_sz, activation, in_drop=0.0, coef_drop=0.0, residual=False):
        """
        attention layer in GAT
        :param fea_in:
        :param adj_in:
        :param out_sz:
        :param activation:
        :param in_drop:
        :param coef_drop:
        :param residual:
        :return:
        """
        with tf.name_scope('my_attn'):
            if in_drop != 0.0:
                fea_in = tf.nn.dropout(fea_in, 1.0 - in_drop)

            fea_in = tf.layers.conv1d(fea_in, out_sz, 1, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(fea_in, 1, 1)
            f_2 = tf.layers.conv1d(fea_in, 1, 1)
            logits = f_1 + tf.transpose(f_2, [0, 2, 1])
            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - adj_in)
            coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + mask)

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                fea_in = tf.nn.dropout(fea_in, 1.0 - in_drop)

            vals = tf.matmul(coefs, fea_in)
            ret = tf.contrib.layers.bias_add(vals)

            # residual connection
            if residual:
                if fea_in.shape[-1] != ret.shape[-1]:
                    ret = ret + tf.layers.conv1d(fea_in, ret.shape[-1], 1)  # activation
                else:
                    ret = ret + fea_in

            return activation(ret)  # activation

    def predict(self, attn_drop, ffd_drop, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        """
        predict old_model
        :param attn_drop:
        :param ffd_drop:
        :param hid_units:
        :param n_heads:
        :param activation:
        :param residual:
        :return:
        """
        attns = []
        for _ in range(n_heads[0]):
            attns.append(self.attn_head(self.fea_in, self.adj_in, out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(self.attn_head(h_1, self.adj_in, out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)

        # shape=(B, N, hid_units[-1]*n_heads[-2])
        self.gat_embed = h_1

        out = []
        for i in range(n_heads[-1]):
            out.append(self.attn_head(h_1, self.adj_in, out_sz=self.config.class_num, activation=lambda x: x,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]

        return logits

    def group_accuracy(self):
        """
        calculate group acc for each mention
        :return:
        """
        # shape=(B,C)
        mention_logits = self.logits[:, :, -1][:, :self.config.global_candidate_num]
        self.logits_max = tf.argmax(mention_logits, -1)
        self.label_max = tf.argmax(tf.argmax(self.label_in, -1)[:, :self.config.global_candidate_num], -1)
        correct_prediction = tf.equal(self.logits_max, self.label_max)
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(accuracy_all)

    def masked_softmax_cross_entropy(self, logits, labels_in):
        """
        Softmax cross-entropy loss with masking.
        :param logits:
        :param labels:
        :param mask:
        :return:
        """
        mask_logits = logits[:, :self.config.global_candidate_num, :]
        mask_label = labels_in[:, :self.config.global_candidate_num, :]
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=mask_logits, labels=mask_label)
        return tf.reduce_mean(loss)


class GATProcess(object):

    """
    train, validate GAT old_model
    """

    def __init__(self, config_util, data_util):
        self.config_util = config_util
        self.data_util = data_util
        self.gat_model = GAT(config_util)

    def load_data(self, data_path, data_name="Train, Validate or Test"):
        """
        load Train, Validate or Test data for Gat
        :param data_path:
        :param data_name:
        :return:
        """

        print("Loading {0} data...".format(data_name))
        start_time = time.time()

        # start load data
        fea_array, adj_array, label_array = self.data_util.load_global_data(data_path, self.config_util.global_candidate_num)

        run_time = self.data_util.get_time_dif(start_time)
        print("Time usage:{0}".format(run_time))

        return fea_array, adj_array, label_array

    def evaluate(self, sess, eval_data):
        """
        evaluate GAT old_model
        :param sess:
        :param eval_data:
        :return:
        """
        val_fea, val_adj, val_label = eval_data

        data_len = val_fea.shape[0]
        batch_eval = self.data_util.generate_global_batch(eval_data, self.config_util.gat_batch_size)
        total_loss = 0.0
        total_acc = 0.0
        for fea_batch, adj_batch, label_batch in batch_eval:
            batch_len = fea_batch.shape[0]
            feed_dict = {
                self.gat_model.fea_in: fea_batch,
                self.gat_model.adj_in: adj_batch,
                self.gat_model.label_in: label_batch,
                self.gat_model.is_train: False,
                self.gat_model.attn_drop: 0.0,
                self.gat_model.ffd_drop: 0.0
            }
            loss, _, _, acc = sess.run([self.gat_model.loss, self.gat_model.logits_max, self.gat_model.label_max, self.gat_model.group_acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return float(total_loss) / data_len, float(total_acc) / data_len

    def train(self):
        """
        train gat old_model
        :return:
        """
        # load train data
        train_fea, train_adj, train_label = self.load_data(
            self.config_util.gat_train_path,
            "Train"
        )

        # load validate data
        val_fea, val_adj, val_label = self.load_data(
            self.config_util.gat_validate_path,
            "Validate"
        )

        start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0
        # record the last promotion batch
        last_improved = 0
        early_stop_flag = False

        saver = tf.train.Saver()
        if not os.path.exists(self.config_util.save_gat_dir):
            os.makedirs(self.config_util.save_gat_dir)

        # create session
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        for epoch in range(self.config_util.gat_num_epochs):
            print("Epoch: {0}".format(epoch + 1))

            batch_train = self.data_util.generate_global_batch([train_fea, train_adj, train_label],
                                                                 self.config_util.gat_batch_size,
                                                                 is_random=False)

            for fea_batch, adj_batch, label_batch in batch_train:
                feed_dict = {
                    self.gat_model.fea_in: fea_batch,
                    self.gat_model.adj_in: adj_batch,
                    self.gat_model.label_in: label_batch,
                    self.gat_model.is_train: True,
                    self.gat_model.attn_drop: 0.6,
                    self.gat_model.ffd_drop: 0.6
                }

                # 输出在训练集上的性能
                if total_batch % self.config_util.print_per_batch == 0:
                    loss_train, acc_train = session.run([self.gat_model.loss, self.gat_model.group_acc],
                                                        feed_dict=feed_dict)
                    run_time = self.data_util.get_time_dif(start_time)
                    msg = "Iter: {0:>2}, Train Loss: {1:>4}, Train Group Acc: {2:>4}," \
                          + " Time: {3:>6}"
                    print(msg.format(total_batch, loss_train, acc_train, str(run_time)))

                    if total_batch % 50 == 0:
                        loss_val, acc_val = self.evaluate(session, [val_fea, val_adj, val_label])

                        # 保存最好结果
                        if acc_val > best_acc_val:
                            best_acc_val = acc_val
                            last_improved = total_batch
                            saver.save(sess=session, save_path=self.config_util.save_gat_dir)
                            improved_str = '*'
                        else:
                            improved_str = ''

                        run_time = self.data_util.get_time_dif(start_time)
                        msg = 'Iter: {0:>2}, Train Loss: {1:>4}, Train Acc: {2:>4},' \
                              + ' Val Loss: {3:>4}, Val Acc: {4:>4}, Time: {5} {6}'
                        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, str(run_time), improved_str))

                # 对loss进行优化
                session.run(self.gat_model.optim, feed_dict=feed_dict)
                total_batch += 1

                # 验证集正确率长期不提升，提前结束训练
                if total_batch - last_improved > self.config_util.require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    early_stop_flag = True
                    break

            # early stopping
            if early_stop_flag:
                break

        session.close()

    def predict(self):
        """
        predict gat old_model
        :return:
        """
        print("Test Model...")

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # read old_model
        saver.restore(sess=session, save_path=self.config_util.save_gat_dir)

        # load test data
        test_fea, test_adj, test_label = self.load_data(self.config_util.gat_test_path, "Test")

        loss_test, acc_test = self.evaluate(session, [test_fea, test_adj, test_label])
        print("loss_test:{0}, acc_test:{1}".format(loss_test, acc_test))

        session.close()

    def read_gat_vec(self, data_path):
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # read old_model
        saver.restore(sess=session, save_path=self.config_util.save_gat_dir)

        print(data_path)

        # load test data
        test_fea, test_adj, test_label = self.load_data(data_path, "Test")
        feed_dict = {
            self.gat_model.fea_in: test_fea,
            self.gat_model.adj_in: test_adj,
            self.gat_model.label_in: test_label,
            self.gat_model.is_train: False,
            self.gat_model.attn_drop: 0,
            self.gat_model.ffd_drop: 0
        }

        # (3535, 17, 256)
        gat_embedd = session.run(self.gat_model.gat_embed, feed_dict=feed_dict)
        candidate_gat_embedd = gat_embedd[:, :self.config_util.global_candidate_num, :]
        candidate_gat_embedd = candidate_gat_embedd.tolist()

        gat_vec_path = data_path + "_vec"
        with open(data_path, "r", encoding="utf-8") as data_file:
            with open(gat_vec_path, "w", encoding="utf-8") as gat_vec_file:
                for graph_index, item in enumerate(data_file):
                    item = item.strip()

                    candidate_graph_list = json.loads(item)
                    candidate_graph_list = candidate_graph_list[:self.config_util.global_candidate_num]
                    graph_vec_list = candidate_gat_embedd[graph_index]

                    for candidate_index, candidate_obj in enumerate(candidate_graph_list):
                        candidate_obj["gat_vec"] = graph_vec_list[candidate_index]

                        gat_vec_file.write(json.dumps(candidate_obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    config_util = config_util
    data_util = data_util.DataUtil()

    gat_process = GATProcess(config_util, data_util)
    gat_process.predict()
    # gat_process.read_gat_vec(config_util.gat_test_path)
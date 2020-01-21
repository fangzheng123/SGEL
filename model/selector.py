# encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import config_util
import data_util
import os
import time
import json
import numpy as np
from model_helper import ModelHelper
import networkx as nx

class Selector(object):

    def __init__(self):
        # local
        self.mention_bert_in = tf.placeholder(dtype=tf.float32, shape=(None, config_util.candidate_num, config_util.bert_hidden_size))
        self.candidate_bert_in = tf.placeholder(dtype=tf.float32, shape=(None, config_util.candidate_num, config_util.bert_hidden_size))
        self.candidate_feature_in = tf.placeholder(dtype=tf.float32, shape=(None, config_util.candidate_num, config_util.xgboost_fea_size))
        self.static_score_in = tf.placeholder(dtype=tf.float32, shape=(None, config_util.candidate_num))

        # global
        self.gat_node_in = tf.placeholder(dtype=tf.float32, shape=(None, config_util.node_num, config_util.bert_hidden_size))
        self.adj_in = tf.placeholder(dtype=tf.float32, shape=(None, config_util.node_num, config_util.node_num))
        self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())

        self.label_in = tf.placeholder(dtype=tf.float32, shape=(None, config_util.candidate_num))
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.create_model()

    def create_model(self):
        mention_candidate_bert = tf.concat([self.candidate_bert_in, self.mention_bert_in], axis=-1)
        mlp_output = mention_candidate_bert
        for l_size in ([128, 64, 32][:config_util.combine_mlp_layer] + [2]):
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)
            mlp_output = tf.nn.dropout(mlp_output, keep_prob=self.keep_prob)

        local_output = tf.concat([mlp_output, self.candidate_feature_in], axis=-1)

        # shape=(B,N,hid_units[-1]*n_heads[-1])
        global_output = self.extract_global_feature(self.attn_drop, self.ffd_drop, hid_units=config_util.hid_units,
                                                           n_heads=config_util.n_heads, activation=tf.nn.elu, residual=config_util.residual)

        candidate_score_in = tf.tile(tf.expand_dims(self.static_score_in, -1), [1, 1, 10])
        mlp_output = tf.concat([global_output, local_output, candidate_score_in], axis=-1)

        for l_size in ([64, 32][:config_util.combine_mlp_layer] + [2]):
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)

        batch_loss_list = []
        batch_prob_list = []
        for batch_index in range(config_util.combine_batch_size):
            logits = mlp_output[batch_index, :, :]
            # [C, 2]
            prob = tf.nn.softmax(logits, axis=-1)
            # [C]
            label = self.label_in[batch_index, :]
            loss = self.rank_loss(prob, label)
            batch_loss_list.append(loss)
            batch_prob_list.append(prob)

        self.batch_prob = tf.stack(batch_prob_list)
        self.group_acc = self.group_accuracy()

        self.total_loss = tf.reduce_sum(batch_loss_list)

        self.optim = tf.train.AdamOptimizer(learning_rate=config_util.combine_learning_rate).minimize(self.total_loss)

    def extract_global_feature(self, attn_drop, ffd_drop, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        """
        extract global feature by gat model
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
            attns.append(self.attn_head(self.gat_node_in, self.adj_in, out_sz=hid_units[0], activation=activation,
                                          in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(self.attn_head(h_1, self.adj_in, out_sz=hid_units[i], activation=activation,
                                              in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)

        # h_1: shape=(B, N, hid_units[-1]*n_heads[-1])
        return h_1[:, :config_util.candidate_num, :]

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

    def rank_loss(self, prob, label):
        """

        :param prob:
        :param label:
        :return:
        """
        # label_in:[C]
        # [P]
        pos_indices = tf.where(tf.equal(label, 1))
        # [N]
        neg_indices = tf.where(tf.equal(label, 0))

        # [P,2]
        pos_metric = tf.gather_nd(prob, pos_indices)
        # [N,2]
        neg_metric = tf.gather_nd(prob, neg_indices)

        pos_one_hot = tf.constant([[0, 1]], dtype=tf.float32)
        # [P,2]
        pos_one_hot_labels = tf.tile(pos_one_hot, [tf.shape(pos_indices)[0], 1])
        # [N,2]
        neg_one_hot_labels = tf.tile(pos_one_hot, [tf.shape(neg_indices)[0], 1])

        # only calculate the probability of label 1
        # [P]
        pos_metric = tf.reduce_sum(pos_metric * pos_one_hot_labels, axis=-1)
        # [N]
        neg_metric = tf.reduce_sum(neg_metric * neg_one_hot_labels, axis=-1)

        # do the substraction
        # [P, N]
        pos_metric = tf.tile(tf.expand_dims(pos_metric, 1), [1, tf.shape(neg_indices)[0]])
        # [P, N]
        neg_metric = tf.tile(tf.expand_dims(neg_metric, 0), [tf.shape(pos_indices)[0], 1])
        # [P, N]
        delta = neg_metric - pos_metric

        loss = tf.reduce_mean(tf.nn.relu(config_util.combine_margin + delta))

        return loss

    def group_accuracy(self):
        """
        calculate group acc for each mention
        :return:
        """
        # shape=(B,C)
        mention_logits = self.batch_prob[:, :, -1]
        # shape=(B)
        self.logits_max = tf.argmax(mention_logits, -1)
        label_max = tf.argmax(self.label_in, -1)
        correct_prediction = tf.equal(self.logits_max, label_max)
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(accuracy_all)

class SelectorProcess(object):
    def __init__(self, data_util, combine_model, model_helper):
        self.data_util = data_util
        self.combine_model = combine_model
        self.model_helper = model_helper

    def load_data(self, data_path, data_name="Train, Validate or Test"):
        """
        load data
        :param data_path:
        :param data_name:
        :return:
        """
        print("Loading {0} data...".format(data_name))
        start_time = time.time()

        # start load data
        mention_bert, candidate_bert, candidate_gat, candidate_adj, candidate_fea, candidate_score, label = self.model_helper.load_selector_data(data_path)

        run_time = self.data_util.get_time_dif(start_time)
        print("Time usage:{0}".format(run_time))

        data_batch = [mention_bert, candidate_bert, candidate_gat, candidate_adj, candidate_fea, candidate_score, label]

        return data_batch

    def update_gat_input(self, choose_action_list, source_gat_input):
        """

        :param choose_action_list: shape=(S, B)
        :param source_gat_input: shape=(B,N,H)
        :return:
        """
        next_mention_index = len(choose_action_list)

        next_gat_mention = source_gat_input[:, next_mention_index*config_util.candidate_num:(next_mention_index+1)*config_util.candidate_num, :]
        other_gat_mention = np.concatenate([source_gat_input[:, :next_mention_index*config_util.candidate_num, :],
                                            source_gat_input[:, (next_mention_index+1)*config_util.candidate_num:, :]], axis=1)
        next_gat_input = np.concatenate([next_gat_mention, other_gat_mention], axis=1)

        # shape=(B,S)
        select_action_np = np.stack(choose_action_list, axis=1).tolist()

        all_adj_list = []
        for seq_action_list in select_action_np:
            choose_index_list = []
            for i, choose_index in enumerate(seq_action_list):
                choose_index_list.append((i+1)*config_util.candidate_num + choose_index)

            disam_index_list = [index for index in range((next_mention_index+1)*config_util.candidate_num, config_util.candidate_num * config_util.sequence_len)]

            adj_index_list = []
            adj_index_list.extend(choose_index_list)
            adj_index_list.extend(disam_index_list)

            index_list_dict = {}
            for i in range(config_util.candidate_num):
                index_list_dict[i] = adj_index_list
            for i in range(config_util.candidate_num * config_util.sequence_len):
                if i not in adj_index_list and i not in index_list_dict:
                    index_list_dict[i] = [i]

            # adjacency matrix(N * N)
            graph_adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(index_list_dict)).todense()

            all_adj_list.append(graph_adj_matrix)

        # next_gat_input:shape=(B,N,H), all_adj_np:shape=(B,N,N)
        return next_gat_input, np.array(all_adj_list)

    def evaluate(self, sess, eval_data, is_cal_loss=True):
        """
        evaluate combine model
        :param sess:
        :param eval_data:
        :param is_cal_loss: when test data, one group may not contain label 0 and 1, so the rank loss can't calculate
        :return:
        """
        val_mention_bert, val_candidate_bert, val_candidate_gat, var_candidate_adj, \
        val_candidate_fea, val_candidate_score, val_label = eval_data

        data_len = val_mention_bert.shape[0]
        batch_eval = self.model_helper.selector_batch_iter(eval_data, config_util.combine_batch_size)
        total_loss = 0.0
        total_acc = 0.0
        for mention_bert_batch, candidate_bert_batch, candidate_gat_batch, candidate_adj_batch, \
            candidate_fea_batch, candidate_score_batch, label_batch in batch_eval:
            batch_len = len(mention_bert_batch)

            current_gat = candidate_gat_batch
            current_adj = candidate_adj_batch
            choose_action_list = []
            for seq_index in range(config_util.sequence_len):
                current_mention_bert = mention_bert_batch[:, seq_index, :, :]
                current_entity_bert = candidate_bert_batch[:, seq_index, :, :]
                current_fea = candidate_fea_batch[:, seq_index, :, :]
                current_score = candidate_score_batch[:, seq_index, :]
                current_label = label_batch[:, seq_index, :]

                feed_dict = {
                    self.combine_model.mention_bert_in: current_mention_bert,
                    self.combine_model.candidate_bert_in: current_entity_bert,
                    self.combine_model.gat_node_in: current_gat,
                    self.combine_model.adj_in: current_adj,
                    self.combine_model.candidate_feature_in: current_fea,
                    self.combine_model.static_score_in: current_score,
                    self.combine_model.label_in: current_label,
                    self.combine_model.attn_drop: 0,
                    self.combine_model.ffd_drop: 0,
                    self.combine_model.keep_prob: 1.0
                }
                logits_max = sess.run(self.combine_model.logits_max, feed_dict=feed_dict)
                # update gat input
                current_choose_action = logits_max.tolist()
                # shape=(S,B)
                choose_action_list.append(current_choose_action)
                if seq_index < config_util.sequence_len - 1:
                    next_gat_input, next_adj_input = self.update_gat_input(np.array(choose_action_list), candidate_gat_batch)
                    current_gat = next_gat_input
                    current_adj = next_adj_input

                if is_cal_loss:
                    loss, acc = sess.run([self.combine_model.total_loss, self.combine_model.group_acc], feed_dict=feed_dict)
                    total_loss += loss * batch_len
                else:
                    acc = sess.run(self.combine_model.group_acc, feed_dict=feed_dict)

                total_acc += acc * batch_len

        return total_loss / (data_len * config_util.sequence_len), total_acc / (data_len * config_util.sequence_len)

    def train(self):
        """
        train entity selector
        :return:
        """
        # load train data
        all_train_data = self.load_data(config_util.combine_train_path, data_name="Train")

        # load validate data
        all_val_data = self.load_data(config_util.combine_val_path, data_name="Validate")

        start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0
        # record the last promotion batch
        last_improved = 0
        early_stop_flag = False

        saver = tf.train.Saver()
        if not os.path.exists(config_util.save_combine_dir):
            os.makedirs(config_util.save_combine_dir)

        # create session
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        for epoch in range(config_util.combine_num_epochs):
            print("Epoch: {0}".format(epoch + 1))
            batch_train = self.model_helper.selector_batch_iter(all_train_data, config_util.combine_batch_size,
                                                                 is_random=False)

            # mention_bert:shape=(B,S,C,H), candidate_gat:shape=(B,N,H)
            for mention_bert_batch, candidate_bert_batch, candidate_gat_batch, candidate_adj_batch, \
                candidate_fea_batch, candidate_score_batch, label_batch in batch_train:

                current_gat = candidate_gat_batch
                current_adj = candidate_adj_batch
                choose_action_list = []
                for seq_index in range(config_util.sequence_len):
                    current_mention_bert = mention_bert_batch[:, seq_index, :, :]
                    current_entity_bert = candidate_bert_batch[:, seq_index, :, :]
                    current_fea = candidate_fea_batch[:, seq_index, :, :]
                    current_score = candidate_score_batch[:, seq_index, :]
                    current_label = label_batch[:, seq_index, :]

                    feed_dict = {
                        self.combine_model.mention_bert_in: current_mention_bert,
                        self.combine_model.candidate_bert_in: current_entity_bert,
                        self.combine_model.gat_node_in: current_gat,
                        self.combine_model.adj_in: current_adj,
                        self.combine_model.candidate_feature_in: current_fea,
                        self.combine_model.static_score_in: current_score,
                        self.combine_model.label_in: current_label,
                        self.combine_model.attn_drop: 0.6,
                        self.combine_model.ffd_drop: 0.6,
                        self.combine_model.keep_prob: 0.8
                    }

                    logits_max, loss_train, acc_train = session.run([self.combine_model.logits_max,
                                                                     self.combine_model.total_loss,
                                                                     self.combine_model.group_acc],
                                                                    feed_dict=feed_dict)

                    run_time = self.data_util.get_time_dif(start_time)
                    msg = "Iter: {0:>2}, Train Loss: {1:>4}, Train Group Acc: {2:>4}," \
                          + " Time: {3:>4}"
                    print(msg.format(total_batch, loss_train, acc_train, str(run_time)))

                    # optimize loss
                    session.run(self.combine_model.optim, feed_dict=feed_dict)
                    total_batch += 1

                    # update gat input
                    current_choose_action = logits_max.tolist()
                    # shape=(S,B)
                    choose_action_list.append(current_choose_action)

                    if seq_index < config_util.sequence_len - 1:
                        next_gat_input, next_adj_input = self.update_gat_input(np.array(choose_action_list), candidate_gat_batch)
                        current_gat = next_gat_input
                        current_adj = next_adj_input

                if total_batch % (config_util.sequence_len*10) == 0:
                    loss_val, acc_val = self.evaluate(session, all_val_data)
                    # save best result
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=config_util.save_combine_dir)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    run_time = self.data_util.get_time_dif(start_time)
                    msg = 'Iter: {0:>2}, Val Loss: {1:>4}, Val Acc: {2:>4}, Time: {3} {4}'
                    print(msg.format(total_batch, loss_val, acc_val, str(run_time), improved_str))
                    print("\n")

                # auto-stopping
                if total_batch - last_improved > config_util.require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    early_stop_flag = True
                    break

            # early stopping
            if early_stop_flag:
                break

        session.close()

    def predict(self):
        """
        predict combine model
        :return:
        """
        print("Test Model...")

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # read model
        saver.restore(sess=session, save_path=config_util.save_combine_dir)

        # load test data
        all_test_data = self.load_data(config_util.combine_test_path, "Test")

        loss_test, acc_test = self.evaluate(session, all_test_data, is_cal_loss=False)
        print("loss_test:{0}, acc_test:{1}".format(loss_test, acc_test))

        session.close()

    def export_predict_result(self):
        """
        export model predict result, set batch_size = 1
        :return:
        """
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # read model
        saver.restore(sess=session, save_path=config_util.save_combine_dir)

        # load test data
        all_test_data = self.load_data(config_util.combine_test_path, "Test")

        batch_test = self.model_helper.selector_batch_iter(all_test_data, config_util.combine_batch_size)

        all_data_len = len(all_test_data[0])
        all_batch_len = 0
        all_pred_list = []
        # mention_bert:shape=(B,S,C,H), candidate_gat:shape=(B,N,H)
        for mention_bert_batch, candidate_bert_batch, candidate_gat_batch, candidate_adj_batch, \
            candidate_fea_batch, candidate_score_batch, label_batch in batch_test:

            current_gat = candidate_gat_batch
            current_adj = candidate_adj_batch
            choose_action_list = []
            for seq_index in range(config_util.sequence_len):
                current_mention_bert = mention_bert_batch[:, seq_index, :, :]
                current_entity_bert = candidate_bert_batch[:, seq_index, :, :]
                current_fea = candidate_fea_batch[:, seq_index, :, :]
                current_score = candidate_score_batch[:, seq_index, :]
                current_label = label_batch[:, seq_index, :]

                feed_dict = {
                    self.combine_model.mention_bert_in: current_mention_bert,
                    self.combine_model.candidate_bert_in: current_entity_bert,
                    self.combine_model.gat_node_in: current_gat,
                    self.combine_model.adj_in: current_adj,
                    self.combine_model.candidate_feature_in: current_fea,
                    self.combine_model.static_score_in: current_score,
                    self.combine_model.label_in: current_label,
                    self.combine_model.attn_drop: 0,
                    self.combine_model.ffd_drop: 0,
                    self.combine_model.keep_prob: 1.0
                }
                all_batch_len += mention_bert_batch.shape[0]
                logits_max = session.run(self.combine_model.logits_max, feed_dict=feed_dict)

                all_pred_list.extend(logits_max.tolist())

                # update gat input
                current_choose_action = logits_max.tolist()
                # shape=(S,B)
                choose_action_list.append(current_choose_action)

                if seq_index < config_util.sequence_len - 1:
                    next_gat_input, next_adj_input = self.update_gat_input(np.array(choose_action_list), candidate_gat_batch)
                    current_gat = next_gat_input
                    current_adj = next_adj_input

        if len(all_pred_list) == all_data_len * config_util.sequence_len:
            filter_pred_list = []

            # filter padding mention
            group_item_dict = self.data_util.get_group_list(config_util.combine_test_path)
            file_group_dict = self.data_util.get_file_group(config_util.combine_test_path)

            seq_index = 0
            for mention_file, group_list in file_group_dict.items(): 
                seq_num = int((len(group_list) - 1) / config_util.sequence_len) + 1

                for i in range(seq_num):
                    seq_group_list = group_list[i*config_util.sequence_len: (i+1)*config_util.sequence_len]
                    pred_group_list = all_pred_list[seq_index * config_util.sequence_len:
                                                    seq_index * config_util.sequence_len + len(seq_group_list)]

                    pred_item_list = [group_item_dict[group_id][pred_group_list[index]] for index, group_id in enumerate(seq_group_list)]

                    filter_pred_list.extend(pred_item_list)

                    seq_index += 1

            print(len(filter_pred_list))
            predict_path = config_util.combine_test_path + "_predict"
            with open(predict_path, "w", encoding="utf-8") as predict_file:
                for item in filter_pred_list:
                    predict_file.write(item + "\n")
        else:
            print("Error...........{0}, {1}".format(all_data_len, len(all_pred_list)))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_util = data_util.DataUtil()
    model_helper = ModelHelper(data_util)

    combine_model = Selector()
    model_process = SelectorProcess(data_util, combine_model, model_helper)

    model_process.export_predict_result()

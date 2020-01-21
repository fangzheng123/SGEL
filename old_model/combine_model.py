# encoding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim
import config_util
import data_util
import os
import time
import json

class EntitySelector(object):

    def __init__(self, config):
        self.config = config

        self.mention_bert_in = tf.placeholder(dtype=tf.float32, shape=(None, self.config.global_candidate_num, self.config.bert_hidden_size))
        self.candidate_bert_in = tf.placeholder(dtype=tf.float32, shape=(None, self.config.global_candidate_num, self.config.bert_hidden_size))
        self.candidate_gat_in = tf.placeholder(dtype=tf.float32, shape=(None, self.config.global_candidate_num, self.config.gat_hidden_size))
        self.candidate_feature_in = tf.placeholder(dtype=tf.float32, shape=(None, self.config.global_candidate_num, self.config.xgboost_fea_size))
        self.label_in = tf.placeholder(dtype=tf.float32, shape=(None, self.config.global_candidate_num))
        self.static_score_in = tf.placeholder(dtype=tf.float32, shape=(None, self.config.global_candidate_num))
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.create_model()

    def create_model(self):
        mention_candidate_bert = tf.concat([self.candidate_bert_in, self.mention_bert_in], axis=-1)
        mlp_output = mention_candidate_bert
        for l_size in [256, 128, 64, 32][:self.config.combine_mlp_layer]:
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)
            mlp_output = tf.nn.dropout(mlp_output, keep_prob=self.keep_prob)

        mention_candidate_input = tf.concat([self.candidate_gat_in, mlp_output], axis=-1)

        mlp_output = mention_candidate_input
        for l_size in ([64, 32][:self.config.combine_mlp_layer] + [2]):
            mlp_output = slim.fully_connected(mlp_output, l_size, activation_fn=tf.nn.softplus)
            mlp_output = tf.nn.dropout(mlp_output, keep_prob=self.keep_prob)

        candidate_score_in = tf.tile(tf.expand_dims(self.static_score_in, -1), [1, 1, 50])
        mlp_output = tf.concat([mlp_output, self.candidate_feature_in, candidate_score_in], axis=-1)
        mlp_output = slim.fully_connected(mlp_output, 2, activation_fn=tf.nn.softplus)

        batch_loss_list = []
        batch_prob_list = []
        for batch_index in range(self.config.combine_batch_size):
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

        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.combine_learning_rate).minimize(self.total_loss)

    def rank_loss(self, prob, label):
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

        loss = tf.reduce_mean(tf.nn.relu(self.config.combine_margin + delta))

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


class ModelProcess(object):
    def __init__(self, config_util, data_util):
        self.config_util = config_util
        self.data_util = data_util
        self.combine_model = EntitySelector(config_util)

    def load_data(self, data_path, is_train=True, data_name="Train, Validate or Test"):
        """
        load data
        :param data_path:
        :param data_name:
        :return:
        """
        print("Loading {0} data...".format(data_name))
        start_time = time.time()

        # start load data
        mention_bert, candidate_bert, candidate_gat, candidate_fea, candidate_score, label = self.data_util.load_combine_data(data_path, is_train)

        run_time = self.data_util.get_time_dif(start_time)
        print("Time usage:{0}".format(run_time))

        return mention_bert, candidate_bert, candidate_gat, candidate_fea, candidate_score, label

    def evaluate(self, sess, eval_data, is_cal_loss=True):
        """
        evaluate combine old_model
        :param sess:
        :param eval_data:
        :param is_cal_loss: when test data, one group may not contain label 0 and 1, so the rank loss can't calculate
        :return:
        """
        val_mention_bert, val_candidate_bert, val_candidate_gat, \
        val_candidate_fea, val_candidate_score, val_label = eval_data

        data_len = val_mention_bert.shape[0]
        batch_eval = self.data_util.generate_combine_batch(eval_data, self.config_util.combine_batch_size)
        total_loss = 0.0
        total_acc = 0.0
        for mention_bert_batch, candidate_bert_batch, candidate_gat_batch, \
            candidate_fea_batch, candidate_score_batch, label_batch in batch_eval:
            batch_len = mention_bert_batch.shape[0]
            feed_dict = {
                self.combine_model.mention_bert_in: mention_bert_batch,
                self.combine_model.candidate_bert_in: candidate_bert_batch,
                self.combine_model.candidate_gat_in: candidate_gat_batch,
                self.combine_model.candidate_feature_in: candidate_fea_batch,
                self.combine_model.static_score_in: candidate_score_batch,
                self.combine_model.label_in: label_batch,
                self.combine_model.keep_prob: 1.0
            }
            if is_cal_loss:
                loss, acc = sess.run([self.combine_model.total_loss, self.combine_model.group_acc], feed_dict=feed_dict)
                total_loss += loss * batch_len
            else:
                acc = sess.run(self.combine_model.group_acc, feed_dict=feed_dict)

            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len

    def train(self):
        """
        train entity selector
        :return:
        """
        # load train data
        train_mention_bert, train_candidate_bert, train_candidate_gat, \
        train_candidate_fea, train_candidate_score, train_label = self.load_data(self.config_util.combine_train_path,"Train")

        # load validate data
        val_mention_bert, val_candidate_bert, val_candidate_gat, \
        val_candidate_fea, val_candidate_score, val_label = self.load_data(self.config_util.combine_validate_path, "Validate")

        start_time = time.time()
        total_batch = 0
        best_acc_val = 0.0
        # record the last promotion batch
        last_improved = 0
        early_stop_flag = False

        saver = tf.train.Saver()
        if not os.path.exists(self.config_util.save_combine_dir):
            os.makedirs(self.config_util.save_combine_dir)

        # create session
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        for epoch in range(self.config_util.combine_num_epochs):
            print("Epoch: {0}".format(epoch + 1))

            batch_train = self.data_util.generate_combine_batch([train_mention_bert, train_candidate_bert,
                                                                train_candidate_gat, train_candidate_fea,
                                                                 train_candidate_score, train_label],
                                                                 self.config_util.combine_batch_size,
                                                                 is_random=False)

            for mention_bert_batch, candidate_bert_batch, candidate_gat_batch, \
                candidate_fea_batch, candidate_score_batch, label_batch in batch_train:
                feed_dict = {
                    self.combine_model.mention_bert_in: mention_bert_batch,
                    self.combine_model.candidate_bert_in: candidate_bert_batch,
                    self.combine_model.candidate_gat_in: candidate_gat_batch,
                    self.combine_model.candidate_feature_in: candidate_fea_batch,
                    self.combine_model.static_score_in: candidate_score_batch,
                    self.combine_model.label_in: label_batch,
                    self.combine_model.keep_prob: 0.8
                }

                # 输出在训练集上的性能
                if total_batch % self.config_util.print_per_batch == 0:
                    loss_train, acc_train = session.run([self.combine_model.total_loss, self.combine_model.group_acc],
                                                        feed_dict=feed_dict)
                    run_time = self.data_util.get_time_dif(start_time)
                    msg = "Iter: {0:>2}, Train Loss: {1:>4}, Train Group Acc: {2:>4}," \
                          + " Time: {3:>6}"
                    print(msg.format(total_batch, loss_train, acc_train, str(run_time)))

                    if total_batch % 100 == 0:
                        loss_val, acc_val = self.evaluate(session, [val_mention_bert, val_candidate_bert,
                                                                    val_candidate_gat, val_candidate_fea,
                                                                    val_candidate_score, val_label])

                        # 保存最好结果
                        if acc_val > best_acc_val:
                            best_acc_val = acc_val
                            last_improved = total_batch
                            saver.save(sess=session, save_path=self.config_util.save_combine_dir)
                            improved_str = '*'
                        else:
                            improved_str = ''

                        run_time = self.data_util.get_time_dif(start_time)
                        msg = 'Iter: {0:>2}, Train Loss: {1:>4}, Train Acc: {2:>4},' \
                              + ' Val Loss: {3:>4}, Val Acc: {4:>4}, Time: {5} {6}'
                        print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, str(run_time), improved_str))

                # 对loss进行优化
                session.run(self.combine_model.optim, feed_dict=feed_dict)
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
        predict combine old_model
        :return:
        """
        print("Test Model...")

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # read old_model
        saver.restore(sess=session, save_path=self.config_util.save_combine_dir)

        # load test data
        test_mention_bert, test_candidate_bert, test_candidate_gat, \
        test_candidate_fea, test_candidate_score, test_label = self.load_data(self.config_util.combine_test_path, False, "Test")

        loss_test, acc_test = self.evaluate(session, [test_mention_bert, test_candidate_bert, test_candidate_gat,
                                                      test_candidate_fea, test_candidate_score, test_label], is_cal_loss=False)
        print("loss_test:{0}, acc_test:{1}".format(loss_test, acc_test))

        session.close()

    def export_predict_result(self):
        """
        export old_model predict result
        :return:
        """
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # read old_model
        saver.restore(sess=session, save_path=self.config_util.save_combine_dir)

        # load test data
        test_mention_bert, test_candidate_bert, test_candidate_gat, \
        test_candidate_fea, test_candidate_score, test_label = self.load_data(config_util.combine_test_path, False, "Test")

        batch_test = self.data_util.generate_combine_batch([test_mention_bert, test_candidate_bert, test_candidate_gat,
                                                            test_candidate_fea, test_candidate_score, test_label],
                                                           self.config_util.combine_batch_size)

        all_data_len = len(test_mention_bert)
        all_batch_len = 0
        all_pred_list = []
        for mention_bert_batch, candidate_bert_batch, candidate_gat_batch, \
            candidate_fea_batch, candidate_score_batch, label_batch in batch_test:
            feed_dict = {
                self.combine_model.mention_bert_in: mention_bert_batch,
                self.combine_model.candidate_bert_in: candidate_bert_batch,
                self.combine_model.candidate_gat_in: candidate_gat_batch,
                self.combine_model.candidate_feature_in: candidate_fea_batch,
                self.combine_model.static_score_in: candidate_score_batch,
                self.combine_model.label_in: label_batch,
                self.combine_model.keep_prob: 1.0
            }
            all_batch_len += mention_bert_batch.shape[0]
            logits_max = session.run(self.combine_model.logits_max, feed_dict=feed_dict)

            # the last batch, filter padding mention
            if all_batch_len > all_data_len:
                logits_max = logits_max[all_batch_len-all_data_len:]

            all_pred_list.extend(logits_max.tolist())

        if len(all_pred_list) == all_data_len:
            mention_obj_list = []

            with open(config_util.combine_test_path, "r", encoding="utf-8") as combine_test_file:
                for index, item in enumerate(combine_test_file):
                    item = item.strip()

                    mention_obj = json.loads(item)
                    mention_obj["combine_model_predict_candidate"] = all_pred_list[index]
                    mention_obj_list.append(mention_obj)

            predict_path = config_util.combine_test_path + "_predict"
            with open(predict_path, "w", encoding="utf-8") as predict_file:
                for item in mention_obj_list:
                    predict_file.write(json.dumps(item) + "\n")
        else:
            print("Error...........{0}, {1}".format(all_data_len, len(all_pred_list)))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    config_util = config_util
    data_util = data_util.DataUtil()
    model_process = ModelProcess(config_util, data_util)

    model_process.predict()
    model_process.export_predict_result()

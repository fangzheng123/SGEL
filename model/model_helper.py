# encoding: utf-8

import config_util
import json
from data_util import DataUtil
import numpy as np
import networkx as nx

class ModelHelper(object):
    def __init__(self, data_util):
        self.data_util = data_util

    def load_selector_data(self, rank_mention_path):
        """
        load data for selector model
        :param rank_mention_path:
        :return:
        """
        group_item_dict = self.data_util.get_group_list(rank_mention_path)
        file_group_dict = self.data_util.get_file_group(rank_mention_path)

        candidate_index_list = [index for index in range(config_util.candidate_num*config_util.sequence_len)]
        index_list_dict = {}
        for i in range(config_util.candidate_num):
            index_list_dict[i] = candidate_index_list[config_util.candidate_num:]
        # Adjacency matrix(N * N)
        graph_adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(index_list_dict)).todense()

        mention_num = 0
        all_group = 0
        all_mention_bert_list = []
        all_entity_bert_list = []
        all_entity_gat_list = []
        all_entity_adj_list = []
        all_fea_list = []
        all_score_list = []
        all_label_list = []
        for mention_file, group_list in file_group_dict.items():
            all_group += len(group_list)

            for group_id in group_list:
                item_list = group_item_dict[group_id]

                tmp_mention_bert_list = []
                tmp_entity_bert_list = []
                tmp_entity_gat_list = []
                tmp_entity_adj_list = []
                tmp_fea_list = []
                tmp_score_list = []
                tmp_label_list = []
                item_num = len(item_list)

                for item in item_list:
                    item = item.strip()

                    group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                    mention_obj = json.loads(mention_str)
                    entity_obj = json.loads(entity_str)
                    fea_obj = json.loads(fea_str)
                    feature = [val for name, val in fea_obj.items()][:config_util.xgboost_fea_size]
                    xgb_score = (5 - entity_obj["xgboost_rank_position"]) * 0.2

                    # shape = (S, H)
                    tmp_mention_bert_list.append(mention_obj["vec"])
                    tmp_entity_bert_list.append(entity_obj["vec"])
                    tmp_fea_list.append(feature)
                    tmp_score_list.append(xgb_score)
                    tmp_label_list.append(int(label_str))

                # entity padding
                if item_num != config_util.candidate_num:
                    pad_mention_bert = tmp_mention_bert_list[-1]
                    pad_entity_bert = tmp_entity_bert_list[-1]
                    pad_fea = tmp_fea_list[-1]
                    pad_score = tmp_score_list[-1]
                    pad_label = tmp_label_list[-1]
                    for i in range(config_util.candidate_num - item_num):
                        tmp_mention_bert_list.append(pad_mention_bert)
                        tmp_entity_bert_list.append(pad_entity_bert)
                        tmp_fea_list.append(pad_fea)
                        tmp_score_list.append(pad_score)
                        tmp_label_list.append(pad_label)

                # label to one hot, shape=(C,class_num)
                # candidate_label_np = np.eye(self.local_config.class_num)[candidate_label_list]

                all_mention_bert_list.append(tmp_mention_bert_list)
                all_entity_bert_list.append(tmp_entity_bert_list)
                all_fea_list.append(tmp_fea_list)
                all_score_list.append(tmp_score_list)
                all_label_list.append(tmp_label_list)

                mention_num += 1
                # construct gat input
                if mention_num % config_util.sequence_len == 0:
                    for group_entity_bert_list in all_entity_bert_list[mention_num-config_util.sequence_len:mention_num]:
                        tmp_entity_gat_list.extend(group_entity_bert_list)

                    # build graph and adj
                    all_entity_gat_list.append(tmp_entity_gat_list)
                    all_entity_adj_list.append(graph_adj_matrix)

            # mention padding
            if len(group_list) % config_util.sequence_len != 0:
                remainder = len(group_list) % config_util.sequence_len

                pad_mention_bert = all_mention_bert_list[-1]
                pad_entity_bert = all_entity_bert_list[-1]
                pad_fea = all_fea_list[-1]
                pad_score = all_score_list[-1]
                pad_label = all_label_list[-1]
                for i in range(config_util.sequence_len - remainder):
                    all_mention_bert_list.append(pad_mention_bert)
                    all_entity_bert_list.append(pad_entity_bert)
                    all_fea_list.append(pad_fea)
                    all_score_list.append(pad_score)
                    all_label_list.append(pad_label)

                    mention_num += 1

                # construct gat input
                tmp_entity_gat_list = []
                if mention_num % config_util.sequence_len == 0:
                    for group_entity_bert_list in all_entity_bert_list[mention_num - config_util.sequence_len:mention_num]:
                        tmp_entity_gat_list.extend(group_entity_bert_list)

                    all_entity_gat_list.append(tmp_entity_gat_list)
                    all_entity_adj_list.append(graph_adj_matrix)

        print(len(all_mention_bert_list), len(all_entity_bert_list), len(all_entity_gat_list), len(all_entity_adj_list),
              len(all_fea_list), len(all_score_list), len(all_label_list))

        # mention_bert_array:shape=(B,C,H), entity_gat_array:shape=(B/S, C, H)
        mention_bert_array = np.array(all_mention_bert_list)
        entity_bert_array = np.array(all_entity_bert_list)
        entity_gat_array = np.array(all_entity_gat_list)
        entity_adj_array = np.array(all_entity_adj_list)
        fea_array = np.array(all_fea_list)
        score_array = np.array(all_score_list)
        label_array = np.array(all_label_list)

        # shape=(B,S,C,H)
        mention_bert_array = np.reshape(mention_bert_array, (-1, config_util.sequence_len, config_util.candidate_num,
                                                                config_util.bert_hidden_size))
        entity_bert_array = np.reshape(entity_bert_array, (-1, config_util.sequence_len, config_util.candidate_num,
                                                               config_util.bert_hidden_size))
        fea_array = np.reshape(fea_array, (-1, config_util.sequence_len, config_util.candidate_num, config_util.xgboost_fea_size))
        score_array = np.reshape(score_array, (-1, config_util.sequence_len, config_util.candidate_num))
        label_array = np.reshape(label_array, (-1, config_util.sequence_len, config_util.candidate_num))

        return mention_bert_array, entity_bert_array, entity_gat_array, \
               entity_adj_array, fea_array, score_array, label_array

    def selector_batch_iter(self, data_list, batch_size, is_random=False):
        """
        build batch data for selector
        :param data_list:
        :param batch_size:
        :param is_random:
        :return:
        """
        # mention_bert:shape=(B,S,C,H), candidate_gat:shape=(B,C,H)
        mention_bert, candidate_bert, candidate_gat, candidate_adj, candidate_fea, candidate_score, label = data_list

        all_data_num = mention_bert.shape[0]
        batch_num = int((all_data_num - 1) / batch_size) + 1

        # shuffle
        if is_random:
            indices = np.random.permutation(np.arange(all_data_num))
            mention_bert = mention_bert[indices]
            candidate_bert = candidate_bert[indices]
            candidate_gat = candidate_gat[indices]
            candidate_adj = candidate_adj[indices]
            candidate_fea = candidate_fea[indices]
            candidate_score = candidate_score[indices]
            label = label[indices]

        for i in range(batch_num):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, all_data_num)
            if end_id == all_data_num:
                yield mention_bert[max(end_id - batch_size, 0):end_id], \
                      candidate_bert[max(end_id - batch_size, 0):end_id], \
                      candidate_gat[max(end_id - batch_size, 0):end_id], \
                      candidate_adj[max(end_id - batch_size, 0):end_id], \
                      candidate_fea[max(end_id - batch_size, 0):end_id], \
                      candidate_score[max(end_id - batch_size, 0):end_id], \
                      label[max(end_id - batch_size, 0):end_id]
            else:
                yield mention_bert[start_id:end_id], candidate_bert[start_id:end_id], \
                      candidate_gat[start_id:end_id], candidate_adj[start_id:end_id], \
                      candidate_fea[start_id:end_id], candidate_score[start_id:end_id], label[start_id:end_id]


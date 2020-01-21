# encoding: utf-8

import config_util
import json
from data_util import DataUtil

class LocalRanker(object):
    """
    rank mention
    """
    def __init__(self, data_util):
        self.data_util = data_util

    def rank_mention(self, candidate_path, rank_mention_path, is_train=False):
        """
        sort mentions according to the prediction results of the local model
        :param data_path:
        :param is_train: is train data
        :return:
        """
        print('Rank Mention...')

        group_item_dict = self.data_util.get_group_list(candidate_path)
        file_group_dict = self.data_util.get_file_group(candidate_path)
        pre_group_list = []
        next_group_list = []
        rank_group_list = []
        rank_file_group_dict = {}
        for group_id, item_list in group_item_dict.items():
            # the first position in xgboost
            group_str, label_str, fea_str, mention_str, entity_str = item_list[0].split("\t")

            group_id = int(group_str)
            label = int(label_str)
            fea = json.loads(fea_str)
            mention_obj = json.loads(mention_str)
            mention_file = mention_obj["mention_file"]

            current_file_group_list = file_group_dict[mention_file]

            if is_train:
                if label == 1:
                    pre_group_list.append(group_id)
                else:
                    next_group_list.append(group_id)
            else:
                if fea["same_candidate_word_num"] > 0:
                    pre_group_list.append(group_id)
                else:
                    next_group_list.append(group_id)

            if len(pre_group_list) + len(next_group_list) == config_util.sequence_len or \
                    len(rank_group_list) + config_util.sequence_len > len(current_file_group_list):
                rank_group_list.extend(pre_group_list.copy())
                rank_group_list.extend(next_group_list.copy())
                pre_group_list = []
                next_group_list = []

            if len(rank_group_list) == len(current_file_group_list):
                rank_file_group_dict[mention_file] = rank_group_list.copy()
                rank_group_list = []

        print("source mention num:{0}, rank mention num: {1}".format(sum([len(groups) for _, groups in file_group_dict.items()]),
                                                                     sum([len(groups) for _, groups in rank_file_group_dict.items()])))

        with open(rank_mention_path, "w", encoding="utf-8") as rank_mention_file:
            for _, group_list in rank_file_group_dict.items():
                for group_id in group_list:
                    item_list = group_item_dict[group_id]
                    for item in item_list:
                        rank_mention_file.write(item + "\n")

if __name__ == "__main__":

    data_util = DataUtil()

    local_ranker = LocalRanker(data_util)
    local_ranker.rank_mention()
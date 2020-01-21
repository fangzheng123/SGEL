# encoding: utf-8


import json
import random
import data_util


class GlobalProcess(object):
    """
    process data for global entity linking
    """

    def __init__(self):
        self.data_util = data_util.DataUtil()

    def build_doc_mention(self, cut_rank_path, global_doc_mention_path, gat_candidate_num, is_train=False):
        """
        Aggregate mentions which exists in a document
        :param cut_rank_path:
        :param global_doc_mention_path:
        :param gat_candidate_num:
        :param is_train:
        :return:
        """
        doc_mention_dict = {}
        mention_candidate_obj = {}
        with open(cut_rank_path, "r", encoding="utf-8") as cut_rank_file:
            with open(global_doc_mention_path, "w", encoding="utf-8") as global_doc_file:
                for item in cut_rank_file:
                    item = item.strip()

                    mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                    mention_index = int(mention_index_str)
                    label = int(label_str)
                    fea_dict = json.loads(fea_str)
                    mention_obj = json.loads(mention_str)
                    candidate_obj = json.loads(entity_str)

                    candidate_obj["feature"] = fea_dict
                    candidate_obj["label"] = label
                    candidate_obj["mention_index"] = mention_index

                    if mention_index not in mention_candidate_obj:
                        mention_candidate_obj[mention_index] = [candidate_obj]
                    else:
                        mention_candidate_obj[mention_index].append(candidate_obj)

                    mention_file = mention_obj["mention_file"]
                    # avoid saving repeated mention
                    if mention_file not in doc_mention_dict:
                        doc_mention_dict[mention_file] = [mention_obj]
                    else:
                        mention_list = doc_mention_dict[mention_file]
                        mention_index_list = [tmp_obj["mention_index"] for tmp_obj in mention_list]
                        if mention_index not in set(mention_index_list):
                            doc_mention_dict[mention_file].append(mention_obj)

                # match candidates for each mention
                for mention_file, mention_list in doc_mention_dict.items():
                    for mention_obj in mention_list:
                        mention_index = mention_obj["mention_index"]
                        candidate_list = mention_candidate_obj[mention_index]

                        if is_train:
                            tmp_candidate_list = candidate_list[:gat_candidate_num].copy()
                            label_list = [candidate["label"] for candidate in candidate_list]
                            if 1 not in set(label_list[:gat_candidate_num]):
                                tmp_candidate_list[-1] = candidate_list[label_list.index(1)]
                            mention_obj["candidate"] = tmp_candidate_list

                        else:
                            mention_obj["candidate"] = candidate_list[:gat_candidate_num]

                    global_doc_file.write(mention_file + "\t" + json.dumps(mention_list, ensure_ascii=False) + "\n")
                    global_doc_file.flush()

    def rank_doc_mention(self, global_doc_mention_path, global_rank_path):
        """
        Ranking mentions based on the value of bert pred to the 1th xgboost position
        :param global_mention_path:
        :param global_rank_path:
        :return:
        """
        with open(global_doc_mention_path, "r", encoding="utf-8") as global_doc_mention_file:
            with open(global_rank_path, "w", encoding="utf-8") as global_rank_file:
                for item in global_doc_mention_file:
                    item = item.strip()

                    mention_file, mention_list_str = item.split("\t")
                    mention_list = json.loads(mention_list_str)

                    # mention_group_dict[mention_index] = mention_obj
                    mention_group_dict = {}
                    # mention_rank_obj[mention_index] = xgboost pred * bert pred
                    mention_rank_obj = {}
                    for mention_obj in mention_list:
                        mention_index = mention_obj["mention_index"]
                        mention_group_dict[mention_index] = mention_obj
                        candidate_list = mention_obj["candidate"]

                        first_candidate = candidate_list[0]
                        bert_prob = first_candidate["bert_prob"]
                        bert_position = max(1, first_candidate["bert_position"] + 1)

                        if len(candidate_list) > 1:
                            second_candidate = candidate_list[1]
                            xgboost_pred_sub = min(0.01, (first_candidate["xgboost_pred"] - second_candidate["xgboost_pred"]) / abs(second_candidate["xgboost_pred"]))
                            mention_rank_obj[mention_index] = xgboost_pred_sub - (bert_position/8)

                        else:
                            mention_rank_obj[mention_index] = 1

                    # rank mention
                    rank_index_list = [ele[0] for ele in sorted(mention_rank_obj.items(), key=lambda x:x[1], reverse=True)]
                    rank_mention_list = [mention_group_dict[mention_index] for mention_index in rank_index_list]

                    global_rank_file.write(mention_file + "\t" + json.dumps(rank_mention_list, ensure_ascii=False) + "\n")

                    print([mention["mention_form"] for mention in mention_list])
                    top_label_list = [mention["candidate"][0]["label"] for mention in mention_list[:2]]
                    print("top three mention acc :{0}".format(sum(top_label_list) / 2))

                    print("after ranking:")
                    print([mention["mention_form"] for mention in rank_mention_list])
                    top_label_list = [mention["candidate"][0]["label"] for mention in rank_mention_list[:2]]
                    print("top three mention acc :{0}".format(sum(top_label_list) / 2))
                    print("##################################################################################")

    def build_train_mention_graph(self, global_rank_path, global_graph_path, adjacent_num, candidate_num):
        """
        build graph for each mention (other mentions are in current mention's window)
        :param global_rank_path:
        :param global_graph_path:
        :param adjacent_num:
        :param candidate_num: the number of candidate entity for each mention
        :return:
        """
        entity_padding_error_num = 0
        mention_padding_error_num = 0
        with open(global_rank_path, "r", encoding="utf-8") as global_rank_file:
            with open(global_graph_path, "w", encoding="utf-8") as global_graph_file:
                for item in global_rank_file:
                    item = item.strip()

                    mention_file, mention_list_str = item.split("\t")

                    mention_list = json.loads(mention_list_str)

                    mention_list_len = len(mention_list)
                    seq_len = 2*adjacent_num + 1

                    for index, mention_obj in enumerate(mention_list):
                        seq_mention_list = []

                        if mention_list_len < seq_len:
                            padding_mentions = [mention_list[0] for time in range(seq_len - mention_list_len)]
                            seq_mention_list.extend(padding_mentions)
                            seq_mention_list.extend(mention_list)
                            seq_index = len(padding_mentions) + index

                        elif index-adjacent_num <= 0:
                            seq_mention_list = mention_list[0: seq_len]
                            seq_index = index

                        elif index+adjacent_num+1 > mention_list_len:
                            seq_mention_list = mention_list[mention_list_len-seq_len:mention_list_len]
                            seq_index = index - (mention_list_len-seq_len)

                        else:
                            seq_mention_list = mention_list[index-adjacent_num: index+adjacent_num+1]
                            seq_index = adjacent_num

                        # build graph for each mention（include self),
                        graph_candidate_list = []
                        graph_candidate_list.extend(mention_obj["candidate"])
                        # padding candidate entity
                        if len(mention_obj["candidate"]) < candidate_num:
                            padding_num = candidate_num-len(mention_obj["candidate"])
                            graph_candidate_list.extend([mention_obj["candidate"][0] for time in range(padding_num)])

                        correct_count = 0
                        for other_index, other_mention in enumerate(seq_mention_list):
                            if other_index != seq_index:
                                # just add correct entity
                                if correct_count < adjacent_num:
                                    # add golden target entity
                                    target_entity_list = [obj for obj in other_mention["candidate"] if obj["label"] == 1]
                                    if len(target_entity_list) > 1:
                                        target_entity_list = [target_entity_list[0]]

                                    graph_candidate_list.extend(target_entity_list)
                                    correct_count += 1
                                else:
                                    graph_candidate_list.extend(other_mention["candidate"])
                                    # padding candidate entity
                                    if len(other_mention["candidate"]) < candidate_num:
                                        padding_num = candidate_num - len(other_mention["candidate"])
                                        graph_candidate_list.extend(
                                            [other_mention["candidate"][0] for time in range(padding_num)])

                        global_graph_file.write(json.dumps(graph_candidate_list, ensure_ascii=False) + "\n")
                        global_graph_file.flush()

                        if len(seq_mention_list) % seq_len != 0:
                            mention_padding_error_num += 1

                        if len(graph_candidate_list) != candidate_num * (adjacent_num + 1) + adjacent_num:
                            entity_padding_error_num += 1

                print("mention padding error num: {0}".format(mention_padding_error_num))
                print("entity padding error num: {0}".format(entity_padding_error_num))

    def build_test_mention_graph(self, global_rank_path, global_graph_path, adjacent_num, select_mention_num, candidate_num):
        """
        build graph for each mention (other mentions are in current mention's window and ranked mentions)
        :param global_rank_path:
        :param global_graph_path:
        :param adjacent_num: the number of mentions selected from ranked list
        :param select_mention_num: the number of mentions selected from ranked list
        :param candidate_num:
        :return:
        """
        entity_padding_error_num = 0
        mention_padding_error_num = 0
        with open(global_rank_path, "r", encoding="utf-8") as global_rank_file:
            with open(global_graph_path, "w", encoding="utf-8") as global_graph_file:
                for item in global_rank_file:
                    item = item.strip()

                    mention_file, mention_list_str = item.split("\t")

                    mention_list = json.loads(mention_list_str)
                    all_mention_list_len = len(mention_list)

                    select_mention_list = mention_list[:select_mention_num]
                    select_entity_list = [ele["candidate"][0] for ele in select_mention_list]

                    if all_mention_list_len > select_mention_num:
                        disam_mention_list = mention_list[select_mention_num:]
                        other_mention_len = len(disam_mention_list)

                        for index, mention_obj in enumerate(disam_mention_list):
                            seq_mention_list = []

                            if other_mention_len < adjacent_num + 1:
                                padding_mentions = [disam_mention_list[0] for time in range(adjacent_num+1 - other_mention_len)]
                                seq_mention_list.extend(padding_mentions)
                                seq_mention_list.extend(disam_mention_list)
                                seq_index = len(padding_mentions) + index

                            else:
                                if index - adjacent_num <= 0:
                                    seq_mention_list = disam_mention_list[0: adjacent_num+1]
                                    seq_index = index

                                else:
                                    seq_mention_list = mention_list[index - adjacent_num: index + 1]
                                    seq_index = adjacent_num

                            # build graph for each mention（graph include self),
                            graph_candidate_list = []
                            graph_candidate_list.extend(mention_obj["candidate"])
                            # padding candidate entity
                            if len(mention_obj["candidate"]) < candidate_num:
                                padding_num = candidate_num - len(mention_obj["candidate"])
                                graph_candidate_list.extend([mention_obj["candidate"][0] for time in range(padding_num)])

                            # add selected entity
                            graph_candidate_list.extend(select_entity_list)

                            for other_index, other_mention in enumerate(seq_mention_list):
                                if other_index != seq_index:
                                    graph_candidate_list.extend(other_mention["candidate"])
                                    # padding candidate entity
                                    if len(other_mention["candidate"]) < candidate_num:
                                        padding_num = candidate_num - len(other_mention["candidate"])
                                        graph_candidate_list.extend(
                                            [other_mention["candidate"][0] for time in range(padding_num)])

                            global_graph_file.write(json.dumps(graph_candidate_list, ensure_ascii=False) + "\n")
                            global_graph_file.flush()

                            if len(seq_mention_list) % (adjacent_num + 1) != 0:
                                mention_padding_error_num += 1

                            if len(graph_candidate_list) != candidate_num * (adjacent_num + 1) + select_mention_num:
                                entity_padding_error_num += 1

                print("mention padding error num: {0}".format(mention_padding_error_num))
                print("entity padding error num: {0}".format(entity_padding_error_num))


    def control_train(self):
        # for train
        data_name = "aida_train"
        bert_rank_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/bert/" + data_name + "_cut_rank_bert_vec"
        global_doc_mention_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/gat/" + data_name + "_global_doc_mention"
        global_rank_mention_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/gat/" + data_name + "_global_rank_mention"
        global_graph_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/gat/" + data_name + "_global_graph"

        # print(data_name)
        # self.data_util.cal_candidate_recall(bert_rank_path, 5)
        # self.build_doc_mention(bert_rank_path, global_doc_mention_path, 5, is_train=True)
        # self.rank_doc_mention(global_doc_mention_path, global_rank_mention_path)
        # self.build_train_mention_graph(global_rank_mention_path, global_graph_path, 2, 5)

    def control_test(self):
        # for test
        data_name = "kore50"
        bert_rank_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/bert/" + data_name + "_cut_rank_bert_vec"
        global_doc_mention_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/gat/" + data_name + "_global_doc_mention"
        global_rank_mention_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/gat/" + data_name + "_global_rank_mention"
        global_graph_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/gat/" + data_name + "_global_graph"

        print(data_name)
        self.data_util.cal_candidate_recall(bert_rank_path, 5)
        self.build_doc_mention(bert_rank_path, global_doc_mention_path, 5)
        self.rank_doc_mention(global_doc_mention_path, global_rank_mention_path)
        self.build_test_mention_graph(global_rank_mention_path, global_graph_path, 2, 2, 5)


if __name__ == "__main__":
    global_process = GlobalProcess()

    global_process.control_test()




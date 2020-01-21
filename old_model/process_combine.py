# encoding: utf-8


import data_util
import config_util
import json

class CombineProcess(object):
    """
    process data for final combine old_model
    """

    def __init__(self):
        self.data_util = data_util.DataUtil()


    def combine_bert_gat_vec(self, global_rank_path, global_graph_vec_path, combine_vec_path):
        """
        combine bert and gat vector
        :param global_rank_path:
        :param global_graph_vec_path:
        :param combine_vec_path:
        :return:
        """
        mention_candidate_dict = {}
        with open(global_rank_path, "r", encoding="utf-8") as global_rank_file:
            for item in global_rank_file:
                item = item.strip()
                mention_file, mention_list_str = item.split("\t")
                mention_list = json.loads(mention_list_str)

                for mention_obj in mention_list:
                    mention_candidate_dict[mention_obj["group_id"]] = mention_obj

        vec_candidate_dict = {}
        with open(global_graph_vec_path, "r", encoding="utf-8") as global_graph_vec_file:
            for item in global_graph_vec_file:
                item = item.strip()
                candidate_obj = json.loads(item)
                group_id = candidate_obj["group_id"]

                if group_id not in vec_candidate_dict:
                    vec_candidate_dict[group_id] = [candidate_obj]
                else:
                    vec_candidate_dict[group_id].append(candidate_obj)

        with open(combine_vec_path, "w", encoding="utf-8") as combine_vec_file:
            for group_id, gat_candidate_list in vec_candidate_dict.items():
                mention_obj = mention_candidate_dict[group_id]

                mention_obj["candidate"] = gat_candidate_list

                if len(gat_candidate_list) != config_util.global_candidate_num:
                    print("not padding")

                combine_vec_file.write(json.dumps(mention_obj) + "\n")

    def control_vec(self):
        data_name = "kore50"

        print(data_name)
        global_rank_mention_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/gat/" + data_name + "_global_rank_mention"
        global_graph_vec_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/gat/" + data_name + "_global_graph_vec"
        combine_vec_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/combine/" + data_name + "_combine_vec"

        self.combine_bert_gat_vec(global_rank_mention_path, global_graph_vec_path, combine_vec_path)

if __name__ == "__main__":
    combine_process = CombineProcess()
    combine_process.control_vec()
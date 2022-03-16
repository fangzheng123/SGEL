# encoding: utf-8

import json
import data_util

class ProcessBert(object):

    def __init__(self):
        self.data_util = data_util.DataUtil()

    def fine_tuning_bert_data(self, source_path, bert_path):
        """
        build data for fine-tuning bert (train, validate and test)
        :param source_path:
        :param bert_path:
        :return:
        """
        with open(source_path, "r", encoding="utf-8") as source_file:
            with open(bert_path, "w", encoding="utf-8") as bert_file:
                candidate_item_list = []
                fine_tuing_group_dict = {}

                for item in source_file:
                    item = item.strip()
                    group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                    group_id = int(group_str)
                    label = int(label_str)
                    mention_obj = json.loads(mention_str)
                    entity_obj = json.loads(entity_str)

                    if "source_name" in entity_obj:
                        entity_obj["name"] = entity_obj["source_name"]

                    mention_context, entity_des = self.data_util.get_context_desc(mention_obj, entity_obj)

                    ########
                    fea_obj = json.loads(fea_str)
                    fea_obj["context_summary_word_cos"] = 0
                    fea_obj["context_category_word_cos"] = 0
                    fea_obj["same_candidate_word_num"] = 0
                    fea_obj["same_mention_word_num"] = 0
                    fea_obj["has_frequency_word"] = 0
                    fea_obj["same_word_id_top"] = 0
                    fea_obj["second_rank_id"] = 0
                    fea_obj["has_keyword"] = 0
                    ########

                    if mention_context.strip() != "" and entity_des.strip() != "":
                        candidate_item_list.append([group_str, label_str, json.dumps(fea_obj), mention_context.replace("\n", ""), entity_des.replace("\n", "")])

                        if group_id not in fine_tuing_group_dict:
                            fine_tuing_group_dict[group_id] = [label]
                        else:
                            fine_tuing_group_dict[group_id].append(label)

                    # if label == 1 and entity_des.strip() == "":
                    #     print(group_id, label, mention_obj["mention_form"], entity_obj["name"])

                filter_group_set = set([ele_id for ele_id in fine_tuing_group_dict
                                               if len(fine_tuing_group_dict[ele_id]) == sum(fine_tuing_group_dict[ele_id])
                                               or sum(fine_tuing_group_dict[ele_id]) == 0])

                print("filter mention num: {0}".format(len(filter_group_set)))

                # for id in filter_group_set:
                #     if len(fine_tuing_group_dict[id]) == sum(fine_tuing_group_dict[id]):
                #         print(id, fine_tuing_group_dict[id], len(fine_tuing_group_dict[id]))

                for item_list in candidate_item_list:
                    group_id = int(item_list[0])
                    # filter group which not contain golden entity or only contain golden entity
                    if group_id not in filter_group_set:
                        bert_file.write("\t".join(item_list) + "\n")

    def map_fine_tuning_vec2data(self, source_path, bert_vec_path, bert_rank_vec_path):
        """
        add fine tuning vec and prob to source data
        :param source_path:
        :param bert_vec_path:
        :param bert_rank_vec_path:
        :return:
        """
        vec_count = 0
        group_vec_dict = {}
        vec_list = []
        with open(bert_vec_path, "r", encoding="utf-8") as bert_vec_file:
            for index, item in enumerate(bert_vec_file):
                item = item.strip()

                if len(item.split("\t")) == 1:
                    continue

                vec_count += 1

                group_str, bert_score_str, rank_position, mention_vec_str, entity_vec_str = item.split("\t")
                group_id = int(group_str)

                bert_score = float(bert_score_str)
                rank_position = int(rank_position)
                mention_vec = json.loads(mention_vec_str)
                entity_vec = json.loads(entity_vec_str)

                bert_vec_list = [bert_score, mention_vec, entity_vec, rank_position-1]
                vec_list.append(bert_vec_list)

                if group_id not in group_vec_dict:
                    group_vec_dict[group_id] = [bert_vec_list]
                else:
                    group_vec_dict[group_id].append(bert_vec_list)

        print("vec_count: {0}".format(vec_count))

        vec_index = 0
        new_item_list = []
        vec_group_set = set(group_vec_dict.keys())

        with open(source_path, "r", encoding="utf-8") as source_file:
            for item in source_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_str)
                mention_obj = json.loads(mention_str)
                entity_obj = json.loads(entity_str)

                if group_id not in vec_group_set:
                    continue

                mention_context, entity_des = self.data_util.get_context_desc(mention_obj, entity_obj)

                if mention_context.strip() != "" and entity_des.strip() != "":
                    vec_item = vec_list[vec_index]
                    bert_score, mention_vec, entity_vec, bert_rank_position = vec_item
                    mention_obj["vec"] = mention_vec
                    entity_obj["vec"] = entity_vec
                    entity_obj["bert_prob"] = bert_score
                    entity_obj["bert_position"] = bert_rank_position

                    new_item_list.append([group_str, label_str, fea_str, json.dumps(mention_obj), json.dumps(entity_obj)])
                    vec_index += 1

        with open(bert_rank_vec_path, "w", encoding="utf-8") as bert_rank_vec_file:
            for item in new_item_list:
                bert_rank_vec_file.write("\t".join(item) + "\n")

    def recall_analyse(self, rank_bert_path, top_num):
        """

        :param bert_path:
        :param top_num:
        :return:
        """
        all_group_set = set()
        group_dict = {}
        with open(rank_bert_path, "r", encoding="utf-8") as bert_file:
            for item in bert_file:
                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_str)
                label = int(label_str)
                entity_obj = json.loads(entity_str)

                if "bert_position" in entity_obj:
                    all_group_set.add(group_id)

                    bert_position = entity_obj["bert_position"]
                    if label == 1 and bert_position < top_num:
                        group_dict[group_id] = 1

        recall_count = len(group_dict)
        all_count = len(all_group_set)

        print(rank_bert_path.split("/")[-1], top_num, recall_count, all_count, recall_count/all_count)

    def combine_xgboost_bert(self, cut_candidate_path, bert_vec_path):
        """
        combine xgboost score and bert score, and add bert's rank postion to entity object
        :param cut_candidate_path:
        :param bert_vec_path:
        :return:
        """

        group_score_dict = {}
        group_id_list = []
        with open(bert_vec_path, "r", encoding="utf-8") as bert_score_file:
            for item in bert_score_file:
                item = item.strip()

                group_id = int(item.split("\t")[0])
                score = float(item.split("\t")[-1])

                if group_id not in set(group_id_list):
                    group_id_list.append(group_id)

                if group_id not in group_score_dict:
                    group_score_dict[group_id] = [score]
                else:
                    group_score_dict[group_id].append(score)

        # rank bert score
        all_bert_list = []
        for group_id in group_id_list:
            score_list = group_id_list[group_id]

            score_dict = {}
            for index, score in enumerate(score_list):
                score_dict[index] = score

            position = 1
            for index, val in sorted(score_dict.items(), key=lambda x: x[1], reverse=True):
                tmp_dict = {}
                tmp_dict["bert_rank_positon"] = position
                tmp_dict["bert_score"] = val
                score_list[index] = tmp_dict
                position += 1

            # score_list = [dict, dict, ...]
            all_bert_list.extend(score_list)

        # combine xgboost and bert score
        cut_candidate_list = []
        with open(cut_candidate_path, "r", encoding="utf-8") as cut_candidate_file:
            for index, item in enumerate(cut_candidate_file):
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                fea_dict = json.loads(fea_str)

                bert_dict = all_bert_list[index]

                fea_dict["bert_rank_position"] = bert_dict["bert_rank_position"]
                fea_dict["bert_score"] = bert_dict["bert_score"]

                fea_str = json.dumps(fea_dict)

                cut_candidate_list.append("\t".join([group_str, label_str, fea_str, mention_str, entity_str]))

        with open(cut_candidate_path, "w", encoding="utf-8") as cut_candidate_file:
            for item in cut_candidate_list:
                cut_candidate_file.write(item + "\n")

    def control_source_bert(self):
        data_name = "ace2004"
        cut_candidate_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/generate/" + data_name + "_cut_rank_format"

        cut_candidate_path = "/home1/fangzheng/data/bert_el_data/aida_train/generate/aida_train_cut_rank_format_small"
        bert_path = "/home1/fangzheng/data/bert_el_data/aida_train/bert/aida_train_cut_rank_format_small"
        mention_vec_path = "/home1/fangzheng/data/bert_el_data/aida_train/bert/aida_train_small_mention_sent_features"
        entity_vec_path = "/home1/fangzheng/data/bert_el_data/aida_train/bert/aida_train_small_entity_sent_features"
        bert_process.build_sent(cut_candidate_path, bert_path)

        mention_vec_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/bert/" + "mention_sent_features"
        entity_vec_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/bert/" + "entity_sent_features"
        bert_process.map_sent_vector(cut_candidate_path, mention_vec_path, entity_vec_path)

    def control_fine_tuning(self):
        data_name = "kore50"
        cut_rank_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_cut_rank_format"
        print(data_name)

        bert_fine_tuning_data_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/bert/" + data_name + "_bert_data"
        bert_process.fine_tuning_bert_data(cut_rank_path, bert_fine_tuning_data_path)

        bert_fine_tuning_vec_path = "/home1/fangzheng/project/bert/predict_bs2/" + data_name + "_merged"
        cut_rank_bert_vec_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/bert/" + data_name + "_cut_rank_bert_vec"
        bert_process.map_fine_tuning_vec2data(cut_rank_path, bert_fine_tuning_vec_path, cut_rank_bert_vec_path)

if __name__ == "__main__":
    bert_process = ProcessBert()

    source_dir = "/data/fangzheng/bert_el/"
    bert_source_dir = "/data/fangzheng/bert/"
    name_list = ["aida_testB"]
    # name_list = ["msnbc", "ace2004", "aquaint", "rss500", "reuters128", "kore50", "aida_testA", "aida_testB"]
    for data_name in name_list:
        print(data_name)

        cut_data_path = source_dir + data_name + "/other_candidate/" + data_name + "_cut_rank_format"
        bert_data_path = bert_source_dir + data_name + "_other/" + data_name + "_cut_sent_format"
        bert_result_path = bert_source_dir + "sent_vecs/" + data_name + "_cut_sent_format.merged"
        bert_rank_path = source_dir + data_name + "/bert/" + data_name + "_cut_rank_format_bert"

        bert_process.fine_tuning_bert_data(cut_data_path, bert_data_path)
        bert_process.map_fine_tuning_vec2data(cut_data_path, bert_result_path, bert_rank_path)
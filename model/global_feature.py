# encoding: utf-8

import json
from collections import Counter
from data_util import DataUtil

class GlobalFeature(object):

    def __init__(self, data_util):
        self.data_util = data_util

    def set_data_name(self, data_name):
        """

        :param data_name:
        :return:
        """
        self.data_name = data_name

    def process_candidate_name(self, name):
        """

        :param name:
        :return:
        """
        name = name.lower().replace("(", "").replace(")", "").replace(".", "").replace(",", "")
        return name

    def get_mention_candidate(self, all_rank_format_path):
        """

        :param all_rank_format_path:
        :return:
        """
        doc_mention_dict = {}
        mention_detail_dict = {}
        mention_candidate_dict = {}
        with open(all_rank_format_path, "r", encoding="utf-8") as all_rank_format_file:
            for item in all_rank_format_file:
                item = item.strip()

                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                entity = json.loads(entity_str)
                mention_obj = json.loads(mention_str)
                mention_file = mention_obj["mention_file"]

                mention_index = int(mention_index_str)

                mention_detail_dict[mention_index] = mention_obj

                if mention_index not in mention_candidate_dict:
                    mention_candidate_dict[mention_index] = [entity]
                else:
                    mention_candidate_dict[mention_index].append(entity)

                if mention_file not in doc_mention_dict:
                    doc_mention_dict[mention_file] = [mention_index]
                else:
                    doc_mention_dict[mention_file].append(mention_index)

        return doc_mention_dict, mention_detail_dict, mention_candidate_dict

    def cal_global_fea(self, mention_obj, entity, doc_mention_dict, mention_detail_dict, mention_candidate_dict):
        """
        calculate global feature between candidates
        :param mention_obj:
        :param entity:
        :param doc_mention_dict:
        :param mention_detail_dict:
        :param mention_candidate_dict:
        :return:
        """
        same_candidate_word_num = 0
        same_mention_word_num = 0
        has_frequency_word = 0
        same_word_id_top = 0
        second_rank_id = 0
        has_keyword = 0

        entity_name = entity["name"]
        current_mention_index = int(mention_obj["mention_index"])
        mention_file = mention_obj["mention_file"]
        current_mention_form = mention_obj["mention_form"]
        current_candidate_index = 0
        for index, ele in enumerate(mention_candidate_dict[current_mention_index]):
            if entity_name == ele["name"]:
                current_candidate_index = index

        doc_mention_list = doc_mention_dict[mention_file]
        adj_candidate_list = []
        adj_mention_list = []
        for other_mention_index in doc_mention_list:
            other_mention = mention_detail_dict[other_mention_index]

            if other_mention_index != current_mention_index \
                    and other_mention["mention_form"].lower() != current_mention_form.lower():

                if abs(other_mention_index - current_mention_index) < 5:
                    other_candidate_list = [self.process_candidate_name(ele["name"].replace(",", "").lower())
                                            for ele in mention_candidate_dict[other_mention_index]]

                    if len(mention_candidate_dict[other_mention_index]) == 1:
                        for i in range(3):
                            adj_candidate_list.extend(other_candidate_list)

                    else:
                        adj_candidate_list.extend(other_candidate_list)

                adj_mention_list.append(other_mention["mention_form"].lower())

        other_candidate_word_list = []
        other_candidate2_word_list = []
        other_candidate3_word_list = []
        for name in adj_candidate_list:
            other_candidate_word_list.extend(name.split("_"))
            other_candidate2_word_list.extend(name.split("_"))
            other_candidate3_word_list.extend(name.split("_"))

        other_mention_word_list = []
        for name in adj_mention_list:
            other_mention_word_list.extend(name.split(" "))

        other_candidate_word_counter = Counter(other_candidate_word_list)
        other_mention_word_counter = Counter(other_mention_word_list)

        sort_candidate_word_list = [pair[0] for pair in sorted(other_candidate_word_counter.items(), key=lambda x: x[1], reverse=True)]

        # print("#####".join(sort_candidate_word_list[:3]))

        if ("mention_context" in mention_obj and mention_obj["mention_context"] != "") \
                or ("mention_context" not in mention_obj):
            clean_entity_name = self.data_util.remove_stop_word(self.process_candidate_name(entity_name.lower().replace("_", " ")))
            for word in clean_entity_name.split(" "):
                # the same word num between current candidate and other mention's candidate
                if word in other_candidate_word_counter:
                    same_candidate_word_num += other_candidate_word_counter[word]

                # the same word num between current candidate and other mention's surface form
                if word in other_mention_word_counter:
                    same_mention_word_num += other_mention_word_counter[word]

                # has high frequency candidate word
                if word in sort_candidate_word_list[:3]:
                    has_frequency_word += 1

        # page id feature
        page_id_list = [ele["page_id"] for ele in mention_candidate_dict[current_mention_index]]
        if (entity["page_id"] == max(page_id_list) or entity["page_id"] == max(page_id_list[:2])) \
                and same_candidate_word_num > 8:
            same_word_id_top = 1
        if entity["page_id"] == max(page_id_list[:2]) and entity["page_id"] > 10 * page_id_list[0]:
            second_rank_id = 1

        if self.data_name.__contains__("aida"):
            if mention_obj["mention_context"].strip() != "":
                keyword = mention_obj["mention_context"].strip().split(" ")[-1].lower().replace("soccer", "football")
                if current_candidate_index < 2 and entity_name.lower().__contains__(keyword):
                    has_keyword = 1

        return same_candidate_word_num, same_mention_word_num, has_frequency_word, same_word_id_top, second_rank_id, has_keyword

    def add_global_fea(self, cut_rank_format_path):
        """
        add global feature
        :param cut_rank_format_path:
        :return:
        """
        doc_mention_dict, mention_detail_dict, mention_candidate_dict = self.get_mention_candidate(cut_rank_format_path)

        global_item_list = []
        with open(cut_rank_format_path, "r", encoding="utf-8") as cut_rank_format_file:
            for item in cut_rank_format_file:
                item = item.strip()

                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                mention_obj = json.loads(mention_str)
                entity = json.loads(entity_str)
                mention_index = int(mention_index_str)

                same_candidate_word_num, same_mention_word_num, has_frequency_word, same_word_id_top, \
                second_rank_id, has_keyword = self.cal_global_fea(mention_obj, entity, doc_mention_dict,
                                                                  mention_detail_dict, mention_candidate_dict)

                fea_obj = json.loads(fea_str)

                fea_obj["same_candidate_word_num"] = same_candidate_word_num
                fea_obj["same_mention_word_num"] = same_mention_word_num
                fea_obj["has_frequency_word"] = has_frequency_word
                fea_obj["same_word_id_top"] = same_word_id_top
                fea_obj["second_rank_id"] = second_rank_id
                fea_obj["has_keyword"] = has_keyword

                global_item_list.append("\t".join([mention_index_str, label_str, json.dumps(fea_obj), mention_str, entity_str]))

        with open(cut_rank_format_path, "w", encoding="utf-8") as all_rank_format_file:
            for item in global_item_list:
                all_rank_format_file.write(item + "\n")
                all_rank_format_file.flush()

if __name__ == "__main__":
    data_util = DataUtil()
    global_fea = GlobalFeature(data_util)

    data_list = ["aida_testB"]
    source_dir = "/data/fangzheng/bert_el/"
    # data_list = ["ace2004", "msnbc", "aquaint", "clueweb", "wiki", "aida_train", "aida_testA", "aida_testB"]
    for data_name in data_list:
        print(data_name)

        global_fea.set_data_name(data_name)
        cut_rank_path = source_dir + data_name + "/candidate/" + data_name + "_cut_rank_format"
        global_fea.add_global_fea(cut_rank_path)
# encoding: utf-8


import json
import re
import os
from urllib import parse
import Levenshtein
import collections
import data_util

class BaseProcess(object):

    def __init__(self):
        self.data_util = data_util.DataUtil()

    def filter_not_inkb_data(self, pre_path, in_wiki_path):
        """
        filter entity that not in kb
        :param pre_path:
        :param in_wiki_path:
        :return:
        """
        in_wiki_data_list = []
        with open(pre_path, "r", encoding="utf-8") as target_file:
            for item in target_file:
                item = item.strip()

                mention_obj = json.loads(item)

                if not mention_obj["target_url"].__contains__("notInWiki"):
                    in_wiki_data_list.append(item)

        with open(in_wiki_path, "w", encoding="utf-8") as in_wiki_target_file:
            for index, item in enumerate(in_wiki_data_list):
                item = item.strip()
                mention_obj = json.loads(item)
                mention_obj["mention_index"] = index

                in_wiki_target_file.write(json.dumps(mention_obj) + "\n")

        print("in kb mention num:{0}".format(len(in_wiki_data_list)))

    def get_father_mention(self, mention_data_path, person_path, is_need_person=True):
        """
        get father mention in the same document
        :param mention_data_path:
        :param person_path:
        :return:
        """
        # all person name
        person_mention_set = self.data_util.read_person_name(person_path)

        # dict[file_name] = mention_list
        doc_mention_dict = self.data_util.read_doc_mention(mention_data_path)

        mention_list = []
        with open(mention_data_path, "r", encoding="utf-8") as pre_file:
            for item in pre_file:
                item = item.strip()

                mention_obj = json.loads(item)

                mention_file = mention_obj["mention_file"]
                mention_form = mention_obj["mention_form"]

                # search mention in doc which contains current mention
                # filter mention whose length less than 2
                person_father_list = []
                if len(mention_form) > 2 and mention_file in doc_mention_dict:
                    doc_mention_list = doc_mention_dict[mention_file]
                    father_mention_list = self.data_util.search_father_mention(mention_form, doc_mention_list)
                    for father_item in set(father_mention_list):
                        if is_need_person:
                            if father_item.lower() in person_mention_set:
                                person_father_list.append(father_item)
                                print(mention_form, father_item)
                        else:
                            person_father_list.append(father_item)
                            print(mention_form, father_item)

                mention_obj["father_mention"] = person_father_list

                mention_list.append(mention_obj)

        with open(mention_data_path, "w", encoding="utf-8") as mention_file:
            for mention_obj in mention_list:
                mention_file.write(json.dumps(mention_obj) + "\n")

    def get_adjacent_mention(self, mention_data_path):
        """
        get adjacent mentions
        :param mention_data_path:
        :return:
        """
        # dict[file_name] = mention_index_list
        doc_mention_dict = self.data_util.get_doc_mention_index(mention_data_path)
        all_mention_dict = self.data_util.get_all_mention(mention_data_path)

        new_data_list = []
        with open(mention_data_path, "r", encoding="utf-8") as mention_data_file:
            for item in mention_data_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]
                mention_index = mention_obj["mention_index"]

                mention_list = doc_mention_dict[mention_file]
                current_index = mention_list.index(mention_index)

                pre_mention_list = mention_list[max(current_index-2, 0): current_index]
                next_mention_list = mention_list[current_index+1:current_index+3]

                mention_obj["pre_mention"] = [all_mention_dict[index]["mention_form"] for index in pre_mention_list]
                mention_obj["next_mention"] = [all_mention_dict[index]["mention_form"] for index in next_mention_list]

                new_data_list.append(mention_obj)

        with open(mention_data_path, "w", encoding="utf-8") as mention_file:
            for mention_obj in new_data_list:
                mention_file.write(json.dumps(mention_obj) + "\n")

    def get_mention_synonym(self, mention_data_path):
        """
        get mention synonym by WordNet
        :param mention_data_path:
        :return:
        """
        mention_list = []
        with open(mention_data_path, "r", encoding="utf-8") as mention_file:
            for item in mention_file:
                item = item.strip()

                mention_obj = json.loads(item)

                mention_form = mention_obj["mention_form"]
                capitalize_mention_form = self.data_util.capitalize_mention(mention_form)
                synonym_list = self.data_util.get_synonym(capitalize_mention_form)
                mention_obj["synonym_mention"] = synonym_list

                mention_list.append(mention_obj)

        with open(mention_data_path, "w", encoding="utf-8") as mention_file:
            for mention_obj in mention_list:
                mention_file.write(json.dumps(mention_obj) + "\n")

    def get_context_keyword(self, mention_data_path):
        """
        get mention context keyword
        :param mention_data_path:
        :return:
        """
        mention_list = []
        with open(mention_data_path, "r", encoding="utf-8") as mention_file:
            for item in mention_file:
                item = item.strip()

                mention_obj = json.loads(item)

                mention_form = mention_obj["mention_form"]
                pre_mention_list = mention_obj["pre_mention"]
                next_mention_list = mention_obj["next_mention"]
                mention_context = mention_obj["mention_context"].lower()
                mention_context = self.data_util.remove_stop_word(mention_context)

                context_keyword_list = []

                # extract keywords
                if mention_context != "":
                    mention_context_lower = mention_context.replace(mention_form.lower(), "")
                    mention_context_lower = self.data_util.replace_keyword(mention_context_lower)
                    context_keyword_tuple = collections.Counter(mention_context_lower.split(" ")[-20:]).most_common(1)[0]

                    if context_keyword_tuple[1] < 3:
                        # use the adjacent mention
                        if len(pre_mention_list) > 0:
                            context_keyword_list.append(pre_mention_list[-1])
                        if len(next_mention_list) > 0:
                            context_keyword_list.append(next_mention_list[0])
                    else:
                        # use the most frequent words
                        context_keyword_list.append(context_keyword_tuple[0])

                mention_obj["context_keyword"] = context_keyword_list

                mention_list.append(mention_obj)

        with open(mention_data_path, "w", encoding="utf-8") as mention_file:
            for mention_obj in mention_list:
                mention_file.write(json.dumps(mention_obj) + "\n")

    def get_hand_result(self, hand_label_path):
        """
        add the result of manual annotations
        :param format_path:
        :param hand_label_path:
        :return:
        """
        hand_target_dict = {}

        if hand_label_path != "":
            with open(hand_label_path, "r", encoding="utf-8") as hand_label_file:
                for item in hand_label_file:
                    item = item.strip()

                    source_target = item.split(" ")[0]
                    hand_target = item.split(" ")[1]

                    hand_target_dict[source_target.lower()] = hand_target

        return hand_target_dict

    def add_manual_target(self, format_data_path, hand_label_path, own_label_format):
        """
        add our own manual target url
        :param format_data_path:
        :return:
        """
        hand_label_dict = self.get_hand_result(hand_label_path)

        count = 0
        recall_count = 0
        with open(format_data_path, "r", encoding="utf-8") as format_data_file:
            with open(own_label_format, "w", encoding="utf-8") as own_format_data_file:

                for item in format_data_file:
                    item = item.strip()
                    mention_obj = json.loads(item)

                    count += 1

                    mention = mention_obj["mention_form"]
                    target_url = mention_obj["own_target_url"]
                    target_url_lower = target_url.split("/")[-1].lower()

                    all_url = []
                    candidate_dict = mention_obj["candidate"]
                    if "doc_candidate" in mention_obj["candidate"]:
                        candidate_dict = mention_obj["candidate"]["doc_candidate"]

                    for key, url_list in candidate_dict.items():
                        all_url.extend(url_list)

                    all_url_lower = [url.lower() for url in all_url]

                    # 1.name comparison
                    if target_url_lower in set(all_url_lower):
                        recall_count += 1

                    # 2. manual labeling
                    elif target_url_lower in hand_label_dict and hand_label_dict[target_url_lower].lower() in set(all_url_lower):
                        mention_obj["own_target_url"] = hand_label_dict[target_url_lower].lower()
                        recall_count += 1

                    else:
                        print(mention, target_url)
                        print(all_url)

                    own_format_data_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")

        print(count, recall_count, count-recall_count, recall_count*1.0/count)

    def filter_disam_candidate(self, disam_name_list, preserve_num, all_entity_dict):
        """
        filter candidate in disambiguate page, preserve top N(pageview) candidate
        :param disam_name_list:
        :param preserve_num:
        :param all_entity_dict:
        :return:
        """
        disam_candidate_list = disam_name_list

        pv_disam_dict = {}
        for name in set(disam_name_list):
            entity_json = {}
            if name in all_entity_dict:
                entity_json = all_entity_dict[name]

            if "popularity" in entity_json and "views_sum" in entity_json["popularity"]:
                entity_pv = entity_json["popularity"]["views_sum"]
                pv_disam_dict[name] = int(entity_pv)

        # rank disambiguate url list, preserve topK
        if len(pv_disam_dict) > preserve_num:
            pv_list = sorted(pv_disam_dict.items(), key=lambda x: x[1], reverse=True)
            disam_candidate_list = [ele[0] for ele in pv_list[:preserve_num]]

        return disam_candidate_list

    # def filter_candidate_by_pv(self, candidate_format_path):
    #     """
    #     filter candidate which has low pv
    #     :param candidate_format_path:
    #     :return:
    #     """
    #     save_list = []
    #     with open(candidate_format_path, "r", encoding="utf-8") as candidate_format_file:
    #         for item in candidate_format_file:
    #             item = item.strip()
    #
    #             group_str, label_str, mention_str, entity_str = item.split("\t")
    #
    #             mention_obj = json.loads(mention_str)
    #             candidate_entity = json.loads(entity_str)
    #
    #             if "popularity" in candidate_entity and "views_sum" in candidate_entity["popularity"]:
    #                 candidate_pv = candidate_entity["popularity"]["views_sum"]
    #
    #                 if candidate_pv > 10000 or candidate_entity["name"] == mention_obj["own_target_url"].split("/")[-1]:
    #                     save_list.append(item)
    #
    #                 else:
    #                     if int(label_str) == 1:
    #                         print("filter target entity: {0}, {1}, {2}".format(mention_obj["own_target_url"],
    #                                                                            candidate_entity["url"], candidate_entity["name"]))
    #                     pass
    #             else:
    #                 if candidate_entity["name"] == mention_obj["target_redirect_url"].split("/")[-1]:
    #                     save_list.append(item)
    #
    #     with open(candidate_format_path, "w", encoding="utf-8") as candidate_format_file:
    #         for item in save_list:
    #             candidate_format_file.write(item + "\n")

    # def relabel_candidate(self, candidate_format_path):
    #     """
    #     relabel candidate, part candidates which have special characters are also golden entities
    #     :param candidate_format_path:
    #     :return:
    #     """
    #     item_list = []
    #     relabel_count = 0
    #     with open(candidate_format_path, "r", encoding="utf-8") as candidate_format_file:
    #         group_candidate_dict = {}
    #         group_mention_dict = {}
    #         for item in candidate_format_file:
    #             item = item.strip()
    #
    #             group_str, label_str, mention_str, entity_str = item.split("\t")
    #
    #             group_id = int(group_str)
    #             label = int(label_str)
    #             mention_obj = json.loads(mention_str)
    #             candidate_entity = json.loads(entity_str)
    #
    #             candidate_entity["label"] = label
    #             candidate_entity["group_id"] = group_id
    #
    #             if group_id not in group_candidate_dict:
    #                 group_candidate_dict[group_id] = [candidate_entity]
    #                 group_mention_dict[group_id] = mention_obj
    #             else:
    #                 group_candidate_dict[group_id].append(candidate_entity)
    #
    #         for group_id, candidate_list in group_candidate_dict.items():
    #             target_name_list = [candidate["name"] for candidate in candidate_list if candidate["label"] == 1]
    #
    #             for candidate in candidate_list:
    #                 name = candidate["name"]
    #                 if candidate["label"] != 1 and (True in [parse.unquote(target_name) == parse.unquote(name) for target_name in target_name_list]):
    #                     candidate["label"] = 1
    #                     relabel_count += 1
    #
    #                 item_list.append([str(group_id), str(candidate["label"]),
    #                                   json.dumps(group_mention_dict[group_id]), json.dumps(candidate)])
    #
    #     with open(candidate_format_path, "w", encoding="utf-8") as candidate_format_file:
    #         for item in item_list:
    #             candidate_format_file.write("\t".join(item) + "\n")
    #
    #     print("relabel candidate num: {0}".format(relabel_count))

if __name__ == "__main__":
    pass
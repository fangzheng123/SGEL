# encoding: utf-8

import json
from urllib import parse
from data_util import DataUtil
import numpy as np
from collections import Counter

class CandidateControl(object):

    def __init__(self, data_util):
        self.data_util = data_util

    def combine_keyword_candidate(self, candidate_path):
        """

        :param candidate_path:
        :return:
        """
        mention_list = []
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                candidate = mention_obj["candidate"]

                if "mention_keyword_search" in mention_obj:
                    candidate["mention_keyword_search"] = mention_obj["mention_keyword_search"]

                mention_list.append(mention_obj)

        with open(candidate_path, "w", encoding="utf-8") as candidate_file:
            for mention_obj in mention_list:
                candidate_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")

    def get_father_candidate(self, candidate_path, is_cover=True):
        """
        get father mention's candidate in same doc
        :param candidate_path:
        :return:
        """
        mention_obj_list = []
        doc_mention_candidate_dict = {}
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_form = mention_obj["mention_form"].strip()
                mention_file = mention_obj["mention_file"]
                candidate = mention_obj["candidate"]

                if mention_file not in doc_mention_candidate_dict:
                    mention_candidate_dict = {}
                    mention_candidate_dict[mention_form] = candidate
                    doc_mention_candidate_dict[mention_file] = mention_candidate_dict
                else:
                    mention_candidate_dict = doc_mention_candidate_dict[mention_file]
                    if mention_form.lower() not in mention_candidate_dict:
                        mention_candidate_dict[mention_form] = candidate

        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()
                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]

                mention_candidate_dict = doc_mention_candidate_dict[mention_file]
                father_mention_list = mention_obj["father_mention"]

                if not is_cover:
                    source_candidate_list = mention_obj["candidate"]["mention_search_page"].copy()

                if len(father_mention_list) > 0:
                    mention_obj["candidate"] = mention_candidate_dict[father_mention_list[0]]

                if not is_cover:
                    mention_obj["candidate"]["source_candidate"] = source_candidate_list

                mention_obj_list.append(mention_obj)

        with open(candidate_path, "w", encoding="utf-8") as candidate_file:
            for mention_obj in mention_obj_list:
                candidate_file.write(json.dumps(mention_obj) + "\n")

    def add_aida_train2test(self, aida_train_path, aida_test_path):
        """
        add target entity in training set as candidate entity in test set
        :param aida_train_path:
        :param aida_test_path:
        :return:
        """
        mention_target_dict = {}
        with open(aida_train_path, "r", encoding="utf-8") as aida_train_file:
            for item in aida_train_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_form = mention_obj["mention_form"]
                mention_form_lower = mention_form.lower()

                target_redirect_url = mention_obj["target_redirect_url"]
                target_name = target_redirect_url.split("/")[-1]

                if mention_form_lower not in mention_target_dict:
                    mention_target_dict[mention_form_lower] = [target_name]
                elif target_name not in mention_target_dict[mention_form_lower]:
                        mention_target_dict[mention_form_lower].append(target_name)

        mention_list = []
        with open(aida_test_path, "r", encoding="utf-8") as aida_test_file:
            for item in aida_test_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_form = mention_obj["mention_form"]
                mention_form_lower = mention_form.lower()

                if mention_form_lower in mention_target_dict:
                    mention_obj["candidate"]["aida_train"] = mention_target_dict[mention_form_lower]
                else:
                    mention_obj["candidate"]["aida_train"] = []

                mention_list.append(mention_obj)

        with open(aida_test_path, "w", encoding="utf-8") as aida_test_file:
            for mention_obj in mention_list:
                aida_test_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")

    def filter_candidate(self, golden_redirect_path, person_path, filter_candidate_path, data_type, is_train=False):
        """
        using rules to filter candidates
        :param golden_redirect_path:
        :param filter_candidate_path:
        :param data_type:
        :return:
        """
        person_mention_set = self.data_util.read_person_name(person_path)

        mention_list = []
        with open(golden_redirect_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                target_redirect_url = mention_obj["target_redirect_url"]
                target_name = target_redirect_url.split("/")[-1]
                if target_name.find('#') > 0:
                    target_name = target_name.split("#")[0]

                mention_form = mention_obj["mention_form"]
                pro_mention_form = mention_form.replace("-", " ")

                candidate_dict = mention_obj["candidate"]

                # filter candidates for mention with more than 1 word
                if len(pro_mention_form.split(" ")) > 1:
                    if not mention_form.lower().__contains__("cup"):
                        for type, candidate_list in candidate_dict.items():
                            if type == "mention_keyword_search":
                                candidate_dict[type] = [key_list[:5] for key_list in candidate_list]
                            else:
                                candidate_dict[type] = candidate_list[:6]
                else:
                    for type, candidate_list in candidate_dict.items():
                        if type == "mention_keyword_search":
                            candidate_dict[type] = [key_list[:7] for key_list in candidate_list]
                        else:
                            candidate_dict[type] = candidate_list[:10]

                # filter person mention
                if mention_form.lower() in person_mention_set:
                    for type, candidate_list in candidate_dict.items():
                        if type == "mention_keyword_search":
                            keyword_search_list = []
                            for key_list in candidate_list:
                                tmp_list = []
                                for index, name in enumerate(key_list):
                                    if index > 1 and not name.replace("_", " ").lower().__contains__(mention_form.lower()):
                                        continue
                                    tmp_list.append(name)
                                keyword_search_list.append(tmp_list)
                            candidate_dict[type] = keyword_search_list
                        else:
                            tmp_list = []
                            for index, name in enumerate(candidate_list):
                                if index > 0 and not name.replace("_", " ").lower().__contains__(mention_form.lower()):
                                    continue
                                tmp_list.append(name)
                            candidate_dict[type] = tmp_list

                if is_train:
                    candidate_dict["target"] = [target_name]

                mention_obj["candidate"] = candidate_dict
                mention_list.append(mention_obj)

        with open(filter_candidate_path, "w", encoding="utf-8") as filter_candidate_file:
            for mention_obj in mention_list:
                filter_candidate_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")

    def filter_candidate_by_pageid(self, filter_candidate_path, page_id_path):
        """

        :param filter_candidate_path:
        :param page_id_path:
        :return:
        """
        page_id_dict = self.data_util.load_page_id(page_id_path)

        mention_list = []
        with open(filter_candidate_path, "r", encoding="utf-8") as filter_candidate_file:
            for item in filter_candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                target_redirect_url = mention_obj["target_redirect_url"]
                target_name = target_redirect_url.split("/")[-1]
                if target_name.find('#') > 0:
                    target_name = target_name.split("#")[0]

                mention_form = mention_obj["mention_form"]
                candidate_dict = mention_obj["candidate"]

                # filter candidate according to page_id
                if not mention_form.lower().__contains__("cup"):
                    for type, candidate_list in candidate_dict.items():
                        if type == "mention_keyword_search":
                            keyword_search_list = []
                            for key_list in candidate_list:
                                tmp_list = []
                                for index, name in enumerate(key_list):
                                    if name in page_id_dict:
                                        page_id = page_id_dict[name]
                                        if index > 2 and page_id > 20000000:
                                            continue
                                    tmp_list.append(name)
                                keyword_search_list.append(tmp_list)
                            candidate_dict[type] = keyword_search_list
                        else:
                            tmp_list = []
                            for index, name in enumerate(candidate_list):
                                if name in page_id_dict:
                                    page_id = page_id_dict[name]
                                    if index > 2 and page_id > 20000000:
                                        continue
                                    tmp_list.append(name)
                            candidate_dict[type] = tmp_list

                mention_obj["candidate"] = candidate_dict
                mention_list.append(mention_obj)

        with open(filter_candidate_path, "w", encoding="utf-8") as filter_candidate_file:
            for mention_obj in mention_list:
                filter_candidate_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")


    def process_wiki_clueweb(self, wiki_clueweb_candidate_path):
        """
        find the keyword in target name (only when wiki_clueweb is training data)
        :param wiki_clueweb_candidate_path:
        :return:
        """
        doc_target_dict = self.data_util.get_doc_mention_target(wiki_clueweb_candidate_path)

        # find the keyword in target name
        doc_keyword_dict = {}
        for mention_file, target_name_list in doc_target_dict.items():
            target_word_list = []
            for target_name in target_name_list:
                pro_target_name = self.data_util.process_candidate_name(target_name)
                target_word_list.extend(pro_target_name.split("_"))

            max_num_word = Counter(target_word_list).most_common(1)[0][0]
            doc_keyword_dict[mention_file] = max_num_word

        mention_list = []
        with open(wiki_clueweb_candidate_path, "r", encoding="utf-8") as wiki_clueweb_candidate_file:
            for item in wiki_clueweb_candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                target_name = mention_obj["target_name"]
                mention_file = mention_obj["mention_file"]
                candidate_dict = mention_obj["candidate"]

                target_keyword = doc_keyword_dict[mention_file]

                pro_target_name = self.data_util.process_candidate_name(target_name)
                if pro_target_name.__contains__(target_keyword):
                    candidate_dict["target_keyword"] = [target_name]
                    # print(target_name, target_keyword)
                else:
                    candidate_dict["target_keyword"] = []

                mention_obj["candidate"] = candidate_dict

                mention_list.append(mention_obj)

        with open(wiki_clueweb_candidate_path, "w", encoding="utf-8") as wiki_clueweb_candidate_file:
            for mention_obj in mention_list:
                wiki_clueweb_candidate_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")


    def remove_duplicate_candidate(self, mention_obj):
        """
        remove duplicate candidate
        :param mention_obj:
        :return:
        """
        candidate_dict = mention_obj["candidate"]

        candidate_pos_dict = {}
        # 每个候选按照其排序最高的来进行保留
        for type, candidate_list in candidate_dict.items():
            if type == "target":
                continue
            if type == "mention_keyword_search":
                for ele_list in candidate_list:
                    for pos, name in enumerate(ele_list):
                        if name not in candidate_pos_dict:
                            candidate_pos_dict[name] = {type: pos}
                        else:
                            pos_dict = candidate_pos_dict[name]
                            if type in pos_dict:
                                if pos < pos_dict[type]:
                                    pos_dict[type] = pos
                            else:
                                pos_dict[type] = pos
                            candidate_pos_dict[name] = pos_dict
            else:
                for pos, name in enumerate(candidate_list):
                    if name not in candidate_pos_dict:
                        candidate_pos_dict[name] = {type: pos}
                    else:
                        pos_dict = candidate_pos_dict[name]
                        if type in pos_dict:
                            if pos < pos_dict[type]:
                                pos_dict[type] = pos
                        else:
                            pos_dict[type] = pos
                        candidate_pos_dict[name] = pos_dict

        # for train data
        if "target" in candidate_dict and candidate_dict["target"][0] not in candidate_pos_dict:
            choose_index = np.random.choice(7, p=[0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01])
            candidate_pos_dict[candidate_dict["target"][0]] = {"target": choose_index}

        sort_candidate_dict = {}
        for name, pos_dict in candidate_pos_dict.items():
            if len(pos_dict) > 1:
                sort_pos = sorted(pos_dict.items(), key=lambda x:x[1])
                first_type = sort_pos[0][0]
                sort_candidate_dict[name] = {first_type:pos_dict[first_type]}
            else:
                sort_candidate_dict[name] = pos_dict

        return sort_candidate_dict

    def format_candidate(self, candidate_path, candidate_wiki_path, candidate_redirect_path, page_id_path, candidate_format_path):
        """
        format candidate data, each mention map to corresponding candidates
        :param candidate_wiki_path: contain candidate wiki info
        :param candidate_redirect_path: contain redirect entity map
        :param page_id_path: contain page_id for each entity
        :param candidate_format_path: each mention map to the candidates
        :return:
        """
        # get all entity info
        all_entity_dict = self.data_util.read_entity(candidate_wiki_path)
        # get redirect dict
        redirect_dict = self.data_util.get_redirect_map(candidate_redirect_path)
        # get page_id dict
        page_id_dict = self.data_util.load_page_id(page_id_path)


        # construct formatted data for ranking
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            with open(candidate_format_path, "w", encoding="utf-8") as candidate_format_file:
                for item in candidate_file:
                    item = item.strip()
                    mention_obj = json.loads(item)

                    mention_index = mention_obj["mention_index"]
                    target_redirect_url = mention_obj["target_redirect_url"]
                    target_name = target_redirect_url.split("/")[-1]
                    if target_name.find('#') > 0:
                        target_name = target_name.split("#")[0]
                    target_name = parse.unquote(target_name)

                    # remove duplicate candidate
                    sort_candidate_dict = self.remove_duplicate_candidate(mention_obj)

                    candidate_set = set()
                    for name, pos_dict in sort_candidate_dict.items():
                        if name in candidate_set:
                            continue

                        entity_json = {}
                        if name in all_entity_dict:
                            entity_json = all_entity_dict[name].copy()
                            if name in page_id_dict:
                                entity_json["page_id"] = page_id_dict[name]
                        elif parse.quote(name) in all_entity_dict:
                            entity_json = all_entity_dict[parse.quote(name)].copy()
                            if parse.quote(name) in page_id_dict:
                                entity_json["page_id"] = page_id_dict[parse.quote(name)]
                        elif name.lower() in redirect_dict:
                            redirect_name = redirect_dict[name.lower()]
                            if redirect_name in all_entity_dict:
                                entity_json = all_entity_dict[redirect_name].copy()
                                if name in page_id_dict:
                                    entity_json["page_id"] = page_id_dict[name]

                        if "name" in entity_json:
                            name = parse.unquote(entity_json["name"])
                            # is not true entity
                            if self.data_util.is_fake_entity(entity_json):
                                entity_json["is_fake_entity"] = True
                                continue

                        # positive:1; negative:0
                        if name.lower() == target_name.lower() or \
                                parse.unquote(target_name.lower()) == parse.unquote(name.lower()):
                            label = 1
                        else:
                            label = 0

                        # candidate
                        if len(entity_json) > 0:
                            entity_json["content"] = ""
                            entity_json["infobox"] = ""
                            entity_json["get_source"] = list(pos_dict.keys())[0]
                            entity_json["source_position"] = list(pos_dict.values())[0] + 1
                            if "summary" in entity_json:
                                raw_summary = self.data_util.filter_wiki_summary(entity_json["summary"])
                                summary = self.data_util.clean_summary(raw_summary)
                                summary_keywords = self.data_util.extract_keyword(summary)
                                entity_json["summary"] = ""
                                entity_json["summary_keywords"] = summary_keywords
                            else:
                                entity_json["summary"] = ""
                                entity_json["summary_keywords"] = []

                            if "page_id" not in entity_json:
                                entity_json["page_id"] = 0

                            candidate_format_file.write(str(mention_index) + "\t" + str(label) + "\t"
                                                        + item + "\t" + json.dumps(entity_json, ensure_ascii=False) + "\n")
                            candidate_format_file.flush()
                        else:
                            if label == 1:
                                print("read candidate entity error, target: {0}, candidate: {1}".format(target_name, name))

                        candidate_set.add(name)

    def static_candidate(self, candidate_format_path):
        """
        static mention num and gold entity num in data
        :param candidate_format_path:
        :return:
        """
        mention_label_dict = {}
        target_dict = {}
        mention_candidate_dict = {}
        with open(candidate_format_path, "r", encoding="utf-8") as candidate_format_file:
            for item in candidate_format_file:
                item = item.strip()

                mention_index_str, label_str, mention_str, entity_str = item.split("\t")
                mention_index = int(mention_index_str)
                label = int(label_str)
                mention_obj = json.loads(mention_str)

                if mention_index not in mention_label_dict:
                    mention_label_dict[mention_index] = [label]
                else:
                    mention_label_dict[mention_index].append(label)

                target_dict[mention_index] = [mention_obj["mention_form"], mention_obj["target_url"], mention_obj["target_redirect_url"]]
                # if mention_index not in mention_candidate_dict:
                #     mention_candidate_dict[mention_index] = [entity_str]
                # else:
                #     mention_candidate_dict[mention_index].append(entity_str)

        recall_num = 0
        for mention_index, label_list in mention_label_dict.items():
            if 1 in label_list:
                recall_num += 1
            else:
                print(target_dict[mention_index])
        print("recall num:{0}, all mention num:{1}".format(recall_num, len(mention_label_dict)))

    def process(self):
        data_name = "ace2004"
        source_dir = "/data/fangzheng/bert_el/"
        person_path = source_dir + "persons.txt"
        candidate_path = source_dir + data_name + "/source/" + data_name + "_golden_redirect"
        filter_candidate_path = source_dir + data_name + "/source/" + data_name + "_filter_candidate"
        wiki_path = source_dir + data_name + "/candidate/" + data_name + "_wiki"
        page_id_path = source_dir + data_name + "/source/" + data_name + "_filter_candidate_pageid"
        redirect_path = source_dir + data_name + "/candidate/" + data_name + "_candidate_redirect"
        candidate_format_path = source_dir + data_name + "/candidate/" + data_name + "_candidate_format"

        # self.combine_keyword_candidate(candidate_path)
        # self.get_father_candidate(candidate_path, is_cover=True)
        # if data_name.__contains__("aida_test"):
        #     aida_train_candidate_path = source_dir + "aida_train/source/aida_train_golden_redirect"
        #     self.add_aida_train2test(aida_train_candidate_path, candidate_path)
        # self.filter_candidate(candidate_path, person_path, filter_candidate_path, data_name, is_train=False)
        # self.filter_candidate_by_pageid(filter_candidate_path, page_id_path)

        if data_name == "wiki_clueweb":
            self.process_wiki_clueweb(filter_candidate_path)

        self.format_candidate(filter_candidate_path, wiki_path, redirect_path, page_id_path, candidate_format_path)
        self.static_candidate(candidate_format_path)

        self.data_util.cal_candidate_recall(filter_candidate_path)




if __name__ == "__main__":
    data_util = DataUtil()
    candidate_control = CandidateControl(data_util)

    candidate_control.process()
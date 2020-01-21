# encoding:utf-8


import json
import re
import os
import time
import sys
from nltk.corpus import wordnet as wn
from datetime import timedelta
import config_util
from urllib import parse

class DataUtil(object):
    def __init__(self):
        pass


    def get_time_dif(self, start_time):
        """
        get run time
        :param start_time: 起始时间
        :return:
        """
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def capitalize_mention(self, mention_form):
        """
        capitalize mention form
        :param mention_form:
        :return:
        """
        word_list = []
        for word in mention_form.strip().split(" "):
            word_list.append(word.capitalize())
        capitalize_mention_form = " ".join(word_list)

        return capitalize_mention_form

    def replace_keyword(self, mention_context):
        """
        replace keyword in mention context
        :param mention_context:
        :return:
        """
        mention_context = mention_context.replace("soccer", "football").replace("nfl", "football").replace("nba", "basketball")
        return mention_context

    def is_extract_year(self, mention_form):
        """
        is need to extract year from context
        :param mention_form:
        :return:
        """
        is_extract_year = False
        year_keywords = ["cup", "olympic"]

        for word in year_keywords:
            if mention_form.lower().__contains__(word):
                is_extract_year = True
                break

        return is_extract_year

    def extract_year(self, mention_obj):
        """
        extract year in mention context or document
        :param mention_obj:
        :return:
        """
        year = ""

        if "mention_context" in mention_obj:
            mention_context = mention_obj["mention_context"]
            context_match = re.findall(r'\d{4}', mention_context)
            if len(context_match) > 0:
                year = context_match[0]

        if "doc_context" in mention_obj:
            doc_context = mention_obj["doc_context"]
            doc_match = re.findall(r'\d{4}[-]\d{2}[-]\d{2}', doc_context)
            if len(doc_match) > 0:
                year = doc_match[0].split("-")[0]

        return year

    def get_synonym(self, name):
        """
        get synonym by WordNet
        :param name:
        :return:
        """
        name_synset = set()

        word_synsets = wn.synsets(name)
        for synset in word_synsets:
            words = synset.lemma_names()
            for word in words:
                word = word.replace('_', ' ')
                name_synset.add(word)

        return list(name_synset)

    def read_mention_candidate(self, candidate_path):
        """
        read mention candidate set
        mention_candidate_dict[mention]=candidate_dict
        :param candidate_path:
        :return:
        """
        mention_candidate_dict = {}
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                mention, candidate_str = item.strip().split("\t")
                candidate_obj = json.loads(candidate_str)
                mention_candidate_dict[mention] = candidate_obj

        return mention_candidate_dict

    def read_person_name(self, person_path):
        """
        read person file which contains 300M+ name
        :param person_path:
        :return:
        """
        person_mention_set = set()
        with open(person_path, "r", encoding="utf-8") as person_file:
            for item in person_file:
                item = item.strip().lower()
                person_mention_set.add(item)

        return person_mention_set

    def read_prior_score(self, prior_path):
        """

        :param aida_mean_path:
        :return:
        """
        prior_dict = {}
        with open(prior_path, "r", encoding="utf-8") as aida_mean_file:
            for item in aida_mean_file:
                item = item.strip()

                entity_name, val_str = item.split("\t")
                val_dict = json.loads(val_str)
                prior_dict[entity_name] = val_dict

        return prior_dict

    def search_father_mention(self, mention, doc_mention_list):
        """
        for each mention m, if there exist mentions that contain m as a continuous subsequence of words,
        then we consider the merged set of the candidate sets of these specific mentions
        as the candidate set for the mention m
        :param mention:
        :param doc_mention_list:
        :return:
        """
        father_mention_list = []

        for item in doc_mention_list:
            item = item.strip()
            item_lower = item.lower()
            if item_lower.__contains__(mention.lower()) and len(item) > len(mention):
                father_mention_list.append(item)

        return father_mention_list

    def read_doc_mention(self, golden_path):
        """
        read all mentions in the same doc
        :param golden_path:
        :return:
        """
        doc_mention_dict = {}

        with open(golden_path, "r", encoding="utf-8") as golden_file:
            for item in golden_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]
                mention_form = mention_obj["mention_form"]

                if mention_file not in doc_mention_dict:
                    doc_mention_dict[mention_file] = [mention_form]
                else:
                    doc_mention_dict[mention_file].append(mention_form)

        return doc_mention_dict

    def get_doc_mention_index(self, mention_data_path):
        """
        read all mentions in the same doc
        :param mention_data_path:
        :return:
        """
        doc_mention_dict = {}
        with open(mention_data_path, "r", encoding="utf-8") as mention_data_file:
            for item in mention_data_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]
                mention_index = mention_obj["mention_index"]

                if mention_file not in doc_mention_dict:
                    doc_mention_dict[mention_file] = [mention_index]
                else:
                    doc_mention_dict[mention_file].append(mention_index)

        return doc_mention_dict

    def get_doc_mention_target(self, mention_data_path):
        """
        read all target entities in the same doc
        :param mention_data_path:
        :return:
        """
        doc_mention_dict = {}
        with open(mention_data_path, "r", encoding="utf-8") as mention_data_file:
            for item in mention_data_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]
                target_name = mention_obj["target_name"]

                if mention_file not in doc_mention_dict:
                    doc_mention_dict[mention_file] = [target_name]
                else:
                    doc_mention_dict[mention_file].append(target_name)

        return doc_mention_dict

    def get_all_mention(self, mention_data_path):
        """
        get mention_dict[mention_index] = mention_obj
        :param mention_data_path:
        :return:
        """
        all_mention_dict = {}

        with open(mention_data_path, "r", encoding="utf-8") as mention_data_file:
            for item in mention_data_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_index = mention_obj["mention_index"]

                all_mention_dict[mention_index] = mention_obj

        return all_mention_dict


    def filter_wiki_summary(self, summary_text):
        """
         Because the previous crawler had problems, resulting in summary analysis errors,
         (the problem was fixed and filtering was not needed for new data)
        :param summary_text:
        :return:
        """
        new_summary = summary_text

        line_list = summary_text.split("\n")
        start_index = 0
        for index, line in enumerate(line_list):
            line = line.strip()
            if index == len(line_list) - 1:
                continue

            if line == "}}":
                if (len(line_list[index + 1]) > 0 and line_list[index + 1][0] != "|") \
                        or len(line_list[index + 1]) == 0:
                    start_index = index + 1
            elif len(line) > 1 and line[0] == "|" and "".join(line[-2:]) == "}}":
                if (len(line_list[index + 1]) > 0 and line_list[index + 1][0] != "|") \
                        or len(line_list[index + 1]) == 0:
                    start_index = index + 1

        summary_list = line_list[start_index:]
        if len(summary_list) > 0:
            new_summary = "\n".join(summary_list)

        return new_summary

    def clean_summary(self, summary_text):
        """
        clean summary data
        :param summary_text:
        :return:
        """
        line_list = summary_text.split("\n")

        new_line_list = []
        for line in line_list:
            line = line.strip()
            if "".join(line[:7]).lower() == "[[image":
                continue
            elif "".join(line[:6]).lower() == "[[file":
                continue

            new_line_list.append(line)

        summary = " ".join(new_line_list)

        # summary = re.sub(u"[^a-zA-Z ]", "", summary)

        summary = re.sub(u"[\n.,?!;:$*/'\\#\"\(\)\{\}\<\>{0-9}]", "", summary)
        summary = re.sub(u"[=|\-]", " ", summary)

        return summary

    def extract_keyword(self, summary_text):
        """
        extract keyword from summary_text
        :param summary_text:
        :return:
        """
        keyword_list = []

        wiki_keywords = re.findall(r'\[\[(.*?)\]\]', summary_text)
        for wiki_item in wiki_keywords:
            keyword_list.extend(wiki_item.lower().split("|"))

        # self.rake.extract_keywords_from_text(re.sub(r'[^\w\s]', "", summary_text))
        # summary_keywords = self.rake.get_ranked_phrases()
        # keyword_list.extend(summary_keywords)

        # filter duplicate keywords
        keyword_set = set()
        filter_keyword_list = []
        for keyword in keyword_list:
            if keyword in keyword_set:
                continue
            else:
                keyword = re.sub(u"[\n.,?!;:$*/'\\#\"\(\)\[\]\{\}\<\>{0-9}]", "", keyword)
                filter_keyword_list.append(keyword)
                keyword_set.add(keyword)

        return filter_keyword_list

    def read_entity(self, entity_path):
        """
        read entity info
        :param entity_path:
        :param redirect_path:
        :return:
        """
        print("Loading entity data...")
        start_time = time.time()

        entity_dict = {}
        with open(entity_path, "r", encoding="utf-8") as entity_file:
            for item in entity_file:
                item = item.strip()

                if len(item.split("\t")) != 2:
                    continue

                entity_name, entity_str = item.split("\t")

                entity_obj = json.loads(entity_str)

                entity_dict[entity_name] = entity_obj

        run_time = self.get_time_dif(start_time)
        print("Time usage:{0}, Memory usage: {1} GB".format(run_time, int(sys.getsizeof(entity_dict)/(1024*1024))))

        return entity_dict

    def get_redirect_map(self, redirect_path):
        """
        read redirect entity for source entity
        :param redirect_path:
        :return:
        """
        redirect_dict = {}

        with open(redirect_path, "r", encoding="utf-8") as redirect_file:
            for item in redirect_file:
                item = item.strip()

                source_name, redirect_name = item.split("\t")
                if source_name != redirect_name:
                    redirect_dict[source_name.lower()] = redirect_name

        return redirect_dict

    def is_fake_entity(self, entity_json):
        """
        some entity pages are disambiguation pages that need to be removed
        :param entity_json:
        :return:
        """
        is_fake = False
        name = entity_json["name"]

        if "summary" in entity_json:
            summary = entity_json["summary"].lower()
            if summary.__contains__("may also refer to:") or summary.__contains__("may refer to:"):
                is_fake = True

                print("fake entity: {0}".format(name))

        return is_fake

    def remove_stop_word(self, text):
        """
        remove stop word from text
        :param text:
        :return:
        """
        stop_word_set = self.load_stop_words()

        new_text = [word for word in text.split(" ") if word.lower() not in stop_word_set and word != ""]

        return " ".join(new_text)

    def load_stop_words(self):
        """
        load english stop words
        :param stop_word_path:
        :return:
        """
        stop_word_set = set()
        with open(config_util.stop_word_path, "r", encoding="utf-8") as stop_word_file:
            for item in stop_word_file:
                item = item.strip().lower()
                stop_word_set.add(item)

        return stop_word_set

    def process_candidate_name(self, name):
        """

        :param name:
        :return:
        """
        name = name.lower().replace("(", "").replace(")", "").replace(".", "").replace(",", "")
        return name


    def load_page_id(self, page_id_path):
        """

        :param page_id_path:
        :return:
        """
        name_pageid_dict = {}
        with open(page_id_path, "r", encoding="utf-8") as page_id_file:
            for item in page_id_file:
                item = item.strip()

                name, page_id_str = item.split("\t")
                name_pageid_dict[name] = int(page_id_str)

        return name_pageid_dict

    def cal_candidate_recall(self, mention_path):
        """

        :param mention_path:
        :return:
        """
        recall_count = 0
        valid_recall_count = 0
        valid_count = 0
        all_count = 0
        with open(mention_path, "r", encoding="utf-8") as mention_file:
            for index, item in enumerate(mention_file):
                item = item.strip()
                mention_obj = json.loads(item)

                mention_form = mention_obj["mention_form"]
                redirect_target_url = mention_obj["target_redirect_url"]
                target_name = redirect_target_url.split("/")[-1]
                if target_name.__contains__("#"):
                    target_name = target_name.split("#")[0]

                mention_candidate_dict = mention_obj["candidate"]

                all_candidate_list = []
                for candidate_type, candidate_list in mention_candidate_dict.items():
                    if candidate_type == "mention_keyword_search":
                        for key_list in candidate_list:
                            all_candidate_list.extend(key_list)
                    else:
                        all_candidate_list.extend(candidate_list)

                # add_dism_candidate = self.add_dism_candidate(mention_form)
                # if add_dism_candidate != "":
                #     all_candidate_list.append(add_dism_candidate)

                parse_candidate_list = [parse.unquote(candidate) for candidate in all_candidate_list]
                if parse.unquote(target_name) in parse_candidate_list or target_name.__contains__("disambiguation") \
                        or target_name == "":
                    recall_count += 1
                else:
                    # print(mention_form, target_name, mention_candidate_dict, mention_obj["context_keyword"])
                    print(mention_obj["mention_index"], mention_form, target_name, mention_obj["context_keyword"], mention_obj["mention_context"])

                if (not target_name.__contains__("disambiguation")) and target_name != "":
                    valid_count += 1

                    if parse.unquote(target_name) in parse_candidate_list:
                        valid_recall_count += 1

                all_count += 1

        print("all count:{0}, valid_count:{1}, valid_recall_count:{2}, "
              "recall count:{3}, recall:{4}, valid_recall:{5}".format(all_count, valid_count, valid_recall_count,
                                                                      recall_count, recall_count / all_count,
                                                                      valid_recall_count / valid_count))

if __name__ == "__main__":
    pass
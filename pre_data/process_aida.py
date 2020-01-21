# encoding: utf-8

import xml.dom.minidom as xmldom
import re
import os
import json
# from rake_nltk import Rake  # self.rake = Rake(max_length=2)
import string
from candidate_analyse import CandidateAnalyse
from base_process import BaseProcess

class ProcessAIDA(BaseProcess):
    """
    process aida data for local entity linking
    """

    def get_golden_data(self, source_data_path, golden_path, data_type):
        """
        get golden data
        :param source_data_path:
        :param golden_path:
        :return:
        """
        with open(source_data_path, "r", encoding="utf-8") as source_aida_file:

            doc_index = 0
            mention_index = 0
            mention_list = []
            for item in source_aida_file:
                item = item.strip()

                if item.__contains__("DOCSTART"):
                    file_name = data_type + "_" + str(doc_index)
                    doc_index += 1
                    continue

                data_list = item.split("\t")
                if len(data_list) > 3 and data_list[1] == "B":
                    mention_form = data_list[2]

                    if data_list[3] == "--NME--":
                        target_url = "http://aksw.org/notInWiki/" + "+".join(mention_form.split(" "))
                    else:
                        target_url = data_list[4]
                        if not target_url.__contains__("http"):
                            target_url = "http://" + target_url

                    mention_obj = {}
                    mention_obj["mention_file"] = file_name
                    mention_obj["mention_form"] = mention_form
                    mention_obj["target_url"] = target_url
                    mention_list.append(mention_obj)

                    mention_index += 1

        with open(golden_path, "w", encoding="utf-8") as golden_file:
            for mention_obj in mention_list:
                golden_file.write(json.dumps(mention_obj) + "\n")

    def get_doc_sent(self, file_path, file_type):
        """
        get sent in doc
        :param file_path:
        :return: doc[file]=sent_dict
        """
        # doc[file]=word_list
        doc_words = {}
        with open(file_path, "r", encoding="utf-8") as conll_file:

            doc_index = 0
            word_list = []
            for item in conll_file:
                item = item.strip()

                if item.__contains__("DOCSTART"):
                    if len(word_list) > 0:
                        file_name = file_type + "_" + str(doc_index)
                        doc_words[file_name] = word_list
                        doc_index += 1
                        word_list = []

                    continue

                word_list.append(item.split("\t")[0])

            # last doc
            doc_words[file_type + "_" + str(doc_index)] = word_list

        # doc[file]=sent_dict
        doc_sents = {}
        for file_name, word_list in doc_words.items():

            other_len = 0
            sent = []
            sent_list = []
            sent_dict = {}
            for index, word in enumerate(word_list):
                # sent end
                if word == "":
                    sent_list.append(sent)

                    # match date
                    if re.match(r'\d{4}[-]\d{2}[-]\d{2}', word_list[index - 1]) and file_type == "aida":
                        other_len = len(sent_list) - 1
                        sent_dict["other"] = sent_list[1:]

                    sent = []
                    continue

                sent.append(word)

            sent_dict["title"] = sent_list[0]
            if "other" not in sent_dict:
                sent_dict["other"] = []
            sent_dict["content"] = sent_list[other_len + 1:]
            sent_dict["all"] = sent_list
            doc_sents[file_name] = sent_dict

        return doc_sents

    def pre_aida_data(self, source_data_path, data_type):
        """
        read aida and aquaint data
        :param self:
        :param source_data_path:
        :param golden_path:
        :param person_path:
        :param data_type: "aida" or "aquaint"
        :return:
        """
        # read sent in aida doc
        doc_sents = self.get_doc_sent(source_data_path, data_type)

        with open(source_data_path, "r", encoding="utf-8") as source_aida_file:
            doc_index = 0
            offset = 0
            sent_index = 0
            file_name = ""
            mention_list = []

            # according to gerbil platform request, for calculating mention offset
            quoteCharSeenBefore = False
            whiteSpaceBehind = True

            for item in source_aida_file:
                item = item.strip()

                if item.__contains__("DOCSTART"):
                    file_name = data_type + "_" + str(doc_index)
                    doc_index += 1
                    sent_index = 0
                    offset = 0
                    quoteCharSeenBefore = False
                    continue

                if item == "":
                    sent_index += 1
                    continue

                # calculate mention offset
                whiteSpaceInFront = whiteSpaceBehind
                whiteSpaceBehind = True
                if offset > 0 and len(item.split("\t")[0]) >= 1:
                    if len(item.split("\t")[0]) == 1:
                        char = item[0]
                        if char in ["?", "!", ',', ')', ']', '}', '.']:
                            whiteSpaceInFront = False

                        elif char == "\"":
                            # if we have seen another quote char before
                            if not quoteCharSeenBefore:
                                whiteSpaceBehind = False
                            else:
                                whiteSpaceInFront = False
                            quoteCharSeenBefore = not quoteCharSeenBefore

                        elif char in ['(', '[', '{']:
                            whiteSpaceBehind = False

                    elif not (item[0].isdigit() or item[0].isalpha()):
                        whiteSpaceInFront = False

                    if whiteSpaceInFront:
                        offset += 1

                data_list = item.split("\t")
                if len(data_list) > 3 and data_list[1] == "B":
                    mention_form = data_list[2]

                    if data_list[3] == "--NME--":
                        target_url = "http://aksw.org/notInWiki/" + "+".join(mention_form.split(" "))
                    else:
                        target_url = data_list[4]
                        if not target_url.__contains__("http"):
                            target_url = "http://" + target_url

                    # identify the location of mention in the article
                    if sent_index == 0:
                        mention_locate = "title"
                    elif sent_index < len(doc_sents[file_name]["other"]) + 1:
                        mention_locate = "other"
                    else:
                        mention_locate = "content"

                    # construct mention context
                    sent_dict = doc_sents[file_name]
                    if mention_locate == "title":
                        mention_context = " ".join(sent_dict["title"][:-1])
                        if data_type == "aida":
                            mention_context += (" " + sent_dict["title"][0]) * 3
                    elif mention_locate == "other":
                        mention_context = ""
                    else:
                        content_index = sent_index - len(doc_sents[file_name]["other"]) - 1

                        mention_context = []
                        mention_context.append(sent_dict["title"])
                        mention_context.append(sent_dict["content"][content_index])
                        if data_type == "aida":
                            mention_context.append([sent_dict["title"][0] for i in range(4)])
                        mention_context = ". ".join([" ".join(sent[:-1]) for sent in mention_context])

                    mention_obj = {}
                    mention_obj["mention_file"] = file_name
                    mention_obj["mention_form"] = mention_form
                    mention_obj["target_url"] = target_url
                    mention_obj["mention_offset"] = offset
                    mention_obj["mention_sent_index"] = sent_index
                    mention_obj["mention_locate"] = mention_locate
                    mention_obj["mention_context"] = mention_context
                    mention_obj["doc_context"] = " ".join([" ".join(word_list) for word_list in sent_dict["all"]])

                    mention_list.append(mention_obj)

                offset += len(item.split("\t")[0])

        source_dir = "/".join(source_data_path.split("/")[:-1])
        if data_type == "aida":
            train_path = source_dir + "/aida_train"
            testA_path = source_dir + "/aida_testA"
            testB_path = source_dir + "/aida_testB"
            with open(train_path, "w", encoding="utf-8") as train_file:
                with open(testA_path, "w", encoding="utf-8") as testA_file:
                    with open(testB_path, "w", encoding="utf-8") as testB_file:
                        for mention_obj in mention_list:
                            if int(mention_obj["mention_file"].split("_")[-1]) < 946:
                                train_file.write(json.dumps(mention_obj) + "\n")
                            elif 946 <= int(mention_obj["mention_file"].split("_")[-1]) < 1162:
                                if mention_obj["target_url"].__contains__("notInWiki"):
                                    target_url = mention_obj["target_url"]
                                    mention_obj["target_url"] = "http://AIDA/CoNLL-Test A/notInWiki/" + \
                                                                target_url.split("/")[-1]
                                testA_file.write(json.dumps(mention_obj) + "\n")
                            elif 1162 <= int(mention_obj["mention_file"].split("_")[-1]) < 1393:
                                if mention_obj["target_url"].__contains__("notInWiki"):
                                    target_url = mention_obj["target_url"]
                                    mention_obj["target_url"] = "http://AIDA/CoNLL-Test B/notInWiki/" + \
                                                                target_url.split("/")[-1]

                                testB_file.write(json.dumps(mention_obj) + "\n")

        elif data_type == "aquaint":
            format_path = source_dir + "/aquaint"
            with open(format_path, "w", encoding="utf-8") as format_file:
                for mention_obj in mention_list:
                    format_file.write(json.dumps(mention_obj) + "\n")

    def split_aida(self):
        process_aida.get_golden_data("/home1/fangzheng/data/bert_el_data/aida/source/AIDA-YAGO2-dataset.tsv",
                                     "/home1/fangzheng/data/bert_el_data/aida/generate/aida_golden", "aida")
        process_aida.pre_aida_data("/home1/fangzheng/data/bert_el_data/aida/source/AIDA-YAGO2-dataset.tsv", "aida")

        pass

    def control_aida(self):
        data_name = "aida_testB"

        source_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/source/" + data_name
        in_wiki_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/source/" + data_name + "_in_wiki"
        person_path = "/home1/fangzheng/data/bert_el_data/persons.txt"

        # self.filter_not_inkb_data(source_path, in_wiki_path)
        # self.get_father_mention(in_wiki_path, person_path)
        # self.get_adjacent_mention(in_wiki_path)
        # self.get_mention_synonym(in_wiki_path)
        # self.get_context_keyword(in_wiki_path)

        # crawl redirect page
        # crawl candidate

        pass

    def control_aquaint(self):
        data_name = "aquaint"
        source_dir = "/data/fangzheng/bert_el/"

        raw_path = source_dir + data_name + "/source/" + data_name + ".conll"
        source_path = source_dir + data_name + "/source/" + data_name
        in_wiki_path = source_dir + data_name + "/source/" + data_name + "_in_wiki"
        person_path = source_dir + "persons.txt"

        self.pre_aida_data(raw_path, data_name)
        self.filter_not_inkb_data(source_path, in_wiki_path)
        self.get_father_mention(in_wiki_path, person_path)
        self.get_adjacent_mention(in_wiki_path)
        self.get_mention_synonym(in_wiki_path)
        self.get_context_keyword(in_wiki_path)

        # crawl redirect page
        # crawl candidate



        pass

if __name__ == "__main__":
    process_aida = ProcessAIDA()

    process_aida.control_aquaint()

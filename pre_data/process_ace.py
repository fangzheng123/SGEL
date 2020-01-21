# encoding:utf-8

import os
import re
import json
import xml.dom.minidom as xmldom
from base_process import BaseProcess
from candidate_analyse import CandidateAnalyse

class ProcessACE(BaseProcess):
    """
    Process Ace2004 and MSNBC dataset
    """

    def read_xml_file(self, data_path):
        """
        read NIF file in Gerbil
        :param data_path: data dir
        :return:
        """
        pattern = re.compile(r'\n|\t')

        data_list = []

        dom_obj = xmldom.parse(data_path)
        element_obj = dom_obj.documentElement

        file_name = element_obj.getElementsByTagName("ReferenceFileName")[0].firstChild.nodeValue
        file_name = re.sub(pattern, "", file_name)

        mention_list = element_obj.getElementsByTagName("ReferenceInstance")
        for mention in mention_list:

            mention_form = mention.getElementsByTagName("SurfaceForm")[0].firstChild.nodeValue
            mention_form = re.sub(pattern, "", mention_form)

            mention_offset = mention.getElementsByTagName("Offset")[0].firstChild.nodeValue
            mention_offset = re.sub(pattern, "", mention_offset)

            mention_len = mention.getElementsByTagName("Length")[0].firstChild.nodeValue
            mention_len = re.sub(pattern, "", mention_len)

            target_url = mention.getElementsByTagName("ChosenAnnotation")[0].firstChild.nodeValue
            target_url = re.sub(pattern, "", target_url)

            if target_url == "*null*":
                # target_url = "http://aksw.org/notInWiki/" + "+".join(mention_form.split(" "))
                continue

            json_data = {}
            json_data["mention_file"] = file_name
            json_data["mention_form"] = mention_form
            json_data["mention_offset"] = mention_offset
            json_data["mention_len"] = mention_len
            json_data["target_url"] = target_url

            data_list.append(json_data)
        return data_list

    def read_xml_dir_data(self, dir_path, result_path):
        """
        read dir in Gerbil
        :param dir_path:
        :param result_path:
        :return:
        """
        mention_num = 0
        mention_index = 0
        for root, dirs, files in os.walk(dir_path):
            with open(result_path, "w", encoding="utf-8") as result_file:
                for item_path in files:
                    data_list = self.read_xml_file(os.path.join(root, item_path))
                    mention_num += len(data_list)

                    for data_item in data_list:
                        data_item["mention_index"] = mention_index
                        result_file.write(json.dumps(data_item, ensure_ascii=False) + "\n")

                        mention_index += 1

        print(mention_num, mention_index)

    def read_ace_raw_file(self, file_path):
        """
        read ace2004 raw file
        :param file_path:
        :return: sent_dict[sent_index] = {content: val, sent_offset: val}
        """
        sent_dict = {}

        with open(file_path, "r", encoding="utf-8") as raw_file:
            raw_text = raw_file.readline()

            sent_list = raw_text.split("  ")
            last_sent_offset = -2
            last_sent_len = 0
            for sent_index, sent in enumerate(sent_list):
                tmp_dict = {}
                tmp_dict["content"] = sent
                tmp_dict["sent_offset"] = last_sent_offset + last_sent_len + 2

                sent_dict[sent_index] = tmp_dict

                last_sent_offset = tmp_dict["sent_offset"]
                last_sent_len = len(sent)

        return sent_dict, raw_text

    def pre_ace_data(self, raw_test_dir, in_wiki_path):
        """

        :param raw_test_dir:
        :param in_wiki_path:
        :return:
        """
        file_dict = {}
        test_dict = {}
        mention_obj_list = []
        with open(in_wiki_path, "r", encoding="utf-8") as in_wiki_file:
            for item in in_wiki_file:
                item = item.strip()
                mention_obj = json.loads(item)

                raw_file = mention_obj["mention_file"]

                if raw_file not in file_dict:
                    raw_path = raw_test_dir + "/" + raw_file
                    sent_dict, raw_text = self.read_ace_raw_file(raw_path)
                    file_dict[raw_file] = sent_dict
                    test_dict[raw_file] = raw_text
                else:
                    sent_dict = file_dict[raw_file]
                    raw_text = test_dict[raw_file]

                mention_obj["doc_context"] = raw_text

                mention_offset = int(mention_obj["mention_offset"])
                mention_len = int(mention_obj["mention_len"])

                for sent_index, content_dict in sent_dict.items():
                    sent_offset = content_dict["sent_offset"]
                    content = content_dict["content"]
                    if sent_index == 0:
                        pre_content = ""
                    else:
                        pre_content = sent_dict[sent_index-1]["content"]

                    if sent_index == len(sent_dict) - 1:
                        next_content = ""
                    else:
                        next_content = sent_dict[sent_index+1]["content"]

                    if mention_offset >= sent_offset:
                        if ((sent_index + 1 in sent_dict) and mention_offset < sent_dict[sent_index + 1]["sent_offset"]) \
                                or (sent_index == len(sent_dict) - 1):
                            mention_sent_offset = mention_offset - sent_offset
                            mention = content[mention_sent_offset: mention_sent_offset + mention_len]

                            if mention == mention_obj["mention_form"]:
                                mention_obj["mention_sent_index"] = sent_index
                                mention_obj["mention_context"] = pre_content + " " + content + " " + next_content
                                if (sent_index == 0 or sent_index == 1) \
                                        and (content.__contains__("(AP)") or content.__contains__("(AFP)")):
                                    mention_obj["locate"] = "title"
                                else:
                                    mention_obj["locate"] = "content"

                                mention_obj_list.append(mention_obj)

                            else:
                                print(raw_file, mention, mention_obj["mention_form"])

        with open(in_wiki_path, "w", encoding="utf-8") as in_wiki_file:
            for mention_obj in mention_obj_list:
                in_wiki_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")

    def read_msnbc_raw_file(self, file_path):
        """
        read msnbc raw file
        :param file_path:
        :return:
        """
        sent_dict = {}
        with open(file_path, "r", encoding="utf-8") as raw_file:
            sent_list = raw_file.readlines()

        sent_offset = 0
        sent_index = 0
        for sent in sent_list:
            tmp_sent = sent.strip()

            if len(tmp_sent) != 0:
                tmp_dict = {}
                tmp_dict["content"] = tmp_sent
                tmp_dict["sent_offset"] = sent_offset
                sent_dict[sent_index] = tmp_dict

                sent_index += 1

            sent_offset += len(sent)

        return sent_dict, " ".join(sent_list)

    def pre_msnbc_data(self, raw_test_dir, in_wiki_path):
        """

        :param golden_path:
        :param raw_test_dir:
        :param pre_path:
        :return:
        """
        file_dict = {}
        text_dict = {}
        mention_obj_list = []
        with open(in_wiki_path, "r", encoding="utf-8") as in_wiki_file:
            for item in in_wiki_file:
                item = item.strip()
                mention_obj = json.loads(item)

                raw_file = mention_obj["mention_file"]
                mention_form = mention_obj["mention_form"]

                if raw_file not in file_dict:
                    raw_path = raw_test_dir + "/" + raw_file
                    sent_dict, doc_context = self.read_msnbc_raw_file(raw_path)
                    file_dict[raw_file] = sent_dict
                    text_dict[raw_file] = doc_context
                else:
                    sent_dict = file_dict[raw_file]
                    doc_context = text_dict[raw_file]

                mention_obj["doc_context"] = doc_context

                mention_offset = int(mention_obj["mention_offset"])
                mention_len = int(mention_obj["mention_len"])

                for sent_index, content_dict in sent_dict.items():
                    sent_offset = content_dict["sent_offset"]
                    content = content_dict["content"]

                    if mention_offset >= sent_offset:
                        if ((sent_index + 1 in sent_dict) and mention_offset < sent_dict[sent_index + 1][
                            "sent_offset"]) \
                                or (sent_index == len(sent_dict) - 1):
                            mention_sent_offset = mention_offset - sent_offset
                            mention = content[mention_sent_offset: mention_sent_offset + mention_len]
                            mention = mention.replace("’", "'")

                            if mention[0] == mention_form[0] and \
                                    mention[-1] == mention_form[-1]:
                                mention_obj["mention_sent_index"] = sent_index

                                if sent_index == 0:
                                    mention_obj["mention_context"] = content
                                    mention_obj["locate"] = "title"
                                else:
                                    mention_obj["mention_context"] = sent_dict[0]["content"] + " " + content
                                    mention_obj["locate"] = "content"

                                mention_obj_list.append(mention_obj)

                            else:
                                print(raw_file, mention, mention_obj["mention_form"])

        with open(in_wiki_path, "w", encoding="utf-8") as in_wiki_file:
            for mention_obj in mention_obj_list:
                in_wiki_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")

    def control_ace(self):
        """

        :return:
        """
        source_dir = "/data/fangzheng/bert_el/"
        data_name = "ace2004"
        xml_path = source_dir + data_name + "/source/" + "ProblemsNoTranscripts"
        raw_dir = source_dir + data_name + "/source/" + "RawTextsNoTranscripts"
        in_wiki_path = source_dir + data_name + "/source/" + data_name + "_in_wiki"
        person_path = "/data/fangzheng/bert_el/persons.txt"

        self.read_xml_dir_data(xml_path, in_wiki_path)
        self.pre_ace_data(raw_dir, in_wiki_path)
        self.filter_not_inkb_data(in_wiki_path, in_wiki_path)
        self.get_father_mention(in_wiki_path, person_path)
        self.get_adjacent_mention(in_wiki_path)
        self.get_mention_synonym(in_wiki_path)
        self.get_context_keyword(in_wiki_path)

        # crawl redirect page
        # crawl candidate
        # crawl wiki data

        pass

    def control_msnbc(self):
        """

        :return:
        """

        source_dir = "/data/fangzheng/bert_el/"
        data_name = "msnbc"
        xml_path = source_dir + data_name + "/source/" + "Problems"
        raw_dir = source_dir + data_name + "/source/" + "RawTextsSimpleChars_utf8"
        in_wiki_path = source_dir + data_name + "/source/" + data_name + "_in_wiki"
        person_path = "/data/fangzheng/bert_el/persons.txt"
        candidate_path = source_dir + data_name + "/source/" + data_name + "_golden_redirect"

        # self.read_xml_dir_data(xml_path, in_wiki_path)
        # self.pre_msnbc_data(raw_dir, in_wiki_path)
        # self.filter_not_inkb_data(in_wiki_path, in_wiki_path)
        # self.get_father_mention(in_wiki_path, person_path, False)
        # self.get_adjacent_mention(in_wiki_path)
        # self.get_mention_synonym(in_wiki_path)
        # self.get_context_keyword(in_wiki_path)

        # crawl redirect page
        # crawl candidate
        # crawl wiki data

        pass

if __name__ == "__main__":
    process_ace = ProcessACE()

    process_ace.control_ace()


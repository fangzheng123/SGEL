# encoding: utf-8

import json
from base_process import BaseProcess
from urllib import parse

class PreOther(BaseProcess):


    def read_candidate(self, other_format_path, our_format_path, candidate_path):
        """

        :param data_path:
        :return:
        """
        our_mention_list = []
        with open(our_format_path, "r", encoding="utf-8") as our_foramt_file:
            for item in our_foramt_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_obj["doc_context"] = ""
                mention_obj["candidate"] = ""
                our_mention_list.append(mention_obj)

        not_match = 0
        with open(other_format_path, "r", encoding="utf-8") as other_format_file:
            with open(candidate_path, "w", encoding="utf-8") as candidate_file:
                for item in other_format_file:
                    item = item.strip()
                    tmp_list = item.split("\t")
                    mention_file = tmp_list[0]
                    mention_form = tmp_list[2]
                    target_name = tmp_list[-1].split(",")[-1]

                    candidate_index = [index for index, ele in enumerate(tmp_list) if ele == "CANDIDATES"][0]
                    candidate_list = tmp_list[candidate_index + 1: -2]
                    candidate_list = [ele.split(",")[-1].strip().replace(" ", "_") for ele in candidate_list]

                    mention_obj = {}
                    target_name = target_name.strip().replace(" ", "_")
                    mention_obj["target_url"] = "http://en.wikipedia.org/wiki/" + target_name
                    mention_obj["mention_form"] = mention_form
                    mention_obj["mention_file"] = mention_file
                    mention_obj["candidate"] = {}
                    mention_obj["candidate"]["main"] = candidate_list

                    match_flag = False

                    for our_mention in our_mention_list:
                        if "flag" in our_mention:
                            continue

                        if parse.unquote(mention_form.lower()) == parse.unquote(our_mention["mention_form"].lower()) \
                            and parse.unquote(target_name) == parse.unquote(our_mention["target_url"].split("/")[-1]):
                            mention_obj["mention_file"] = our_mention["mention_file"]
                            mention_obj["mention_offset"] = our_mention["mention_offset"]
                            mention_obj["mention_context"] = our_mention["mention_context"]
                            our_mention["flag"] = True
                            match_flag = True

                            break

                    if not match_flag:
                        mention_obj["mention_context"] = ""
                        not_match += 1

                    print("not match num: {0}".format(not_match))
                    candidate_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    pre_other = PreOther()


    # other_data_list = ["aida_train"]

    other_data_list = ["wned-msnbc", "wned-ace2004", "wned-aquaint", "aida_testA", "aida_testB", "aida_train"]

    for other_data_name in other_data_list:
        print(other_data_name)

        other_format_path = "/home1/fangzheng/data/bert_el_data/other_people_data/" + other_data_name + ".csv"

        data_name = other_data_name.replace("wned-", "")
        other_candidate_path = "/home1/fangzheng/data/bert_el_data/other_people_data/format/" + data_name
        other_candidate_format_path = "/home1/fangzheng/data/bert_el_data/other_people_data/format/" + data_name + "_candidate_format"

        our_format_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/source/" + data_name + "_pre_in_wiki"
        wiki_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_wiki"
        redirect_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_candidate_redirect"

        pre_other.read_candidate(other_format_path, our_format_path, other_candidate_path)

        pre_other.format_candidate(other_candidate_path, wiki_path, redirect_path, other_candidate_format_path)





# encoding: utf-8

import json

class GenerateGerbilResult(object):

    def __init__(self):
        pass

    def generate_aida_gerbil_golden(self, own_result_path, gerbil_result_path):
        """
        generate own result to gerbil format
        :param own_result_path:
        :param gerbil_result_path:
        :return:
        """
        mention_list = []
        with open(own_result_path, "r", encoding="utf-8") as own_result_file:
            for item in own_result_file:
                item = item.strip()

                mention_obj = json.loads(item)

                gerbil_mention_obj = {}
                gerbil_mention_obj["mentionFile"] = mention_obj["mention_file"]
                gerbil_mention_obj["mentionName"] = mention_obj["mention_form"]
                gerbil_mention_obj["predictUrl"] = mention_obj["target_url"].replace("https", "http")
                # if mention_obj["our_predict"]["label"] == 1:
                #     gerbil_mention_obj["predictUrl"] = mention_obj["our_predict"]["url"].replace("https", "http")
                # elif mention_obj["target_redirect_url"].__contains__("disambiguation"):
                #     gerbil_mention_obj["predictUrl"] = mention_obj["target_url"].replace("https", "http")
                # else:
                #     gerbil_mention_obj["predictUrl"] = "http://aksw.org/notInWiki/ABC"
                gerbil_mention_obj["mentionOffset"] = mention_obj["mention_offset"]
                gerbil_mention_obj["mentionLength"] = len(mention_obj["mention_form"])
                mention_list.append(gerbil_mention_obj)

        with open(gerbil_result_path, "w", encoding="utf-8") as gerbil_result_file:
            for mention_obj in mention_list:
                gerbil_result_file.write(json.dumps(mention_obj) + "\n")


if __name__ == "__main__":
    generate_gerbil = GenerateGerbilResult()

    data_name = "aida_testA"
    source_dir = "/data/fangzheng/bert_el/"
    final_result_v1_path = source_dir + data_name + "/bert/" + data_name + "_final_result_v1"
    gerbil_result_path = source_dir + data_name + "/bert/" + data_name + "_final_result_v1_gerbil"

    generate_gerbil.generate_aida_gerbil_golden(final_result_v1_path, gerbil_result_path)
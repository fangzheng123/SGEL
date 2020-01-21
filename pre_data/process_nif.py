# encoding:utf-8

from base_process import BaseProcess
from candidate_analyse import CandidateAnalyse
import json

class ProcessNIF(BaseProcess):

    def process_target(self, pre_path):
        """

        :param golden_path:
        :return:
        """
        golden_data_list = []
        with open(pre_path, "r") as pre_file:
            for item in pre_file:
                item = item.strip()

                mention_obj = json.loads(item)

                if "wikipedia_url" in mention_obj:
                    mention_obj["target_url"] = mention_obj["wikipedia_url"]
                    golden_data_list.append(mention_obj)

                mention_obj["mention_offset"] = mention_obj["offset"]

        with open(pre_path, "w") as pre_file:
            for mention_obj in golden_data_list:
                pre_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")

        print("golden num: {0}".format(len(golden_data_list)))

    # def get_kore50_keyword(self, pre_in_wiki_path):
    #     """
    #     search other mention as keyword
    #     :param pre_in_wiki_path:
    #     :return:
    #     """
    #     group_mention_dict = {}
    #     with open(pre_in_wiki_path, "r", encoding="utf-8") as pre_in_wiki_file:
    #         for item in pre_in_wiki_file:
    #             item = item.strip()
    #
    #             mention_obj = json.loads(item)
    #
    #             mention_file = mention_obj["mention_file"]
    #
    #             if mention_file not in group_mention_dict:
    #                 group_mention_dict[mention_file] = [mention_obj]
    #             else:
    #                 group_mention_dict[mention_file].append(mention_obj)
    #
    #     data_list = []
    #     with open(pre_in_wiki_path, "r", encoding="utf-8") as pre_in_wiki_file:
    #         for item in pre_in_wiki_file:
    #             item = item.strip()
    #
    #             mention_obj = json.loads(item)
    #
    #             mention_file = mention_obj["mention_file"]
    #             mention_offset = mention_obj["offset"]
    #
    #             other_mention_list = group_mention_dict[mention_file]
    #
    #             other_mention_dict = {}
    #             for other_mention_obj in other_mention_list:
    #                 if other_mention_obj["offset"] != mention_offset:
    #                     other_mention_offset = other_mention_obj["offset"]
    #                     other_mention_dict[other_mention_offset] = other_mention_obj
    #
    #             # search min distance mention
    #             min_dis = 1000
    #             min_offset = 10000
    #             for other_offset, other_mention in other_mention_dict.items():
    #                 if abs(other_offset - mention_offset) < min_dis:
    #                     min_dis = abs(other_offset - mention_offset)
    #                     min_offset = other_offset
    #
    #             context_keyword = ""
    #             if min_offset != 10000:
    #                 other_min_mention = other_mention_dict[min_offset]
    #                 context_keyword = other_min_mention["mention_form"]
    #
    #             mention_obj["context_keyword"] = context_keyword
    #
    #             data_list.append(mention_obj)
    #
    #     with open(pre_in_wiki_path, "w", encoding="utf-8") as pre_in_wiki_file:
    #         for data in data_list:
    #             pre_in_wiki_file.write(json.dumps(data, ensure_ascii=False) + "\n")


    def control_nif(self):
        """

        :return:
        """
        data_name = "reuters128"

        source_dir = "/data/fangzheng/bert_el/"

        in_wiki_path = source_dir + data_name + "/source/" + data_name + "_in_wiki"
        person_path = "/data/fangzheng/bert_el/persons.txt"
        candidate_path = source_dir + data_name + "/source/" + data_name + "_golden_redirect"

        # self.process_target(in_wiki_path)
        # self.filter_not_inkb_data(in_wiki_path, in_wiki_path)
        # self.get_father_mention(candidate_path, person_path, False)
        # self.get_adjacent_mention(in_wiki_path)
        # self.get_mention_synonym(in_wiki_path)
        # self.get_context_keyword(in_wiki_path)

        # crawl redirect page
        # crawl candidate
        # crawl wiki data

        pass


if __name__ == "__main__":
    process_nif = ProcessNIF()

    process_nif.control_nif()




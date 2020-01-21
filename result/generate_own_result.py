# encoding:utf-8


import json
from urllib import parse


class GenerateOwnResult(object):


    def __init__(self):
        pass
        # self.pre_util = pre_util.DataUtil()

    def combine_model_result(self, combine_model_pred_path, xgboost_pred_path, all_model_pred_path):
        """
        combine deep model result and xgboost result
        :param combine_model_pred_path:
        :param xgboost_pred_path:
        :param all_model_pred_path:
        :return:
        """
        model_group_candidate_dict = {}
        with open(combine_model_pred_path, "r", encoding="utf-8") as combine_model_pred_file:
            for item in combine_model_pred_file:
                item = item.strip()

                group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_id_str)
                mention_obj = json.loads(mention_str)
                pred_candidate = json.loads(entity_str)
                fea_obj = json.loads(fea_str)
                label = int(label_str)

                pred_candidate["feature"] = fea_obj
                pred_candidate["label"] = label
                pred_candidate["vec"] = ""

                mention_obj["our_predict"] = pred_candidate
                mention_obj["vec"] = ""
                mention_obj["candidate"] = ""

                model_group_candidate_dict[group_id] = mention_obj

        print("model predict mention num:{0}".format(len(model_group_candidate_dict)))

        xgboost_group_candidate_dict = {}
        with open(xgboost_pred_path, "r", encoding="utf-8") as xgboost_pred_file:
            for item in xgboost_pred_file:
                item = item.strip()
                group_str, label_str, fea_str, mention_str, candidate_str = item.split("\t")

                group_id = int(group_str)
                label = int(label_str)
                fea = json.loads(fea_str)
                mention_obj = json.loads(mention_str)
                candidate_obj = json.loads(candidate_str)

                candidate_obj["feature"] = fea
                candidate_obj["label"] = label

                mention_obj["our_predict"] = candidate_obj

                if "offset" in mention_obj:
                    mention_obj["mention_offset"] = mention_obj["offset"]

                if group_id not in xgboost_group_candidate_dict:
                    xgboost_group_candidate_dict[group_id] = mention_obj

        print("xgboost predict mention num:{0}".format(len(xgboost_group_candidate_dict)))

        combine_result_dict = {}
        for group_id, mention_obj in xgboost_group_candidate_dict.items():
            # data_list = ["aquaint", "rss500"]
            if group_id in model_group_candidate_dict:
                combine_result_dict[group_id] = model_group_candidate_dict[group_id]
            else:
                combine_result_dict[group_id] = mention_obj
        print("final predict mention num:{0}".format(len(combine_result_dict)))

        right_count = 0
        with open(all_model_pred_path, "w", encoding="utf-8") as all_model_pred_file:
            for group_id, mention_obj in combine_result_dict.items():
                predict_candidate = mention_obj["our_predict"]
                if predict_candidate["label"] == 1:
                    right_count += 1

                all_model_pred_file.write(str(group_id) + "\t" + json.dumps(mention_obj) + "\n")

        print("our predict group acc:{0}".format(right_count / len(combine_result_dict)))

    def read_doc_mention_obj(self, data_path):
        """
        read mentions in the same document
        :param data_path:
        :return:
        """
        doc_mention_dict = {}
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                item = item.split("\t")[-1]
                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]

                if mention_file not in doc_mention_dict:
                    doc_mention_dict[mention_file] = [mention_obj]
                else:
                    doc_mention_dict[mention_file].append(mention_obj)

        return doc_mention_dict

    def search_father_result(self, all_model_pred_path, final_result_v1_path):
        """
        deal with mention which has father mention in the same document, generate final result v1
        :param all_model_pred_path:
        :param final_result_v1_path:
        :return:
        """
        doc_mention_dict = self.read_doc_mention_obj(all_model_pred_path)

        all_mention_list = []
        with open(all_model_pred_path, "r", encoding="utf-8") as all_model_pred_file:
            for item in all_model_pred_file:
                item = item.strip()

                group_id, mention_str = item.split("\t")

                mention_obj = json.loads(mention_str)
                mention_file = mention_obj["mention_file"]

                father_mention = mention_obj["father_mention"]
                if len(father_mention) > 0:
                    doc_mention_list = doc_mention_dict[mention_file]
                    doc_form_list = [mention_obj["mention_form"] for mention_obj in doc_mention_list]

                    for other_mention_form, other_mention_obj in zip(doc_form_list, doc_mention_list):
                        if other_mention_form == father_mention[0]:
                            mention_obj["our_predict"] = other_mention_obj["our_predict"]

                all_mention_list.append(mention_obj)

        with open(final_result_v1_path, "w", encoding="utf-8") as final_result_v1_file:
            for mention_obj in all_mention_list:
                final_result_v1_file.write(json.dumps(mention_obj) + "\n")

    def cal_recall(self, final_result_path):
        """

        :param final_result_path:
        :return:
        """
        recall_count = 0
        all_count = 0
        with open(final_result_path, "r", encoding="utf-8") as final_result_file:
            for item in final_result_file:
                item = item.strip()

                mention_obj = json.loads(item)
                if mention_obj["our_predict"]["label"] == 1:
                    recall_count += 1

                all_count += 1

        print(recall_count, all_count, recall_count/all_count)

if __name__ == "__main__":
    generate_result = GenerateOwnResult()

    data_name = "reuters128"
    source_dir = "/data/fangzheng/bert_el/"
    combine_model_result_path = source_dir + data_name + "/bert/" + data_name + "_cut_rank_format_bert_predict"
    xgboost_pred_path = source_dir + data_name + "/candidate/" + data_name + "_cut_rank_format"
    all_model_pred_path = source_dir + data_name + "/bert/" + data_name + "_all_pred"

    golden_path = source_dir + data_name + "/candidate/" + data_name + "_candidate"
    entity_path = source_dir + data_name + "/candidate/" + data_name + "_wiki"

    final_result_v1_path = source_dir + data_name + "/bert/" + data_name + "_final_result_v1"

    print(data_name)
    generate_result.combine_model_result(combine_model_result_path, xgboost_pred_path, all_model_pred_path)
    if data_name != "msnbc":
        generate_result.search_father_result(all_model_pred_path, final_result_v1_path)
    # generate_result.search_keyword_candidate(final_result_v1_path, final_result_v2_path, entity_path)

    generate_result.cal_recall(final_result_v1_path)
    # generate_result.cal_recall(final_result_v2_path)
# encoding: utf-8

import json
from urllib import parse
from data_util import DataUtil
import Levenshtein

class CandidateAnalyse(object):

    def __init__(self):
        self.data_util = DataUtil()

    def cal_candidate_recall(self, candidate_path):
        """

        :param candidate_path:
        :return:
        """
        recall_count = 0
        valid_recall_count = 0
        valid_count = 0
        all_count = 0
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
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
                    all_candidate_list.extend(candidate_list)

                parse_candidate_list = [parse.unquote(candidate) for candidate in all_candidate_list]
                if parse.unquote(target_name) in parse_candidate_list or target_name.__contains__("disambiguation") \
                        or target_name == "":
                    recall_count += 1
                else:
                    print(mention_form, target_name, mention_candidate_dict)

                if (not target_name.__contains__("disambiguation")) and target_name != "":
                    valid_count += 1

                    if parse.unquote(target_name) in parse_candidate_list:
                        valid_recall_count += 1

                all_count += 1

        print("all count:{0}, valid_count:{1}, valid_recall_count:{2}, "
              "recall count:{3}, recall:{4}, valid_recall:{5}".format(all_count, valid_count, valid_recall_count,
                                                                      recall_count, recall_count / all_count,
                                                                      valid_recall_count / valid_count))

    def cal_candidate_format_recall(self, candidate_format_path):
        """
        calculate candidate format data recall
        :param candidate_format_path:
        :return:
        """
        recall_count = 0
        valid_count = 0
        group_id_dict = {}
        with open(candidate_format_path, "r", encoding="utf-8") as candidate_format_file:
            for item in candidate_format_file:
                item = item.strip()
                group_str, label_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_str)
                label = int(label_str)
                mention_obj = json.loads(mention_str)
                redirect_target_url = mention_obj["target_redirect_url"]

                if group_id in group_id_dict:
                    group_id_dict[group_id].add(label)
                else:
                    label_set = set()
                    label_set.add(label)
                    group_id_dict[group_id] = label_set
                    if (not redirect_target_url.__contains__("disambiguation")) and redirect_target_url != "":
                        valid_count += 1

            for group_id, label_set in group_id_dict.items():
                if 1 in label_set:
                    recall_count += 1

        print("recall count: {0}, valid count:{1}, recall: {2}".format(
            recall_count, valid_count, recall_count/valid_count))

    def cal_part_recall(self, candidate_path):
        """
        calculate the gold recall in different part
        :param candidate_path:
        :return:
        """
        part1 = 0
        part2 = 0
        part3 = 0
        all_count = 0
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                redirect_target_url = mention_obj["target_redirect_url"]
                target_name = redirect_target_url.split("/")[-1]
                if target_name.__contains__("#"):
                    target_name = target_name.split("#")[0]

                target_name = parse.unquote(target_name)

                candidate_dict = mention_obj["candidate"]

                part1_flag = True
                part2_flag = True
                part3_flag = True
                for type, candidate_list in candidate_dict.items():
                    parse_candidate_list = [parse.unquote(candidate) for candidate in candidate_list[:30]]
                    if type in ["mention_page", "mention_disam_page", "mention_partial_matching_page", "mention_search_page"]:
                        if part1_flag and target_name in set(parse_candidate_list):
                            part1 += 1
                            part1_flag = False
                            # break

                    elif type in ["mention_keyword_search", "doc_candidate"]:
                        if part2_flag and target_name in set(parse_candidate_list):
                            part2 += 1
                            part2_flag = False
                            # break

                    elif type in ["google_search_result"]:
                        if part3_flag and target_name in set(parse_candidate_list):
                            part3 += 1
                            part3_flag = False
                            # break

                all_count += 1

        print("part1 recall: {0}, {1}\n part2 recall: {2}, {3}\n part3 recall: {4}, {5}".format(
            part1, part1/all_count, part2, part2/all_count, part3, part3/all_count))



    def own_candidate_analyse(self, rank_path):
        """

        :param rank_path:
        :return:
        """
        group_dict = {}
        with open(rank_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_str)
                fea_obj = json.loads(fea_str)
                fea_obj["label"] = int(label_str)

                if group_id not in group_dict:
                    group_dict[group_id] = [fea_obj]
                else:
                    group_dict[group_id].append(fea_obj)

        candidate_num_list = []
        mean_name_jaro_list = []
        mean_sem_sim_list = []
        mean_pv_list = []
        is_name_top1_list = []
        is_name_top3_list = []
        is_pv_top1_list = []
        is_pv_top3_list = []
        is_name_pv_top1_list = []
        is_position_top1_list = []
        is_position_top3_list = []

        for group_id, fea_list in group_dict.items():
            candidate_num_list.append(len(fea_list))

            tmp_name_jaro_dict = {}
            tmp_sem_sim_dict = {}
            tmp_pv_dict = {}
            tmp_position_dict = {}
            for index, fea_obj in enumerate(fea_list):
                if "name_jaro" in fea_obj:
                    tmp_name_jaro_dict[index] = fea_obj["name_jaro"]
                if "context_summary_word_cos" in fea_obj and fea_obj["context_summary_word_cos"] != 0:
                    tmp_sem_sim_dict[index] = fea_obj["context_summary_word_cos"]
                if "pageview" in fea_obj:
                    tmp_pv_dict[index] = fea_obj["pageview"]
                if "candidate_recall_rank" in fea_obj:
                    tmp_position_dict[index] = fea_obj["candidate_recall_rank"]

            # mean name jaro dis
            if len(tmp_name_jaro_dict) != 0:
                name_val_list = [val for key, val in tmp_name_jaro_dict.items()]
                mean_name_jaro_list.append(sum(name_val_list) / len(name_val_list))

            # mean semantic sim
            if len(tmp_sem_sim_dict) != 0:
                sem_sim_list = [val for key, val in tmp_sem_sim_dict.items()]
                mean_sem_sim_list.append(sum(sem_sim_list) / len(sem_sim_list))

            # mean pv
            if len(tmp_pv_dict) != 0:
                pv_val_list = [val for key, val in tmp_pv_dict.items()]
                mean_pv_list.append(sum(pv_val_list) / len(pv_val_list))

            # rank name jaro dis
            tmp_rank_name_list = sorted(tmp_name_jaro_dict.items(), key=lambda x:x[1], reverse=True)
            name_top1_flag = (fea_list[tmp_rank_name_list[0][0]]["label"] == 1)
            name_top3_flag_list = [fea_list[fea_index]["label"] == 1 for fea_index, jaro_val in tmp_rank_name_list[:3]]

            # name top1
            if name_top1_flag:
                is_name_top1_list.append(1)
            else:
                is_name_top1_list.append(0)

            # name top3
            if True in set(name_top3_flag_list):
                is_name_top3_list.append(1)
            else:
                is_name_top3_list.append(0)

            # rank pv
            tmp_rank_pv_list = sorted(tmp_pv_dict.items(), key=lambda x:x[1], reverse=True)
            pv_top1_flag = (fea_list[tmp_rank_pv_list[0][0]]["label"] == 1)
            pv_top3_flag_list = [fea_list[fea_index]["label"] == 1 for fea_index, pv in tmp_rank_pv_list[:3]]

            # pv top1
            if pv_top1_flag:
                is_pv_top1_list.append(1)
            else:
                is_pv_top1_list.append(0)

            # pv top3
            if True in set(pv_top3_flag_list):
                is_pv_top3_list.append(1)
            else:
                is_pv_top3_list.append(0)

            # rank position
            tmp_rank_postion_list = sorted(tmp_position_dict.items(), key=lambda x: x[1])
            postion_top1_flag = (fea_list[tmp_rank_postion_list[0][0]]["label"] == 1)
            postion_top3_flag_list = [fea_list[fea_index]["label"] == 1 for fea_index, postion in tmp_rank_postion_list[:3]]

            # position top1
            if postion_top1_flag:
                is_position_top1_list.append(1)
            else:
                is_position_top1_list.append(0)

            # position top3
            if True in set(postion_top3_flag_list):
                is_position_top3_list.append(1)
            else:
                is_position_top3_list.append(0)

            # name top1 + pv top1
            if name_top1_flag or pv_top1_flag:
                is_name_pv_top1_list.append(1)
            else:
                is_name_pv_top1_list.append(0)



        # 1. mean candidate num
        mean_candidate_num = 0.0
        if len(candidate_num_list) != 0:
            mean_candidate_num = sum(candidate_num_list) / len(candidate_num_list)

        # 2. mean name jaro
        mean_name_jaro = 0.0
        if len(mean_name_jaro_list) != 0:
            mean_name_jaro = sum(mean_name_jaro_list) / len(mean_name_jaro_list)

        mean_sem_sim = 0.0
        if len(mean_sem_sim_list) != 0:
            mean_sem_sim = sum(mean_sem_sim_list) / len(mean_sem_sim_list)

        # 3. name@1
        name_top1_acc = 0.0
        if len(is_name_top1_list) != 0:
            name_top1_acc = sum(is_name_top1_list) / len(is_name_top1_list)

        # 4. name@3
        name_top3_acc = 0.0
        if len(is_name_top3_list) != 0:
            name_top3_acc = sum(is_name_top3_list) / len(is_name_top3_list)

        # 5. pv@1
        pv_top1_acc = 0.0
        if len(is_pv_top1_list) != 0:
            pv_top1_acc = sum(is_pv_top1_list) / len(is_pv_top1_list)

        # 6. pv@3
        pv_top3_acc = 0.0
        if len(is_pv_top3_list) != 0:
            pv_top3_acc = sum(is_pv_top3_list) / len(is_pv_top3_list)

        # 7. name@1 + pv@
        name_pv_top1_acc = 0.0
        if len(is_name_pv_top1_list) != 0:
            name_pv_top1_acc = sum(is_name_pv_top1_list) / len(is_name_pv_top1_list)

        # 8. num_less5
        num_less5 = len([num for num in candidate_num_list if num < 5])

        # 9. postion@1
        postion_top1_acc = 0.0
        if len(is_position_top1_list) != 0:
            postion_top1_acc = sum(is_position_top1_list) / len(is_position_top1_list)

        # 10. postion@3
        postion_top3_acc = 0.0
        if len(is_pv_top3_list) != 0:
            postion_top3_acc = sum(is_position_top3_list) / len(is_position_top3_list)

        # 10. mean pv
        mean_pv = 0.0
        if len(mean_pv_list) != 0:
            mean_pv = sum(mean_pv_list) / len(mean_pv_list)

        print("\n".join(["mean_candidate_num: {0}", "mean_name_jaro: {1}", "name_top1_acc: {2}",
                         "name_top3_acc: {3}", "pv_top1_acc: {4}", "pv_top3_acc: {5}",
                         "name_pv_top1_acc: {6}", "num_less5: {7}", "postion_top1_acc: {8}",
                         "postion_top3_acc: {9}", "mean_pv: {10}"]).format(mean_candidate_num, mean_name_jaro,
                                                          name_top1_acc, name_top3_acc, pv_top1_acc,
                                                          pv_top3_acc, name_pv_top1_acc, num_less5, postion_top1_acc, postion_top3_acc, mean_pv) + "\n")

        print("mean_sem_sim: {0}".format(mean_sem_sim))

    def read_other_candidate(self, other_format_path, other_candidate_path):
        """
        read candidate name in other people's data
        :param other_format_path:
        :param other_candidate_path:
        :return:
        """
        name_set = set()
        with open(other_candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()
                name_set.add(item)

        candidate_name_list = []
        with open(other_format_path, "r", encoding="utf-8") as other_format_file:
            for item in other_format_file:
                item = item.strip()
                tmp_list = item.split("\t")
                candidate_index = [index for index, ele in enumerate(tmp_list) if ele == "CANDIDATES"][0]

                candidate_list = tmp_list[candidate_index+1: -2]
                candidate_list = [ele.split(",")[-1] for ele in candidate_list]
                candidate_name_list.extend(candidate_list)

        with open(other_candidate_path, "a+", encoding="utf-8") as candidate_file:
            for name in candidate_name_list:
                name = name.strip()
                name = name.replace(" ", "_")
                if name not in name_set:
                    name_set.add(name)
                    candidate_file.write(name + "\n")

    def get_other_candidate_pv(self, other_candidate_path, all_entity_path, redirect_path):
        """

        :param other_candidate_path:
        :param all_entity_path:
        :param redirect_path:
        :return:
        """
        # get all entity info
        all_entity_dict = self.data_util.read_all_entity(all_entity_path, redirect_path)

        name_pv_dict = {}
        all_name_list = []
        with open(other_candidate_path, "r", encoding="utf-8") as other_candidate_file:
            for item in other_candidate_file:
                name = item.strip()
                all_name_list.append(name)

                if name in all_entity_dict:
                    entity_obj = all_entity_dict[name]
                    if "popularity" in entity_obj and "views_sum" in entity_obj["popularity"]:
                        candidate_pv = entity_obj["popularity"]["views_sum"]
                        candidate_pv = candidate_pv / 1000000.0
                        name_pv_dict[name] = candidate_pv

        has_pv_num = 0
        with open(other_candidate_path, "w", encoding="utf-8") as other_candidate_file:
            for name in all_name_list:
                if name in name_pv_dict:
                    other_candidate_file.write(name + "\t" + str(name_pv_dict[name]) + "\n")
                    has_pv_num += 1
                else:
                    other_candidate_file.write(name + "\n")

        print("has_pv_num: {0}, all name num: {1}".format(has_pv_num, len(all_name_list)))

    def combine_other_candidate_pv(self, other_candidate_path, other_candidate_crawl_path):
        """

        :param other_candidate_path:
        :param other_candidate_crawl_path:
        :return:
        """
        crawl_pv_dict = {}
        with open(other_candidate_crawl_path, "r", encoding="utf-8") as candidate_crawl_file:
            for item in candidate_crawl_file:
                item = item.strip()
                name, pv_str = item.split("\t")
                crawl_pv_dict[name] = float(pv_str)

        all_name_dict = {}
        with open(other_candidate_path, "r", encoding="utf-8") as other_candidate_file:
            for item in other_candidate_file:
                item = item.strip()

                if len(item.split("\t")) == 2:
                    name, pv_str = item.split("\t")
                    all_name_dict[name] = float(pv_str)
                else:
                    if item in crawl_pv_dict:
                        all_name_dict[item] = crawl_pv_dict[item]
                    else:
                        all_name_dict[item] = 0

        with open(other_candidate_path, "w", encoding="utf-8") as other_candidate_file:
            for name, pv in all_name_dict.items():
                other_candidate_file.write(name + "\t" + str(pv) + "\n")

    def other_candidate_analyse(self, other_format_path, other_candidate_path):
        """

        :param other_format_path:
        :param other_candidate_path:
        :return:
        """
        name_pv_dict = {}
        with open(other_candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()
                name, pv_str = item.split("\t")
                name_pv_dict[name] = float(pv_str)

        group_candidate_dict = {}
        with open(other_format_path, "r", encoding="utf-8") as other_format_file:
            for index, item in enumerate(other_format_file):
                item = item.strip()
                tmp_list = item.split("\t")

                mention = tmp_list[2]
                mention = mention.replace(" ", "_")

                target_name = tmp_list[-1].split(",")[-1].replace(" ", "_")

                candidate_index = [index for index, ele in enumerate(tmp_list) if ele == "CANDIDATES"][0]
                candidate_list = tmp_list[candidate_index+1: -2]
                candidate_list = [ele.split(",")[-1] for ele in candidate_list]

                entity_list = []
                for candidate_name in candidate_list:
                    candidate_name = candidate_name.replace(" ", "_")

                    entity_obj = {}
                    entity_obj["name"] = candidate_name
                    entity_obj["name_jaro"] = Levenshtein.jaro(candidate_name.lower(), mention.lower())

                    if candidate_name == target_name:
                        entity_obj["label"] = 1
                    else:
                        entity_obj["label"] = 0

                    if candidate_name in name_pv_dict:
                        entity_obj["pv"] = name_pv_dict[candidate_name]
                    else:
                        entity_obj["pv"] = 0


                    entity_list.append(entity_obj)

                group_candidate_dict[mention+str(index)] = entity_list

        candidate_num_list = []
        mean_name_jaro_list = []
        mean_pv_list = []
        is_name_top1_list = []
        is_name_top3_list = []
        is_pv_top1_list = []
        is_pv_top3_list = []
        is_name_pv_top1_list = []

        print(len(group_candidate_dict))

        for mention, entity_list in group_candidate_dict.items():
            candidate_num_list.append(len(entity_list))
            if len(entity_list) == 1:
                print(mention, entity_list)

            name_jaro_list = [obj["name_jaro"] for obj in entity_list]
            if len(name_jaro_list) != 0:
                mean_name_jaro_list.append(sum(name_jaro_list) / len(name_jaro_list))

            pv_list = [obj["pv"] for obj in entity_list]
            if len(pv_list) != 0:
                mean_pv_list.append(sum(pv_list) / len(pv_list))

            # rank name jaro dis
            tmp_name_jaro_dict = {}
            for index, obj in enumerate(entity_list):
                tmp_name_jaro_dict[index] = obj["name_jaro"]

            tmp_rank_name_list = sorted(tmp_name_jaro_dict.items(), key=lambda x: x[1], reverse=True)
            name_top1_flag = (entity_list[tmp_rank_name_list[0][0]]["label"] == 1)
            name_top3_flag_list = [entity_list[fea_index]["label"] == 1 for fea_index, jaro_val in
                                       tmp_rank_name_list[:3]]

            # name top1
            if name_top1_flag:
                is_name_top1_list.append(1)
            else:
                is_name_top1_list.append(0)

            # name top3
            if True in set(name_top3_flag_list):
                is_name_top3_list.append(1)
            else:
                is_name_top3_list.append(0)

            # rank pv
            tmp_pv_dict = {}
            for index, obj in enumerate(entity_list):
                tmp_pv_dict[index] = obj["pv"]

            tmp_rank_pv_list = sorted(tmp_pv_dict.items(), key=lambda x: x[1], reverse=True)
            pv_top1_flag = (entity_list[tmp_rank_pv_list[0][0]]["label"] == 1)
            pv_top3_flag_list = [entity_list[fea_index]["label"] == 1 for fea_index, pv in tmp_rank_pv_list[:3]]

            # pv top1
            if pv_top1_flag:
                is_pv_top1_list.append(1)
            else:
                is_pv_top1_list.append(0)

            # pv top3
            if True in set(pv_top3_flag_list):
                is_pv_top3_list.append(1)
            else:
                is_pv_top3_list.append(0)

            # name top1 + pv top1
            if name_top1_flag or pv_top1_flag:
                is_name_pv_top1_list.append(1)
            else:
                is_name_pv_top1_list.append(0)

        # 1. mean candidate num
        mean_candidate_num = 0.0
        if len(candidate_num_list) != 0:
            mean_candidate_num = sum(candidate_num_list) / len(candidate_num_list)

        # 2. mean name jaro
        mean_name_jaro = 0.0
        if len(mean_name_jaro_list) != 0:
            mean_name_jaro = sum(mean_name_jaro_list) / len(mean_name_jaro_list)

        # 3. name@1
        name_top1_acc = 0.0
        if len(is_name_top1_list) != 0:
            name_top1_acc = sum(is_name_top1_list) / len(is_name_top1_list)

        # 4. name@3
        name_top3_acc = 0.0
        if len(is_name_top3_list) != 0:
            name_top3_acc = sum(is_name_top3_list) / len(is_name_top3_list)

        # 5. pv@1
        pv_top1_acc = 0.0
        if len(is_pv_top1_list) != 0:
            pv_top1_acc = sum(is_pv_top1_list) / len(is_pv_top1_list)

        # 6. pv@3
        pv_top3_acc = 0.0
        if len(is_pv_top3_list) != 0:
            pv_top3_acc = sum(is_pv_top3_list) / len(is_pv_top3_list)

        # 7. name@1 + pv@
        name_pv_top1_acc = 0.0
        if len(is_name_pv_top1_list) != 0:
            name_pv_top1_acc = sum(is_name_pv_top1_list) / len(is_name_pv_top1_list)

        # 8. num_less5
        num_less5 = len([num for num in candidate_num_list if num < 5])

        # 9. mean pv
        mean_pv = 0.0
        if len(mean_pv_list) != 0:
            mean_pv = sum(mean_pv_list) / len(mean_pv_list)

        print("\n".join(["mean_candidate_num: {0}", "mean_name_jaro: {1}", "name_top1_acc: {2}",
                         "name_top3_acc: {3}", "pv_top1_acc: {4}", "pv_top3_acc: {5}",
                         "name_pv_top1_acc: {6}", "num_less5: {7}", "mean_pv: {8}"]).format(mean_candidate_num, mean_name_jaro,
                                                          name_top1_acc, name_top3_acc, pv_top1_acc,
                                                          pv_top3_acc, name_pv_top1_acc, num_less5/(len(candidate_num_list)), mean_pv) + "\n")

    def control_other_pv(self):
        """

        :return:
        """
        data_list = ["wned-msnbc.csv", "wned-ace2004.csv", "wned-aquaint.csv", "aida_testA.csv", "aida_testB.csv",
                     "aida_train.csv"]
        candidate_path = "/home1/fangzheng/data/bert_el_data/other_people_data/candidate"
        for other_data_name in data_list:
            other_format_path = "/home1/fangzheng/data/bert_el_data/other_people_data/" + other_data_name
            candidate_analyse.read_other_candidate(other_format_path, candidate_path)

        candidate_analyse.get_other_candidate_pv(candidate_path,
                                                 "/home1/fangzheng/data/bert_el_data/source_data/all_source_data",
                                                 "/home1/fangzheng/data/bert_el_data/candidate/redirect_dict")

        candidate_analyse.combine_other_candidate_pv(candidate_path, "/home1/fangzheng/data/bert_el_data/other_people_data/candidate_crawl_pv")

if __name__ == "__main__":
    candidate_analyse = CandidateAnalyse()

    data_list = ["msnbc", "ace2004", "aquaint", "aida_testA", "aida_testB", "aida_train", "rss500", "reuters128", "kore50"]
    for data_name in data_list:
        print(data_name)
        rank_format_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_rank_format"
        candidate_analyse.own_candidate_analyse(rank_format_path)

        # candidate_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_candidate"
        # candidate_analyse.cal_part_recall(candidate_path)
        # print("\n\n")

    # other_data_list = ["aida_testB.csv"]
    # other_data_list = ["wned-msnbc.csv", "wned-ace2004.csv", "wned-aquaint.csv", "aida_testA.csv", "aida_testB.csv",
    #              "aida_train.csv"]
    # other_candidate_path = "/home1/fangzheng/data/bert_el_data/other_people_data/candidate"
    # for other_data_name in other_data_list:
    #     print(other_data_name)
    #     other_format_path = "/home1/fangzheng/data/bert_el_data/other_people_data/" + other_data_name
    #     candidate_analyse.other_candidate_analyse(other_format_path, other_candidate_path)


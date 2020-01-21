# encoding: utf-8

import json

class Test(object):

    def build_sent(self, cut_candidate_path, bert_path):
        """
        build sentence for bert, the aim is to get sentence embedding
        :param cut_candidate_path:
        :param bert_path:
        :return:
        """
        with open(cut_candidate_path, "r", encoding="utf-8") as cut_candidate_file:
            mention_sent_list = []
            entity_sent_list = []

            group_id_set = set()
            for item in cut_candidate_file:
                item = item.strip()
                group_id, label, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_id)

                mention_obj = json.loads(mention_str)
                entity_obj = json.loads(entity_str)

                mention_context, entity_des = self.data_util.get_context_desc(mention_obj, entity_obj)

                if group_id not in group_id_set:
                    mention_sent_list.append(mention_context)
                    group_id_set.add(group_id)

                entity_sent_list.append(entity_des)

        no_entity_empty_path = bert_path + "_entity_no_empty"
        with open(no_entity_empty_path, "w", encoding="utf-8") as no_entity_empty_file:
            for index, sent in enumerate(entity_sent_list):
                if sent.strip() != "":
                    no_entity_empty_file.write(sent.strip() + "\n")

        no_mention_empty_path = bert_path + "_mention_no_empty"
        with open(no_mention_empty_path, "w", encoding="utf-8") as no_mention_empty_file:
            for index, sent in enumerate(mention_sent_list):
                if sent.strip() != "":
                    no_mention_empty_file.write(sent.strip() + "\n")

        print("mention num: {0}, entity num: {1}".format(len(mention_sent_list), len(entity_sent_list)))

    def read_vec(self, vec_path):
        """
        read vector from file
        :param vec_path:
        :return:
        """
        vec_list = []
        with open(vec_path, "r", encoding="utf-8") as vec_file:
            for item in vec_file:
                item = item.strip()

                vec = json.loads(item)
                vec_list.append(vec)

        return vec_list

    def map_sent_vector(self, cut_candidate_path, mention_vec_path, entity_vec_path):
        """
        map sent to bert sent vector
        :param source_path:
        :param mention_fea_path:
        :param entity_fea_path:
        :return:
        """
        mention_vec_list = self.read_vec(mention_vec_path)
        entity_vec_list = self.read_vec(entity_vec_path)

        group_id_set = set()
        mention_vec_index = -1
        entity_vec_index = 0
        cut_candidate_list = []
        with open(cut_candidate_path, "r", encoding="utf-8") as cut_candidate_file:
            for item in cut_candidate_file:
                item = item.strip()

                group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_id_str)
                mention_obj = json.loads(mention_str)
                entity_obj = json.loads(entity_str)

                mention_context, entity_des = self.data_util.get_context_desc(mention_obj, entity_obj)

                if group_id not in group_id_set:
                    if mention_context.strip() != "":
                        mention_vec_index += 1
                        mention_obj["sent_vec"] = mention_vec_list[mention_vec_index]

                    group_id_set.add(group_id)
                else:
                    if mention_context.strip() != "":
                        mention_obj["sent_vec"] = mention_vec_list[mention_vec_index]

                if entity_des.strip() != "":
                    entity_obj["sent_vec"] = entity_vec_list[entity_vec_index]
                    entity_vec_index += 1

                cut_candidate_list.append("\t".join([group_id_str, label_str, fea_str,
                                                     json.dumps(mention_obj), json.dumps(entity_obj)]))


        print("mention vec num: {0}, entity vec num: {1}".format(mention_vec_index+1, entity_vec_index))

        with open(cut_candidate_path, "w", encoding="utf-8") as cut_candidate_file:
            for item in cut_candidate_list:
                cut_candidate_file.write(item + "\n")

    def combine_bert_fea(self, rank_format_path, bert_prob_path, new_rank_format_path):
        """

        :param rank_format_path:
        :param bert_prob_path:
        :param new_rank_format_path:
        :return:
        """
        mention_prob_dict = {}
        with open(bert_prob_path, "r", encoding="utf-8") as bert_prob_file:
            for item in bert_prob_file:
                item = item.strip()
                group_id_str = item.split("\t")[2]
                group_id = int(group_id_str)

                if group_id not in mention_prob_dict:
                    mention_prob_dict[group_id] = [item]
                else:
                    mention_prob_dict[group_id].append(item)

        count = 0
        has_bert_prob = False
        with open(rank_format_path, "r", encoding="utf-8") as rank_file:
            with open(new_rank_format_path, "w", encoding="utf-8") as new_rank_file:
                for item in rank_file:
                    item = item.strip()
                    group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                    group_id = int(group_id_str)
                    fea_dict = json.loads(fea_str)
                    entity = json.loads(entity_str)
                    summary = entity["summary"]

                    has_bert_prob = False

                    if group_id in mention_prob_dict:
                        candidate_list = mention_prob_dict[group_id]

                        for candidate in candidate_list:
                            prob_str, candidate_label_str, candidate_group, candidate_context, candidate_summary = candidate.split("\t")
                            if candidate_summary == summary and candidate_label_str == label_str:
                                fea_dict["bert_prob"] = float(prob_str)
                                has_bert_prob = True
                                count += 1
                                break

                    if not has_bert_prob:
                        fea_dict["bert_prob"] = 0.0

                    new_rank_file.write("\t".join([group_id_str, label_str, json.dumps(fea_dict), mention_str, entity_str]) + "\n")

                    if group_id % 100 == 0:
                        print(group_id)

        print("add bert fea:{0}".format(count))

    def cal_sent_sim(self, cut_rank_path):
        """
        calculate cosine distance between mention sent embedding and entity sent embedding,
        the sent embedding is output by bert
        :param cut_rank_path: the candidates have been filtered by xgboost
        :return:
        """

        new_item_list = []
        with open(cut_rank_path, "r", encoding="utf-8") as cut_rank_file:
            for item in cut_rank_file:
                item = item.strip()

                group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_id_str)
                fea_dict = json.loads(fea_str)
                mention_obj = json.loads(mention_str)
                entity_obj = json.loads(entity_str)

                sent_dis = 0.0

                if "sent_vec" in mention_obj and "sent_vec" in entity_obj:
                    sent_dis = self.data_util.cos_distance(mention_obj["sent_vec"], entity_obj["sent_vec"])

                fea_dict["sent_dis"] = sent_dis

                # mention_obj["sent_vec"] = []
                # entity_obj["sent_vec"] = []

                new_item_list.append("\t".join([group_id_str, label_str, json.dumps(fea_dict),
                                                json.dumps(mention_obj), json.dumps(entity_obj)]))

        with open(cut_rank_path, "w", encoding="utf-8") as cut_rank_file:
            for item in new_item_list:
                cut_rank_file.write(item + "\n")

    def global_sim(self, global_doc_mention_path, global_sim_cut_rank_path, adjacent_num):
        """
        calculate cosine distance between current entity embedding and other entity embeddings,
        the entity embedding is output by bert
        :param global_doc_mention_path:
        :param global_sim_cut_rank_path:
        :param adjacent_num:
        :return:
        """
        count = 0
        with open(global_doc_mention_path, "r", encoding="utf-8") as global_doc_mention_file:
            with open(global_sim_cut_rank_path, "w", encoding="utf-8") as global_sim_cut_rank_file:
                for item in global_doc_mention_file:
                    item = item.strip()

                    mention_file, mention_list_str = item.split("\t")
                    mention_list = json.loads(mention_list_str)

                    for index, mention_obj in enumerate(mention_list):
                        adjacent_mention_list = mention_list[min(0, index - adjacent_num): max(len(mention_list),
                                                                                               index + adjacent_num + 1)]

                        for candidate_obj in mention_obj["candidate"]:
                            global_entity_mention_sim = 0.0
                            global_entity_entity_sim = 0.0

                            if "sent_vec" in candidate_obj:
                                candidate_vec = candidate_obj["sent_vec"]

                                mention_vec_list = []
                                other_entity_vec_list = []
                                for adjacent_mention in adjacent_mention_list:
                                    if "sent_vec" in adjacent_mention:
                                        mention_vec_list.append(adjacent_mention["sent_vec"])

                                    for other_candidate in adjacent_mention["candidate"]:
                                        if "sent_vec" in other_candidate:
                                            other_entity_vec_list.append(other_candidate["sent_vec"])

                                global_entity_mention_sim = sum([self.data_util.cos_distance(candidate_vec, mention_vec)
                                                                 for mention_vec in mention_vec_list]) / len(mention_vec_list)

                                global_entity_entity_sim = sum([self.data_util.cos_distance(candidate_vec, entity_vec)
                                                                 for entity_vec in other_entity_vec_list]) / len(other_entity_vec_list)

                            fea_dict = candidate_obj["feature"]
                            fea_dict["global_entity_mention_sim"] = global_entity_mention_sim
                            fea_dict["global_entity_entity_sim"] = global_entity_entity_sim

                            label = candidate_obj["label"]
                            group_id = candidate_obj["group_id"]

                            global_sim_cut_rank_file.write("\t".join([str(group_id), str(label), json.dumps(fea_dict), mention_obj["mention_context"].strip(), json.dumps(candidate_obj)]) + "\n")
                            count += 1
                            if count % 100 == 0:
                                print(count)
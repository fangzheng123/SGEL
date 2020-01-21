# encoding:utf-8

import json
import networkx as nx
import numpy as np
import time
import re
import sys
from datetime import timedelta
from sklearn import preprocessing
import config_util

class DataUtil(object):
    """

    """

    def __init__(self):
        self.word_vocab_path = config_util.word_vocab_path
        self.all_word_embedding_path = config_util.word_embed_path

        self.entity_vocab_path = config_util.entity_vocab_path
        self.all_entity_embedding_path = config_util.entity_embed_path

    def get_time_dif(self, start_time):
        """
        get run time
        :param start_time: 起始时间
        :return:
        """
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def load_word_embed(self):
        """
        load all word embedding
        :return:
        """
        print("Loading word embedding...")
        start_time = time.time()

        word_dict = {}
        word_embed = np.load(self.all_word_embedding_path)

        with open(self.word_vocab_path, "r", encoding="utf-8") as word_vocab_file:
            count = 0
            for item in word_vocab_file:
                item = item.strip()
                word = item.split("\t")[0]
                word = word.lower()
                word_dict[word] = word_embed[count]

                count += 1

        run_time = self.get_time_dif(start_time)
        print("Time usage:{0}, Memory usage: {1} GB".format(run_time, int(sys.getsizeof(word_dict) / (1024 * 1024))))

        return word_dict

    def load_entity_embed(self):
        """
        load all entity embedding
        :return:
        """
        print("Loading word embedding...")
        start_time = time.time()

        entity_embed = np.load(self.all_entity_embedding_path)

        entity_dict = {}
        with open(self.entity_vocab_path, "r", encoding="utf-8") as entity_url_file:
            count = 0
            for item in entity_url_file:
                item = item.strip()
                url = item.split("\t")[0]
                url = url.replace("en.wikipedia.org/wiki/", "")
                url = url.lower()
                entity_dict[url] = entity_embed[count]

                count += 1

        run_time = self.get_time_dif(start_time)
        print("Time usage:{0}, Memory usage: {1} GB".format(run_time, int(sys.getsizeof(entity_dict) / (1024 * 1024))))

        return entity_dict

    def remove_special_char(self, text):
        """
        remove special char from text
        :param text:
        :return:
        """
        special_char = u"[\n.,?!;:$*/'\\#\"\(\)\[\]\{\}\<\>]"
        text = re.sub(special_char, "", text)

        text = re.sub(u"-", " ", text)

        return text

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

    def cos_distance(self, vector1, vector2):
        """
        余弦距离
        :param vector1:
        :param vector2:
        :return:
        """
        cos = 0.0
        vector1_norm = np.linalg.norm(vector1)
        vector2_norm = np.linalg.norm(vector2)
        if len(vector1) == len(vector2) and len(vector1) > 0 \
                and vector1_norm != 0 and vector2_norm != 0:
            cos = np.dot(vector1, vector2) / (vector1_norm * vector2_norm)

        return cos

    def cal_candidate_recall(self, fea_data_path, candidate_num):
        """

        :param candidate_format_path:
        :param candidate_num:
        :return:
        """
        group_id_dict = {}
        with open(fea_data_path, "r", encoding="utf-8") as fea_file:
            for item in fea_file:
                item = item.strip()

                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_str)
                label = int(label_str)

                if group_id in group_id_dict:
                    if len(group_id_dict[group_id]) < candidate_num:
                        group_id_dict[group_id].append(label)
                else:
                    group_id_dict[group_id] = [label]

        recall_count = 0
        for group_id, label_list in group_id_dict.items():
            if 1 in set(label_list):
                recall_count += 1
            else:
                print("not recall group id: {0}".format(group_id))

        print("all count:{0}, recall count:{1}, recall:{2}".format(len(group_id_dict), recall_count, float(recall_count)/len(group_id_dict)))


    def get_mention_num(self, cut_candidate_path):
        """
        get mention num in cut candidate file
        :param cut_candidate_path:
        :return:
        """
        group_id_set = set()
        with open(cut_candidate_path, "r", encoding="utf-8") as cut_candidate_file:
            for item in cut_candidate_file:
                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_str)
                if group_id not in group_id_set:
                    group_id_set.add(group_id)

        return len(group_id_set)

    def process_candidate_name(self, name):
        """

        :param name:
        :return:
        """
        name = name.lower().replace("(", "").replace(")", "").replace(".", "").replace(",", "")
        return name

    def replace_context_keyword(self, context):
        """

        :param context:
        :return:
        """
        new_content = context.replace("soccer", "football").replace("nfl", "football").replace("nba", "basketball")
        return new_content

    def get_context_desc(self, mention_obj, entity_obj):
        """
        load mention context and entity description
        :param mention_obj:
        :param entity_obj:
        :return:
        """
        mention_form = mention_obj["mention_form"].lower()
        entity_name = self.process_candidate_name(entity_obj["name"]).replace("_", " ")

        mention_context = ""
        if "mention_context" in mention_obj:
            mention_context = mention_obj["mention_context"].lower()
            mention_context = self.replace_context_keyword(mention_context)
            mention_context = mention_context.replace(mention_form, "")

            # mention_context = self.remove_stop_word(mention_context)
            # mention_context = " ".join([word for word in mention_context.split(" ") if not word.isdigit()])

        summary_keywords = ""
        if "summary_keywords" in entity_obj:
            summary_keywords = " ".join(entity_obj["summary_keywords"][:30]).lower()
            summary_keywords = self.remove_special_char(summary_keywords)

        category = ""
        if "category" in entity_obj:
            category = " ".join(entity_obj["category"]).lower()
            category = self.remove_special_char(category)

        entity_des = summary_keywords + " " + category

        return mention_context, entity_des

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
                    redirect_dict[source_name] = redirect_name
                    redirect_dict[redirect_name] = source_name

        return redirect_dict

    def get_group_list(self, data_path):
        """
        get dict[group_id]=candidate_list
        :param data_path:
        :return:
        """
        group_item_dict = {}
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_str)
                if group_id not in group_item_dict:
                    group_item_dict[group_id] = [item]
                else:
                    group_item_dict[group_id].append(item)

        return group_item_dict

    def get_file_group(self, data_path):
        """

        :param data_path:
        :return:
        """
        file_group_dict = {}
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()
                group_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                mention_obj = json.loads(mention_str)
                mention_file = mention_obj["mention_file"]
                group_id = int(group_str)

                if mention_file not in file_group_dict:
                    file_group_dict[mention_file] = [group_id]
                elif group_id not in file_group_dict[mention_file]:
                    file_group_dict[mention_file].append(group_id)

        return file_group_dict


    def load_global_data(self, global_graph_path, candidate_num):
        """
        load global data for GAT
        :param global_graph_path:
        :param candidate_num: the number of candidate for each mention
        :return: fea_array(G,N,F), adj_array(G,N,N), mask_array(G,N), mask_label_array(G,N,2),
        """

        all_fea_list = []
        all_adj_list = []
        all_label_list = []
        with open(global_graph_path, "r", encoding="utf-8") as global_graph_file:
            for item in global_graph_file:
                item = item.strip()

                graph_candidate_list = json.loads(item)

                graph_fea_list = []
                graph_label_list = []
                for candidate_obj in graph_candidate_list:
                    candidate_vecs = candidate_obj["vec"]
                    label = candidate_obj["label"]

                    if len(candidate_vecs) == 0:
                        candidate_vecs = [0.0 for i in range(config_util.fea_size)]

                    graph_fea_list.append(candidate_vecs)
                    graph_label_list.append(label)

                # label to one hot
                graph_label_np = np.eye(config_util.class_num)[graph_label_list]

                # # build mask candidate array
                # mask = np.zeros(graph_label_np.shape[0])
                # # the first n entities are needed to be disambiguated, others are masked
                # mask[:candidate_num] = 1
                # graph_mask_array = np.array(mask, dtype=np.bool)

                # build graph and adj
                candidate_index_list = [index for index in range(len(graph_candidate_list))]
                index_list_dict = {}
                for i in range(candidate_num):
                    index_list_dict[i] = candidate_index_list[candidate_num:]

                # Adjacency matrix(N * N)
                graph_adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(index_list_dict)).todense()

                all_fea_list.append(graph_fea_list)
                all_adj_list.append(graph_adj_matrix)
                all_label_list.append(graph_label_np)

        # shape: (G,N,F), (G,N,N), (G,N,2)
        return np.array(all_fea_list), np.array(all_adj_list), np.array(all_label_list)

    def generate_global_batch(self, all_data, batch_size, is_random=False):
        """

        :param all_data:
        :param batch_size:
        :param is_random:
        :return:
        """
        fea_array, adj_array, label_array = all_data

        all_data_num = fea_array.shape[0]
        batch_num = int((all_data_num - 1) / batch_size) + 1

        # shuffle
        if is_random:
            indices = np.random.permutation(np.arange(all_data_num))
            fea_array = fea_array[indices]
            adj_array = adj_array[indices]
            label_array = label_array[indices]

        for i in range(batch_num):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, all_data_num)
            yield fea_array[start_id:end_id], adj_array[start_id:end_id], label_array[start_id:end_id]

    def normalize_feature(self, all_feature_list):
        """
        Min-Max normalization
        :param all_feature_list:
        :return:
        """
        mention_num = len(all_feature_list)
        candidate_num = config_util.global_candidate_num

        all_feas = [feas for candidate_fea_list in all_feature_list for feas in candidate_fea_list]
        norm_feas = preprocessing.minmax_scale(np.array(all_feas))

        all_norm_fea = np.reshape(norm_feas, (mention_num, candidate_num, -1))

        return all_norm_fea

    def load_combine_data(self, data_path, is_train=True):
        """
        load data for combine old_model
        :param data_path:
        :param is_train:
        :return:
        """
        all_mention_list = []
        all_bert_list = []
        all_gat_list = []
        all_fea_list = []
        all_static_score_list = []
        all_label_list = []
        count = 0
        with open(data_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()

                mention_obj = json.loads(item)

                mention_bert = mention_obj["vec"]

                candidate_mention_list = []
                candidate_bert_list = []
                candidate_gat_list = []
                candidate_fea_list = []
                candidate_score_list = []
                candidate_label_list = []
                for index, candidate_obj in enumerate(mention_obj["candidate"]):
                    candidate_bert = candidate_obj["vec"]
                    candidate_gat = candidate_obj["gat_vec"]
                    candidate_fea = candidate_obj["feature"]
                    candidate_fea = [val for name, val in candidate_fea.items()]
                    static_score = candidate_obj["xgboost_pred"]
                    label = candidate_obj["label"]

                    if index == 0 and label == 1:
                        count += 1

                    candidate_mention_list.append(mention_bert)
                    candidate_bert_list.append(candidate_bert)
                    candidate_gat_list.append(candidate_gat)
                    candidate_fea_list.append(candidate_fea)
                    candidate_score_list.append(static_score)
                    candidate_label_list.append(label)

                if is_train and len(set(candidate_label_list)) != 2:
                    continue

                all_mention_list.append(candidate_mention_list)
                all_bert_list.append(candidate_bert_list)
                all_gat_list.append(candidate_gat_list)
                all_fea_list.append(candidate_fea_list)
                all_static_score_list.append(candidate_score_list)
                all_label_list.append(candidate_label_list)

            # normalize feature
            norm_fea_array = self.normalize_feature(all_fea_list)

        print("xgboost group acc: {0}".format(count/len(all_mention_list)))

        return np.array(all_mention_list), np.array(all_bert_list), np.array(all_gat_list), \
               norm_fea_array, np.array(all_static_score_list), np.array(all_label_list)

    def generate_combine_batch(self, all_data, batch_size, is_random=False):
        """

        :param all_data:
        :param batch_size:
        :param is_random:
        :return:
        """
        mention_bert_array, candidate_bert_array, candidate_gat_array, \
        candidate_fea_array, candidate_score_array, label_array = all_data

        all_data_num = mention_bert_array.shape[0]
        batch_num = int((all_data_num - 1) / batch_size) + 1

        if batch_num == 1:
            mention_bert_array = np.repeat(mention_bert_array, 3, axis=0)
            candidate_bert_array = np.repeat(candidate_bert_array, 3, axis=0)
            candidate_gat_array = np.repeat(candidate_gat_array, 3, axis=0)
            candidate_fea_array = np.repeat(candidate_fea_array, 3, axis=0)
            candidate_score_array = np.repeat(candidate_score_array, 3, axis=0)
            label_array = np.repeat(label_array, 3, axis=0)
            all_data_num = mention_bert_array.shape[0]

        # shuffle
        if is_random:
            indices = np.random.permutation(np.arange(all_data_num))
            mention_bert_array = mention_bert_array[indices]
            candidate_bert_array = candidate_bert_array[indices]
            candidate_gat_array = candidate_gat_array[indices]
            candidate_fea_array = candidate_fea_array[indices]
            candidate_score_array = candidate_score_array[indices]
            label_array = label_array[indices]

        for i in range(batch_num):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, all_data_num)
            if end_id == all_data_num:
                yield mention_bert_array[max(end_id-batch_size, 0):end_id], candidate_bert_array[max(end_id-batch_size, 0):end_id], \
                      candidate_gat_array[max(end_id-batch_size, 0):end_id], candidate_fea_array[max(end_id-batch_size, 0):end_id], \
                      candidate_score_array[max(end_id-batch_size, 0):end_id], label_array[max(end_id-batch_size, 0):end_id]
            else:
                yield mention_bert_array[start_id:end_id], candidate_bert_array[start_id:end_id], \
                      candidate_gat_array[start_id:end_id], candidate_fea_array[start_id:end_id], \
                      candidate_score_array[start_id:end_id], label_array[start_id:end_id]

if __name__ == "__main__":
    data_util = DataUtil()
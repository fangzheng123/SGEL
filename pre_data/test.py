# encoding: utf-8


import json
import random
from urllib import parse
from data_util import DataUtil

class Test(object):

    def get_golden_url(self, golden_path):
        golden_list = []
        with open(golden_path, "r", encoding="utf-8") as golden_file:
            for item in golden_file:
                item = item.strip()

                mention_obj = json.loads(item)

                target_url = mention_obj["target_url"]
                target_url = "https://" + target_url
                mention_obj["target_url"] = target_url

                golden_list.append(mention_obj)

        with open(golden_path, "w", encoding="utf-8") as golden_file:
            for mention_obj in golden_list:
                golden_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")



    def no_own_target(self, own_target_path):

        with open(own_target_path, "r", encoding="utf-8") as own_golden_file:
            for item in own_golden_file:
                item = item.strip()

                mention_obj = json.loads(item)
                if "own_target_url" not in mention_obj:
                    print(mention_obj)


    def clean_aida_train(self, aida_train_rank_path):
        """

        :param aida_train_rank_path:
        :return:
        """
        group_entity_dict = {}
        with open(aida_train_rank_path, "r", encoding="utf-8") as aida_train_rank_file:
            for item in aida_train_rank_file:
                item = item.strip()

                group_id_str, label_str, features_str, mention_str, entity_str = item.split("\t")

                group_id = int(group_id_str)

                if group_id not in group_entity_dict:
                    group_entity_dict[group_id] = [item]
                else:
                    group_entity_dict[group_id].append(item)


            for group_id, item_list in group_entity_dict.items():
                for item in item_list:
                    group_id_str, label_str, features_str, mention_str, entity_str = item.split("\t")

                    label = int(label_str)
                    fea_dict = json.loads(features_str)

                    if label == 1 and (fea_dict["name_jaro"] < 0.3 or fea_dict["candidate_recall_rank"] < 0.1):
                        group_entity_dict[group_id] = []
                        break

        count = 0
        clean_path = aida_train_rank_path + "_clean"
        with open(clean_path, "w", encoding="utf-8") as aida_train_clean_rank_file:
            for group_id, item_list in group_entity_dict.items():
                if len(item_list) > 0:
                    count += 1
                for item in item_list:
                    aida_train_clean_rank_file.write(item + "\n")

        print("group num:{0}".format(count))


    def get_aida_train_small(self, aida_train_rank_path):
        """

        :param aida_train_rank_path:
        :return:
        """
        aida_train_small_path = aida_train_rank_path + "_small"

        small_list = []
        with open(aida_train_rank_path, "r", encoding="utf-8") as data_file:
            for item in data_file:
                item = item.strip()

                group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                group_id = int(group_id_str)

                small_list.append(item)

                if int(group_id_str) > 500:
                    break

        with open(aida_train_small_path, "w", encoding="utf-8") as small_file:
            for item in small_list:
                small_file.write(item + "\n")


    def combine_train_data(self, data_path_list, new_path):

        file_len = 0
        group_id = 0
        data_list = []

        for data_path in data_path_list:
            file_len = len(data_list)

            data_name = data_path.split("/")[-1]
            with open(data_path, "r", encoding="utf-8") as data_file:
                for item in data_file:
                    item = item.strip()

                    group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                    group_id = int(group_id_str) + file_len

                    if data_name == "aida_train_rank_format" and int(group_id_str) > 500:
                        break

                    data_list.append("\t".join([str(group_id), label_str, fea_str, mention_str, entity_str]))

        with open(new_path, "w", encoding="utf-8") as new_file:
            for item in data_list:
                new_file.write(item + "\n")

    def golden_2_gerbil(self, golden_path, gerbil_path):
        """

        :param golden_path:
        :return:
        """
        with open(golden_path, "r", encoding="utf-8") as golden_file:
            with open(gerbil_path, "w", encoding="utf-8") as gerbil_file:
                for item in golden_file:
                    item = item.strip()

                    mention_obj = json.loads(item)

                    gerbil_obj = {}
                    gerbil_obj["mentionFile"] = mention_obj["mention_file"]
                    gerbil_obj["mentionName"] = mention_obj["mention_form"]
                    gerbil_obj["mentionOffset"] = mention_obj["mention_offset"]
                    gerbil_obj["mentionLength"] = len(mention_obj["mention_form"])
                    # target_name = mention_obj["target_url"].split("/")[-1]
                    # target_name = parse.quote(target_name)
                    # gerbil_obj["predictUrl"] = "http://en.wikipedia.org/wiki/" + target_name

                    gerbil_obj["predictUrl"] = mention_obj["target_url"]

                    gerbil_file.write(json.dumps(gerbil_obj) + "\n")

    def wiki2name(self, wiki_path):
        name_list = []
        with open(wiki_path, "r", encoding="utf-8") as entity_file:
            for item in entity_file:
                item = item.strip()

                if len(item.split("\t")) != 2:
                    continue

                entity_name, entity_str = item.split("\t")

                name_list.append(entity_name)

        with open(wiki_path+"_name", "w", encoding="utf-8") as name_file:
            for item in name_list:
                name_file.write(item + "\n")


    def combine_redirect(self, candidate_path, golden_redirect_path):
        """

        :param candidate_path:
        :param golden_redirect_path:
        :return:
        """
        mention_keyword_candidate_dict = {}
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)

                mention_keyword_candidate_dict[mention_obj["mention_index"]] = mention_obj["mention_keyword_search"]

        mention_list = []
        with open(golden_redirect_path, "r", encoding="utf-8") as golden_redirect_file:
            for item in golden_redirect_file:
                item = item.strip()

                mention_obj = json.loads(item)

                mention_obj["mention_keyword_search"] = mention_keyword_candidate_dict[mention_obj["mention_index"]]

                mention_list.append(mention_obj)

        with open(golden_redirect_path, "w", encoding="utf-8") as golden_redirect_file:
            for mention_obj in mention_list:
                golden_redirect_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")


    def has_no_wiki(self, filter_candidate_path, wiki_path, redirect_path, url_path):
        """

        :param filter_candidate_path:
        :return:
        """
        data_util = DataUtil()
        redirect_map = data_util.get_redirect_map(redirect_path)
        all_entity_map = data_util.read_entity(wiki_path)

        # url_pre = "https://en.wikipedia.org/wiki/"
        url_pre = ""
        url_set = set()
        with open(filter_candidate_path, "r", encoding="utf-8") as filter_candidate_file:
            for item in filter_candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)

                target_redirect_url = mention_obj["target_redirect_url"]
                target_name = target_redirect_url.split("/")[-1]
                if target_name.find('#') > 0:
                    target_name = target_name.split("#")[0]

                if target_name.lower() in redirect_map:
                    target_name = redirect_map[target_name.lower()]

                if target_name not in all_entity_map:
                    url_set.add(url_pre + target_name)

                candidate_dict = mention_obj["candidate"]
                all_candidate_list = []
                for type, candidate_list in candidate_dict.items():
                    if type == "mention_keyword_search":
                        for ele_list in candidate_list:
                            all_candidate_list.extend(ele_list)
                    else:
                        all_candidate_list.extend(candidate_list)

                for name in all_candidate_list:
                    if name.lower() in redirect_map:
                        name = redirect_map[name.lower()]

                    if name not in all_entity_map:
                        url_set.add(url_pre + name)

        print(len(url_set))
        with open(url_path, "w", encoding="utf-8") as url_file:
            for url in url_set:
                url_file.write(url + "\n")

    def add_wiki(self, source_wiki_path, add_wiki_path):
        """

        :param source_wiki_path:
        :param add_wiki_path:
        :return:
        """
        # get all entity info
        data_util = DataUtil()
        all_entity_dict = data_util.read_entity(source_wiki_path)

        add_entity_dict = data_util.read_entity(add_wiki_path)

        for entity_name, entity_obj in add_entity_dict.items():
            if entity_name not in all_entity_dict:
                all_entity_dict[entity_name] = entity_obj

        with open(source_wiki_path, "w", encoding="utf-8") as all_wiki_file:
            for entity_name, entity_obj in all_entity_dict.items():
                all_wiki_file.write(entity_name + "\t" + json.dumps(entity_obj, ensure_ascii=False) + "\n")

    def build_prior_score(self):
        entity_prior_dict = {}
        data_list = ["ace2004", "msnbc", "aquaint", "clueweb", "wiki", "aida_train", "aida_testA", "aida_testB"]
        for data_name in data_list:
            print(data_name)

            candidate_path = "/home1/fangzheng/data/rlel_data/" + data_name + "/candidate/" + data_name + "_candidate_format"

            with open(candidate_path, "r", encoding="utf-8") as candidate_format_file:
                for item in candidate_format_file:
                    item = item.strip()
                    mention_index_str, label_str, mention_str, entity_str = item.split("\t")
                    entity_obj = json.loads(entity_str)
                    if "name" not in entity_obj:
                        entity_obj["name"] = entity_obj["source_name"]

                    name = entity_obj["name"]
                    yago_score = entity_obj["yago_score"]
                    cross_score = entity_obj["cross_score"]

                    if name not in entity_prior_dict:
                        entity_prior_dict[name] = {"yago_score": yago_score, "cross_score": cross_score}

        prior_path = "/home1/fangzheng/data/bert_el_data/prior"

        with open(prior_path, "w", encoding="utf-8") as prior_file:
            for name, prior_dict in entity_prior_dict.items():
                prior_file.write(name + "\t" + json.dumps(prior_dict, ensure_ascii=False) + "\n")



    def search_noisy(self, rank_format_path):
        """

        :param rank_format_path:
        :return:
        """
        f_num = 0
        s_num = 0
        t_num = 0
        mention_index_set = set()
        with open(rank_format_path, "r", encoding="utf-8") as rank_format_file:
            for item in rank_format_file:
                item = item.strip()
                mention_index_str, label_str, fea_str, mention_str, entity_str = item.split("\t")

                fea_obj = json.loads(fea_str)
                label = int(label_str)
                mention_index = int(mention_index_str)
                if mention_index not in mention_index_set:
                    mention_index_set.add(mention_index)

        print(f_num, s_num, t_num, len(mention_index_set))

    def replace_target_redirect(self, candidate_path, golden_redirect_path):
        """

        :param candidate_path:
        :param golden_redirect_path:
        :return:
        """
        redirect_url_list = []
        with open(golden_redirect_path, "r", encoding="utf-8") as golden_redirect_file:
            for item in golden_redirect_file:
                item = item.strip()

                mention_obj = json.loads(item)
                target_redirect_url = mention_obj["target_redirect_url"]
                redirect_url_list.append(target_redirect_url)

        mention_list = []
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for index, item in enumerate(candidate_file):
                item = item.strip()

                mention_obj = json.loads(item)
                mention_obj["target_redirect_url"] = redirect_url_list[index]
                mention_list.append(mention_obj)

        with open(golden_redirect_path, "w", encoding="utf-8") as golden_redirect_file:
            for mention_obj in mention_list:
                golden_redirect_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")
                golden_redirect_file.flush()

    def read_filter_candidate_name(self, filter_candidate_path, candidate_redirect_path, entity_name_path):
        """

        :param filter_candidate_path:
        :param candidate_redirect_path:
        :param entity_name_path:
        :return:
        """
        # get redirect dict
        data_util = DataUtil()
        redirect_dict = data_util.get_redirect_map(candidate_redirect_path)

        entity_name_set = set()
        with open(filter_candidate_path, "r", encoding="utf-8") as filter_candidate_file:
            for item in filter_candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                target_redirect_url = mention_obj["target_redirect_url"]
                target_name = target_redirect_url.split("/")[-1]
                if target_name.find('#') > 0:
                    target_name = target_name.split("#")[0]
                target_name = parse.unquote(target_name)

                entity_name_set.add(target_name)

                for type_name, candidate_list in mention_obj["candidate"].items():
                    if type_name == "mention_keyword_search":
                        for ele_list in candidate_list:
                            for name in ele_list:
                                if name.lower() in redirect_dict:
                                    name = redirect_dict[name.lower()]

                                entity_name_set.add(name)
                    else:
                        for name in candidate_list:
                            if name.lower() in redirect_dict:
                                name = redirect_dict[name.lower()]

                            entity_name_set.add(name)

        print(len(entity_name_set))

        with open(entity_name_path, "w", encoding="utf-8") as entity_name_file:
            for name in entity_name_set:
                if name == "":
                    continue
                entity_name_file.write(name + "\n")

    def add_page_id(self, filter_candidate_path, page_id_path):
        """

        :param filter_candidate_path:
        :param page_id_path:
        :return:
        """
        data_util = DataUtil()
        name_pageid_dict = data_util.load_page_id(page_id_path)

        target_pageid_dict = {}
        with open(filter_candidate_path, "r", encoding="utf-8") as filter_candidate_file:
            for item in filter_candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                target_redirect_url = mention_obj["target_redirect_url"]
                mention_form = mention_obj["mention_form"]
                target_name = target_redirect_url.split("/")[-1]
                if target_name.find('#') > 0:
                    target_name = target_name.split("#")[0]
                target_name = parse.unquote(target_name)

                if target_name in name_pageid_dict:
                    target_pageid_dict[mention_form + "+" + target_name] = name_pageid_dict[target_name]

        sort_id = sorted(target_pageid_dict.items(), key=lambda x:x[1], reverse=True)
        for name, page_id in sort_id:
            print(name.split("+")[0], name.split("+")[1], page_id)


    def combine_rank_format(self):
        """

        :param rank_format_path:
        :return:
        """
        data_name = "wiki_clueweb"
        source_dir = "/data/fangzheng/rlel/"
        rank_format_path_pre = source_dir + data_name + "/candidate/" + data_name + "_rank_format_"

        item_list = []
        for path in range(12):
            rank_format_path = rank_format_path_pre + str(path)

            with open(rank_format_path, "r", encoding="utf-8") as rank_format_file:
                for item in rank_format_file:
                    item = item.strip()
                    item_list.append(item)

        rank_format_path = source_dir + data_name + "/candidate/" + data_name + "_rank_format"
        with open(rank_format_path, "w", encoding="utf-8") as rank_format_file:
            for item in item_list:
                rank_format_file.write(item + "\n")

    def replace_mention_context(self, filter_candidate_path, train_format_path):
        """

        :param filter_candidate_path:
        :param train_format_path:
        :return:
        """
        mention_context_list = []
        with open(train_format_path, "r", encoding="utf-8") as train_format_file:
            for item in train_format_file:
                item = item.strip()

                mention_obj = json.loads(item)

                mention_context_list.append(mention_obj["mention_context"])

        mention_obj_list = []
        with open(filter_candidate_path, "r", encoding="utf-8") as filter_candidate_file:
            for index, item in enumerate(filter_candidate_file):
                item = item.strip()

                mention_obj = json.loads(item)
                mention_obj["mention_context"] = mention_context_list[index]
                mention_obj_list.append(mention_obj)

        with open(filter_candidate_path, "w", encoding="utf-8") as filter_candidate_file:
            for mention_obj in mention_obj_list:
                filter_candidate_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")


    def combine_common_candidate(self, common_candidate_path, in_wiki_path):
        """

        :param common_candidate_path:
        :param in_wiki_path:
        :return:
        """
        mention_file_dict = {}
        with open(common_candidate_path, "r", encoding="utf-8") as common_candidate_file:
            for item in common_candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_file = mention_obj["mention_file"]

                if "offset" in mention_obj:
                    mention_offset = mention_obj["offset"]
                else:
                    mention_offset = mention_obj["mention_offset"]
                if mention_file not in mention_file_dict:
                    mention_file_dict[mention_file] = {mention_offset: mention_obj}
                else:
                    mention_file_dict[mention_file][mention_offset] = mention_obj

        new_mention_list = []
        with open(in_wiki_path, "r", encoding="utf-8") as in_wiki_file:
            for index, item in enumerate(in_wiki_file):
                item = item.strip()

                current_mention_obj = json.loads(item)
                mention_file = current_mention_obj["mention_file"]
                mention_offset = current_mention_obj["mention_offset"]

                mention_common_obj = mention_file_dict[mention_file][mention_offset]

                candidate_dict = mention_common_obj["candidate"]
                new_candidate_dict = {}
                for name, candidate_list in candidate_dict.items():
                    new_candidate_dict[name] = candidate_list[:20]

                current_mention_obj["candidate"] = new_candidate_dict
                current_mention_obj["target_redirect_url"] = mention_common_obj["target_redirect_url"]

                new_mention_list.append(current_mention_obj)

        with open(in_wiki_path, "w", encoding="utf-8") as in_wiki_file:
            for mention_obj in new_mention_list:
                in_wiki_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")




    def combine_mention_context(self, in_wiki_path, filter_candidate_path):
        """

        :param in_wiki_path:
        :param filter_candidate_path:
        :return:
        """
        all_mention_dict = {}
        with open(in_wiki_path, "r", encoding="utf-8") as in_wiki_file:
            for item in in_wiki_file:
                item = item.strip()

                mention_obj = json.loads(item)

                all_mention_dict[mention_obj["mention_index"]] = mention_obj

        print(len(all_mention_dict))

        mention_obj_list = []
        with open(filter_candidate_path, "r", encoding="utf-8") as filter_candidate_file:
            for item in filter_candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_obj["mention_context"] = all_mention_dict[mention_obj["mention_index"]]["mention_context"]
                mention_obj_list.append(mention_obj)

        print(len(mention_obj_list))

        with open(filter_candidate_path, "w", encoding="utf-8") as filter_candidate_file:
            for mention_obj in mention_obj_list:
                filter_candidate_file.write(json.dumps(mention_obj) + "\n")


    def static_file(self, train_path):
        """

        :param train_path:
        :return:
        """
        mention_set = set()
        with open(train_path, "r", encoding="utf-8") as train_file:
            for item in train_file:
                item = item.strip()

                mention_obj = json.loads(item)

                mention_file = mention_obj["mention_file"]

                mention_set.add(mention_file)

        print(len(mention_set))



if __name__ == "__main__":
    test = Test()

    # test.build_bert_train("/home1/fangzheng/data/bert_el_data/aida/aida_testB_cut_rank_format",
    #                        "/home1/fangzheng/data/bert_el_data/bert/bert_test")

    # test.clean_aida_train("/home1/fangzheng/data/bert_el_data/aida/generate/aida_train_rank_format")
    # test.get_aida_train_small("/home1/fangzheng/data/bert_el_data/aida_train/candidate/aida_train_rank_format")
    # data_path_list = ["/home1/fangzheng/data/bert_el_data/aquaint/candidate/aquaint_rank_format",
    #                   "/home1/fangzheng/data/bert_el_data/aida_train/candidate/aida_train_rank_format_small"]
    # new_data_path = "/home1/fangzheng/data/bert_el_data/xgboost/train_data/aida_aquaint"
    # test.combine_train_data(data_path_list, new_data_path)

    # test.golden_2_gerbil("/Users/fang/Desktop/BestEL/gerbil_data/datasets/aida/aida_golden_testB",
    #                      "/Users/fang/Desktop/BestEL/gerbil_data/datasets/aida/aida_gerbil_testB")

    data_name = "aida_testB"
    candidate_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/source/" + data_name + "_candidate"
    golden_redirect_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/source/" + data_name + "_golden_redirect"
    # test.combine_redirect(candidate_path, golden_redirect_path)

    filter_candidate_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/source/" + data_name + "_filter_candidate"
    wiki_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_wiki"
    redirect_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_candidate_redirect"
    url_path = filter_candidate_path + "_no_wiki"
    # test.has_no_wiki(filter_candidate_path, wiki_path, redirect_path, url_path)

    source_wiki_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_wiki"
    add_wiki_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/add_wiki/" + data_name + "_wiki"
    # test.add_wiki(source_wiki_path, add_wiki_path)

    # test.build_prior_score()

    rank_format_path = "/home1/fangzheng/data/bert_el_data/" + data_name + "/candidate/" + data_name + "_cut_rank_format"
    # test.search_noisy(rank_format_path)

    # data_name = "aida_testA"
    # source_dir = "/data/fangzheng/bert_el/"
    # candidate_path = source_dir + data_name + "/source/" + data_name + "_candidate"
    # golden_redirect_path = source_dir + data_name + "/source/" + data_name + "_golden_redirect"
    # test.replace_target_redirect(candidate_path, golden_redirect_path)

    # data_name = "kore50"
    # source_dir = "/data/fangzheng/bert_el/"
    # filter_candidate_path = source_dir + data_name + "/source/" + data_name + "_filter_candidate"
    # redirect_path = source_dir + data_name + "/candidate/" + data_name + "_candidate_redirect"
    # name_path = source_dir + data_name + "/source/" + data_name + "_filter_candidate_name"
    # test.read_filter_candidate_name(filter_candidate_path, redirect_path, name_path)

    # data_name = "aida_testB"
    # source_dir = "/data/fangzheng/bert_el/"
    # filter_candidate_path = source_dir + data_name + "/source/" + data_name + "_filter_candidate"
    # page_id_path = source_dir + data_name + "/source/" + data_name + "_filter_candidate_pageid"
    # test.add_page_id(filter_candidate_path, page_id_path)

    test.combine_rank_format()

    # data_name = "wiki_clueweb"
    # source_dir = "/data/fangzheng/bert_el/"
    # train_format_path = source_dir + data_name + "/source/" + data_name + "_train_format"
    # filter_candidate_path = source_dir + data_name + "/source/" + data_name + "_filter_candidate"
    # test.replace_mention_context(filter_candidate_path, train_format_path)

    # data_name = "reuters128"
    # source_dir = "/data/fangzheng/bert_el/"
    # common_candidate_path = "/data/fangzheng/tmp/" + data_name + "/candidate/" + data_name + "_candidate"
    # in_wiki_path = source_dir + data_name + "/source/" + data_name + "_golden_redirect"
    # test.combine_common_candidate(common_candidate_path, in_wiki_path)

    # data_name = "wiki"
    # source_dir = "/data/fangzheng/bert_el/"
    # wiki_train_format_path = source_dir + "wiki_clueweb/source/wiki_clueweb_train_format"
    # test.static_file(wiki_train_format_path)
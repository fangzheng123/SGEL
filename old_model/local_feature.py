# encoding: utf-8

import json
import Levenshtein
import data_util
import re
import collections
from urllib import parse

class LocalFeature(object):
    """
    hand design local features for entity linking
    """

    def __init__(self, is_load_embed=True):
        self.is_load_embeds = is_load_embed

        self.data_util = data_util.DataUtil()

        if self.is_load_embeds:
            self.word_embed_dict = self.data_util.load_word_embed()

    def replace_context_keyword(self, context):
        """

        :param context:
        :return:
        """
        new_content = context.replace("soccer", "football").replace("nfl", "football").replace("nba", "basketball")
        return new_content

    def replace_entity_keyword(self, desc):
        """

        :param desc:
        :return:
        """
        new_content = desc.replace("fc", "football").replace("afc", "football")
        return new_content

    def cal_feature(self, mention, candidate_entity):
        """
        calculate features between mention and candidate entities
        :param mention:
        :param candidate_entity:
        :return:
        """
        candidate_pv = self.get_pv(candidate_entity)
        candidate_recall_rank = self.get_candidate_rank(candidate_entity)
        name_jaro = self.get_name_distance(mention, candidate_entity)
        common_word_in_name, is_same_long_name = self.get_common_word_in_name(mention, candidate_entity)
        is_abbre = self.is_abbre(mention, candidate_entity)
        common_name_count, common_summary_count, common_category_count = self.common_word_count(mention, candidate_entity)
        context_summary_word_cos = self.context_keyword_sim(mention, candidate_entity)
        context_category_word_cos = self.context_category_sim(mention, candidate_entity)

        feature_dict = {
            "pageview": candidate_pv,
            "candidate_recall_rank": candidate_recall_rank,
            "name_jaro": round(name_jaro, 6),
            "common_word_in_name": common_word_in_name,
            "is_same_long_name": is_same_long_name,
            "is_abbre": is_abbre,
            "common_name_count": round(common_name_count, 6),
            "common_summary_count": round(common_summary_count, 6),
            "common_category_count": round(common_category_count, 6),
            "context_summary_word_cos": round(context_summary_word_cos, 6),
            "context_category_word_cos": round(context_category_word_cos, 6),
        }
        return feature_dict

    def get_pv(self, candidate_entity):
        """
        get candidate entity pageview
        :param candidate_entity:
        :return:
        """
        candidate_pv = 0
        if "popularity" in candidate_entity and "views_sum" in candidate_entity["popularity"]:
            candidate_pv = candidate_entity["popularity"]["views_sum"]
            candidate_pv = candidate_pv / 1000000.0

        return candidate_pv

    def get_candidate_rank(self, candidate_entity):
        """
        get candidate entity position in list(include disambiguation page, wikipedia search, wikipedia recommend, google search)
        :param candidate_entity:
        :return:
        """
        source_rank = 10

        if "source_position" in candidate_entity:
            source_rank = candidate_entity["source_position"]

        return source_rank

    def get_name_distance(self, mention, candidate_entity):
        """
        ratio and jaro distance between mention_name and entity_name
        :param mention:
        :param candidate_entity:
        :return:
        """
        mention_form = mention["mention_form"].lower()
        entity_name = candidate_entity["name"].lower().replace(",", "")
        entity_name = parse.unquote(entity_name)

        mention_form = mention_form.replace(" ", "_")

        name_jaro = Levenshtein.jaro(mention_form, entity_name)

        return name_jaro

    def get_common_word_in_name(self, mention, candidate_entity):
        """
        common word between mention form and entity name
        :param mention:
        :param candidate_entity:
        :return:
        """
        is_same_long_name = 0

        mention_form = mention["mention_form"].lower()
        entity_name = candidate_entity["name"].lower()
        entity_name = parse.unquote(entity_name)
        entity_name = re.sub(u"[,-.'()]", "_", entity_name)

        mention_word_list = mention_form.strip().split(" ")
        entity_word_list = entity_name.strip().split("_")

        common_words = len(set([word for word in mention_word_list if word != ""]) &
                           set([word for word in entity_word_list if word != ""]))

        if len(mention_word_list) > 2:
            if mention_form.replace(" ", "_") == entity_name:
                is_same_long_name = 1

        return common_words, is_same_long_name

    def is_abbre(self, mention, candidate_entity):
        """
        Determine whether mention is an abbreviation for a candidate entity
        :param mention:
        :param candidate_entity:
        :return:
        """
        is_same = 0

        mention_form = mention["mention_form"]
        entity_name = candidate_entity["name"].lower()
        entity_name = parse.unquote(entity_name)

        mention_first_word = mention_form.split(" ")[0]
        first_word_len = len(mention_first_word)
        if first_word_len < 4 and mention_first_word.isupper():
            entity_word_list = entity_name.split("_")

            if len(entity_word_list) >= first_word_len:
                first_chars = "".join([word[0] for word in entity_word_list if len(word) > 0])
                if mention_first_word.lower() == first_chars.lower():
                    is_same = 1

        return is_same

    def common_word_count(self, mention, candidate_entity):
        """
        common word between: mention context and summary, mention context and category, mention context and entity name
        :param mention:
        :param candidate_entity:
        :return:
        """
        common_name_count = 0
        common_summary_count = 0
        common_category_count = 0

        mention_form = mention["mention_form"].lower()
        entity_name = candidate_entity["name"].lower()
        entity_name = parse.unquote(entity_name)
        entity_name = re.sub(u"[,-.'()]", "_", entity_name)
        entity_name = self.replace_entity_keyword(entity_name)

        mention_context = mention["mention_context"].lower()
        mention_context = self.replace_context_keyword(mention_context)
        summary_keywords = candidate_entity["summary_keywords"]
        summary_keywords = " ".join(summary_keywords).lower()

        category_list = candidate_entity["category"]
        category = ""
        if len(category_list) > 0:
            category = " ".join(category_list[1:]).lower()

        mention_context = self.data_util.remove_special_char(mention_context)
        summary_keywords = self.data_util.remove_special_char(summary_keywords)
        category = self.data_util.remove_special_char(category)

        mention_context = self.data_util.remove_stop_word(mention_context)
        summary_keywords = self.data_util.remove_stop_word(summary_keywords)
        category = self.data_util.remove_stop_word(category)

        # avoid introducing noise
        mention_context = mention_context.replace(mention_form, "")

        common_summary_words = set([word for word in mention_context.split(" ") if word != ""]) \
                       & set(word for word in summary_keywords.split(" ") if word != "")

        common_category_words = set([word for word in mention_context.split(" ") if word != ""]) \
                               & set(word for word in category.split(" ") if word != "")

        context_word_count = collections.Counter(mention_context.split(" "))
        summary_word_count = collections.Counter(summary_keywords.split(" "))
        category_word_count = collections.Counter(category.split(" "))

        all_context_word_num = sum([val for key, val in dict(context_word_count).items()])
        all_summary_word_num = sum([val for key, val in dict(summary_word_count).items()])
        all_category_word_num = sum([val for key, val in dict(category_word_count).items()])

        for word in common_summary_words:
            common_summary_count += context_word_count[word] * summary_word_count[word]

        for word in common_category_words:
            common_category_count += context_word_count[word] * category_word_count[word]

        if all_context_word_num + all_summary_word_num != 0:
            common_summary_count = float(common_summary_count) / (all_context_word_num + all_summary_word_num)

        if all_context_word_num + all_category_word_num != 0:
            common_category_count = float(common_category_count) / (all_context_word_num + all_category_word_num)

        entity_name = self.data_util.remove_stop_word(" ".join(entity_name.split("_")))
        if len(entity_name.split(" ")) > 1:
            common_name_words = set([word for word in mention_context.split(" ") if word != ""]) \
                                & set(word for word in entity_name.split(" ") if word != "")
            name_word_count = collections.Counter(entity_name.split(" "))
            for word in common_name_words:
                common_name_count += context_word_count[word] * name_word_count[word]

            common_name_count /= len(entity_name.split(" "))

        return common_name_count, common_summary_count, common_category_count

    def context_keyword_sim(self, mention, candidate_entity):
        """
        calculate cosine distance between mention context embedding and entity summary keyword embedding
        (embedding is trained from w2v)
        :param mention:
        :param candidate_entity:
        :return:
        """
        cos_distance = 0.0

        mention_form = mention["mention_form"].lower()

        mention_context = mention["mention_context"].lower()
        summary_keywords = candidate_entity["summary_keywords"]
        summary_keywords = " ".join(summary_keywords).lower()

        mention_context = self.data_util.remove_special_char(mention_context)
        mention_context = self.replace_context_keyword(mention_context)
        summary_keywords = self.data_util.remove_special_char(summary_keywords)

        mention_context = self.data_util.remove_stop_word(mention_context)
        summary_keywords = self.data_util.remove_stop_word(summary_keywords)

        # avoid introducing noise
        mention_context = mention_context.replace(mention_form, "")

        mention_context_embed = []
        for word in mention_context.split(" "):
            if word in self.word_embed_dict:
                mention_context_embed.append(self.word_embed_dict[word])

        summary_embedd = []
        for word in summary_keywords.split(" "):
            if word in self.word_embed_dict:
                summary_embedd.append(self.word_embed_dict[word])

        if len(mention_context_embed) > 0 and len(summary_embedd) > 0:
            for context_word_embedd in mention_context_embed:
                for summary_word_embedd in summary_embedd:
                    cos_distance += self.data_util.cos_distance(context_word_embedd, summary_word_embedd)

            if len(mention_context_embed) * len(summary_embedd) != 0:
                cos_distance = float(cos_distance) / (len(mention_context_embed) * len(summary_embedd))

        return cos_distance

    def context_category_sim(self, mention, candidate_entity):
        """
        calculate cosine distance between mention context embedding and entity category embedding
        (embedding is trained from w2v)
        :param mention:
        :param candidate_entity:
        :return:
        """
        cos_distance = 0.0

        mention_form = mention["mention_form"].lower()
        mention_context = mention["mention_context"].lower()
        mention_context = self.replace_context_keyword(mention_context)

        category_list = candidate_entity["category"]
        category = ""
        if len(category_list) > 0:
            category = " ".join(category_list[1:]).lower()

        if len(category) > 0:
            mention_context = self.data_util.remove_special_char(mention_context)
            category = self.data_util.remove_special_char(category)

            mention_context = self.data_util.remove_stop_word(mention_context)
            category = self.data_util.remove_stop_word(category)

            # avoid introducing noise
            mention_context = mention_context.replace(mention_form, "")

            mention_context_embed = []
            for word in mention_context.split(" "):
                if word in self.word_embed_dict:
                    mention_context_embed.append(self.word_embed_dict[word])

            category_embedd = []
            for word in category.split(" "):
                if word in self.word_embed_dict:
                    category_embedd.append(self.word_embed_dict[word])

            if len(mention_context_embed) > 0 and len(category_embedd) > 0:
                for context_word_embedd in mention_context_embed:
                    for category_word_embedd in category_embedd:
                        cos_distance += self.data_util.cos_distance(context_word_embedd, category_word_embedd)

                if len(mention_context_embed) * len(category_embedd) != 0:
                    cos_distance = float(cos_distance) / (len(mention_context_embed) * len(category_embedd))

        return cos_distance


    def build_rank_feature(self, candidate_path, rank_format_path):
        """

        :param candidate_path:
        :param rank_format_path:
        :return:
        """
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            with open(rank_format_path, "w", encoding="utf-8") as rank_format_file:
                for item in candidate_file:
                    item = item.strip()
                    group_str_id, label_str, mention_str, entity_str = item.split("\t")

                    group_id = int(group_str_id)

                    mention = json.loads(mention_str)
                    candidate_entity = json.loads(entity_str)

                    feature_dict = self.cal_feature(mention, candidate_entity)

                    rank_format_file.write(group_str_id + "\t" + label_str + "\t" + json.dumps(feature_dict) + "\t"
                                           + mention_str + "\t" + entity_str + "\n")
                    rank_format_file.flush()

                    if group_id % 100 == 50:
                        print(group_id)



if __name__ == "__main__":
    el_fea = LocalFeature(is_load_embed=True)

    data_name = "wiki_clueweb"
    source_dir = "/home1/fangzheng/data/bert_el_data/"

    candidate_path = source_dir + data_name + "/candidate/" + data_name + "_candidate_format"
    rank_format_path = source_dir + data_name + "/candidate/" + data_name + "_rank_format"
    cut_rank_path = source_dir + data_name + "/candidate/" + data_name + "_cut_rank_format"

    print(data_name)

    el_fea.build_rank_feature(candidate_path, rank_format_path)




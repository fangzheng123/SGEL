# encoding: utf-8

from base_process import BaseProcess
import json

class WikiProcess(BaseProcess):
    """
    process wiki and clueweb data
    """
    def generate_train_data(self, wiki_path, clueweb_path, wiki_train_path):
        """
        combine wiki and clueweb data, filter data and generate training data
        :param wiki_path:
        :param clueweb_path:
        :param wiki_clueweb_path:
        :return:
        """
        # combine data
        all_data_list = []
        with open(wiki_path, "r", encoding="utf-8") as wiki_file:
            for item in wiki_file:
                item = item.strip()
                all_data_list.append(item)

        with open(clueweb_path, "r", encoding="utf-8") as clueweb_file:
            for item in clueweb_file:
                item = item.strip()
                all_data_list.append(item)

        # filter mention
        filter_data_list = []
        for item in all_data_list:
            tmp_list = item.split("\t")

            mention_form = tmp_list[2]

            if len(mention_form) < 3 or mention_form.replace("\u2013", "").isdigit():
                continue

            filter_data_list.append(item)

        print("all data num:{0}, filter data num:{1}".format(len(all_data_list), len(filter_data_list)))

        with open(wiki_train_path, "w", encoding="utf-8") as wiki_train_file:
            for item in filter_data_list:
                wiki_train_file.write(item + "\n")

    def format_other_data(self, other_format_path, our_format_path):
        """
        format data constructed by others
        :param other_format_path:
        :param our_format_path:
        :return:
        """
        with open(other_format_path, "r", encoding="utf-8") as other_format_file:
            with open(our_format_path, "w", encoding="utf-8") as our_format_file:
                for mention_index, item in enumerate(other_format_file):
                    item = item.strip()
                    tmp_list = item.split("\t")
                    mention_file = tmp_list[0]
                    mention_form = tmp_list[2]
                    mention_left_context = tmp_list[3]
                    mention_right_context = tmp_list[4]

                    if len(tmp_list[-1].split(",")) <= 4:
                        target_name = tmp_list[-1].split(",")[-1]
                    else:
                        gt = tmp_list[-1]
                        target_offset = gt.index(gt.split(",")[3][0])
                        target_name = gt[target_offset:]

                    target_index = int(tmp_list[-1].split(",")[0])

                    candidate_index = [index for index, ele in enumerate(tmp_list) if ele == "CANDIDATES"][0]
                    candidate_list = tmp_list[candidate_index + 1: -2]

                    candidate_obj_list = []
                    for ele in candidate_list:
                        if ele == "EMPTYCAND":
                            continue

                        candidate = {}

                        if len(ele.split(",")) <= 3:
                            candidate_name = ele.split(",")[-1].strip().replace(" ", "_")
                        else:
                            name_offset = ele.index(ele.split(",")[2][0])
                            candidate_name = ele[name_offset:]
                            candidate_name = candidate_name.strip().replace(" ", "_")

                        candidate["name"] = candidate_name
                        candidate["yago_score"] = float(ele.split(",")[0])
                        candidate["cross_score"] = float(ele.split(",")[1])

                        candidate_obj_list.append(candidate)

                    mention_obj = {}
                    target_name = target_name.strip().replace(" ", "_")
                    mention_obj["target_url"] = "http://en.wikipedia.org/wiki/" + target_name
                    mention_obj["target_name"] = target_name
                    mention_obj["mention_form"] = mention_form
                    mention_obj["mention_context"] = mention_left_context + " " + mention_right_context
                    mention_obj["mention_file"] = mention_file
                    # mention_obj["candidate"] = candidate_obj_list
                    mention_obj["mention_index"] = mention_index

                    our_format_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")

    def controll_wiki(self):
        source_dir = "/data/fangzheng/bert_el/"
        wiki_source_path = source_dir + "wiki_clueweb/source/wiki"
        clueweb_source_path = source_dir + "wiki_clueweb/source/clueweb"
        wiki_train_path = source_dir + "wiki_clueweb/source/wiki_clueweb_train"
        wiki_train_format_path = source_dir + "wiki_clueweb/source/wiki_clueweb_train_format"
        person_path = source_dir + "persons.txt"


        # self.generate_train_data(wiki_source_path, clueweb_source_path, wiki_train_path)
        # self.format_other_data(wiki_train_path, wiki_train_format_path)
        # self.get_father_mention(wiki_train_format_path, person_path)
        # self.get_adjacent_mention(wiki_train_format_path)
        # self.get_mention_synonym(wiki_train_format_path)
        # self.get_context_keyword(wiki_train_format_path)

        # crawl redirect page
        # crawl candidate



if __name__ == "__main__":

    wiki_process = WikiProcess()

    wiki_process.controll_wiki()

# encoding: utf-8

import json

class Tmp(object):

    def combine_wiki(self, wiki_wiki_path, clueweb_wiki_path, wiki_clueweb_wiki_path):
        """

        :param wiki_wiki_path:
        :param clueweb_wiki_path:
        :param wiki_clueweb_wiki_path:
        :return:
        """
        entity_dict = {}
        with open(wiki_wiki_path, "r", encoding="utf-8") as wiki_wiki_file:
            for item in wiki_wiki_file:
                item = item.strip()

                if len(item.split("\t")) != 2:
                    continue

                name, entity_str = item.split("\t")

                entity_dict[name] = entity_str

        with open(clueweb_wiki_path, "r", encoding="utf-8") as clueweb_wiki_file:
            for item in clueweb_wiki_file:
                item = item.strip()

                if len(item.split("\t")) != 2:
                    continue

                name, entity_str = item.split("\t")

                if name not in entity_dict:
                    entity_dict[name] = entity_str

        with open(wiki_clueweb_wiki_path, "w", encoding="utf-8") as wiki_clueweb_wiki_file:
            for name, entity_str in entity_dict.items():
                wiki_clueweb_wiki_file.write(name + "\t" + entity_str + "\n")

    def combine_candidate_redirect(self, wiki_redirect_path, clueweb_redirect_path, wiki_clueweb_redirect_path):
        """

        :param wiki_redirect_path:
        :param clueweb_redirect_path:
        :param wiki_clueweb_redirect_path:
        :return:
        """
        redirect_dict = {}
        with open(wiki_redirect_path, "r", encoding="utf-8") as wiki_redirect_file:
            for item in wiki_redirect_file:
                item = item.strip()

                name, redirect_name = item.split("\t")
                redirect_dict[name] = redirect_name

        with open(clueweb_redirect_path, "r", encoding="utf-8") as clueweb_redirect_file:
            for item in clueweb_redirect_file:
                item = item.strip()

                name, redirect_name = item.split("\t")
                if name not in redirect_dict:
                    redirect_dict[name] = redirect_name

        with open(wiki_clueweb_redirect_path, "w", encoding="utf-8") as wiki_clueweb_redirect_file:
            for name, redirect_name in redirect_dict.items():
                wiki_clueweb_redirect_file.write(name + "\t" + redirect_name + "\n")


if __name__ == "__main__":
    tmp = Tmp()

    source_dir = "/data/fangzheng/rlel/"

    wiki_wiki_path = source_dir + "wiki/source/wiki_candidate_redirect"
    clueweb_wiki_path = source_dir + "clueweb/source/clueweb_candidate_redirect"
    wiki_clueweb_wiki_path = source_dir + "wiki_clueweb/candidate/wiki_clueweb_candidate_redirect"
    tmp.combine_wiki(wiki_wiki_path, clueweb_wiki_path, wiki_clueweb_wiki_path)






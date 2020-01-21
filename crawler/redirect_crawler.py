# encoding: utf-8

import json
import requests
from lxml import etree
from urllib import parse
import shutil

class RedirectCrawler(object):
    def __init__(self):
        pass

    def crawl_redirect_page(self, target_url):
        """
        Get redirected pages of entity
        :param target_url:
        :return:
        """
        target_redirect_url = ""

        try:
            response = requests.get(url=target_url)
            response.encoding = "utf-8"
            html = etree.HTML(response.text)
            script_str = html.xpath("/html/body/script[@type='application/ld+json']/text()")[0]

            entity_json = json.loads(script_str)

            if "url" in entity_json:
                url = entity_json["url"]
                target_redirect_url = url.replace("\\", "")

        except:
            print("crawl redirected error")

        # print(target_url, own_target_url)

        return target_redirect_url

    def crawl_golden_redirect(self, golden_path, golden_redirect_path):
        """
        Get redirected pages of gold entities
        :param golden_path:
        :param golden_redirect_path:
        :return:
        """
        target_redirect_count = 0
        with open(golden_path, "r", encoding="utf-8") as golden_file:
            with open(golden_redirect_path, "w", encoding="utf-8") as golden_redirect_file:
                for item in golden_file:
                    item = item.strip()

                    mention_obj = json.loads(item)
                    target_url = mention_obj["target_url"]

                    target_redirect_url = self.crawl_redirect_page(target_url)

                    # disambiguation url
                    disam_url = target_url + "_(disambiguation)"
                    target_disam_redirect_url = self.crawl_redirect_page(disam_url)

                    if target_redirect_url != "":
                        # this entity is a disambiguation entity
                        if target_redirect_url == target_disam_redirect_url:
                            mention_obj["target_redirect_url"] = disam_url
                        else:
                            mention_obj["target_redirect_url"] = target_redirect_url

                        target_redirect_count += 1
                    else:
                        mention_obj["target_redirect_url"] = ""

                    golden_redirect_file.write(json.dumps(mention_obj, ensure_ascii=False) + "\n")
                    golden_redirect_file.flush()

        print(target_redirect_count)

    def crawl_candidate_redirect(self, candidate_path, redirect_dict_path):
        """
        crawl candidate entity redirect name
        :param candidate_path:
        :param redirect_dict_path:
        :return:
        """
        name_set = set()
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                candidate_dict = mention_obj["candidate"]

                for candidate_type, candidate_list in candidate_dict.items():
                    if candidate_type == "mention_keyword_search":
                        tmp_list = []
                        for ele_list in candidate_list:
                            tmp_list.extend(ele_list)
                        candidate_list = tmp_list

                    for candidate_name in candidate_list:
                        if candidate_name == "" or candidate_name.__contains__("disambiguation") \
                                or candidate_name.__contains__("#"):
                            continue

                        name_set.add(candidate_name)

                # for candidate_obj in candidate_dict:
                #     candidate_name = candidate_obj["name"]
                #     name_set.add(candidate_name)

        print(len(name_set))
        with open(redirect_dict_path, "w", encoding="utf-8") as redirect_dict_file:
            for name in name_set:
                url = "https://en.wikipedia.org/wiki/" + parse.quote(name)
                redirect_url = self.crawl_redirect_page(url)
                redirect_name = redirect_url.split("/")[-1].split("#")[0]

                if redirect_name != "" and parse.unquote(name) != parse.unquote(redirect_name):
                    redirect_dict_file.write(name + "\t" + redirect_name + "\n")
                    redirect_dict_file.flush()

        dump_path = redirect_dict_path + ".dump"
        shutil.copyfile(redirect_dict_path, dump_path)

    def crawl_name_redirect(self, name_path, redirect_dict_path):
        """
        crawl name redirect
        :param name_path:
        :param redirect_dict_path:
        :return:
        """
        with open(redirect_dict_path, "w", encoding="utf-8") as redirect_dict_file:
            with open(name_path, "r", encoding="utf-8") as name_file:
                for item in name_file:
                    name = item.strip()

                    if name == "":
                        continue

                    url = "https://en.wikipedia.org/wiki/" + parse.quote(name)
                    redirect_url = self.crawl_redirect_page(url)
                    redirect_name = redirect_url.split("/")[-1].split("#")[0]

                    if redirect_name != "" and parse.unquote(name) != parse.unquote(redirect_name):
                        redirect_dict_file.write(name + "\t" + redirect_name + "\n")
                        redirect_dict_file.flush()


if __name__ == "__main__":
    redirect_crawl = RedirectCrawler()

    # data_list = ["ace2004", "msnbc", "aquaint", "clueweb", "wiki", "aida_train", "aida_testA", "aida_testB"]

    data_type = "aida_train"
    candidate_path = "/root/fangzheng/data/candidate/" + data_type + "_candidate"
    candidate_redirect_path = "/root/fangzheng/data/golden_redirect/" + data_type + "_golden_redirect"
    # redirect_crawl.crawl_golden_redirect(candidate_path, candidate_redirect_path)

    filter_candidate_path = "/root/fangzheng/data/filter_candidate/" + data_type + "_filter_candidate"
    filter_candidate_redirect_path = "/root/fangzheng/data/candidate_redirect/" + data_type + "_candidate_redirect"
    redirect_crawl.crawl_candidate_redirect(filter_candidate_path, filter_candidate_redirect_path)

    name_path = "/root/fangzheng/data/filter_candidate/" + data_type + "_filter_candidate_no_wiki"
    filter_candidate_redirect_path = "/root/fangzheng/data/candidate_redirect/" + data_type + "_candidate_redirect"
    # redirect_crawl.crawl_name_redirect(name_path, filter_candidate_redirect_path)

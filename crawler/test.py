# encoding:utf-8

import json
import wiki_crawler
from lxml import etree
import urllib
import urllib.request
import requests

class Test(object):
    """
    test func
    """

    def code2char(self, url):
        code_dict = {"%26": "&", "%27": "'"}

        for code, char in code_dict.items():
            if url.__contains__(code):
                url = url.replace(code, char)

        return url


    def candidate_match(self, golden_path, candidate_path):
        """
        Test whether the gold entity is in the candidate list
        :param golden_path:
        :param candidate_path:
        :return:
        """
        with open(golden_path, "r", encoding="utf-8") as golden_file:
            with open(candidate_path, "r", encoding="utf-8") as candidate_file:
                candidate_dict = {}
                for item in candidate_file:
                    mention, candidates = item.strip().split("\t")
                    tmp_dict = json.loads(candidates)
                    url_list = []
                    for key, eles in tmp_dict.items():
                        for ele in eles:
                            ele = self.code2char(ele)
                            url_list.append(ele.lower())

                            candidate_dict[mention.lower()] = url_list

                count = 0
                unknown_count = 0
                recall_count = 0
                unrecall_mention_set = set()
                for item in golden_file:
                    tmp_dict = json.loads(item.strip())
                    mention = tmp_dict["mention_form"]
                    golden_url = tmp_dict["target_url"].lower()
                    ele = tmp_dict["target_url"].split("/")[-1].lower()
                    if golden_url.__contains__("unknown"):
                        unknown_count += 1

                    if ele in set(candidate_dict[mention.lower()]) or golden_url.__contains__("unknown"):
                        recall_count += 1
                    else:
                        if mention.lower() not in unrecall_mention_set:
                            unrecall_mention_set.add(mention.lower())
                        print(mention, ele)
                        print(candidate_dict[mention.lower()])

                    count += 1

                print(recall_count, unknown_count, count, len(unrecall_mention_set), recall_count*1.0/count)

    def read_unknown_candidate(self, candidate_path, unknown_path):
        """

        :param candidate_path:
        :param unknown_path:
        :return:
        """
        mention_list = []
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()
                mention, candidate_str = item.split("\t")
                candidate_dict = json.loads(candidate_str)
                if "unknown" in candidate_dict:
                    mention_list.append(mention)

        print("unknown list: " + str(len(mention_list)))
        with open(unknown_path, "w", encoding="utf-8") as unknown_file:
            for mention in mention_list:
                unknown_file.write(mention + "\n")


    def combine_unknown(self, candidate_path, unknown_path, combine_path):
        """

        :param candidate_path:
        :param unknown_path:
        :param combine_path:
        :return:
        """
        mention_google_dict = {}
        with open(unknown_path, "r", encoding="utf-8") as unknown_file:
            for item in unknown_file:
                item = item.strip()
                obj = json.loads(item)

                mention = obj["mention"]
                url_list = [url["link"].split("/")[-1] for url in obj["uris"] if "link" in url]

                mention_google_dict[mention] = url_list

        print("google search: " + str(len(mention_google_dict)))

        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            with open(combine_path, "w", encoding="utf-8") as combine_file:
                for item in candidate_file:
                    item = item.strip()
                    mention, candidate_str = item.split("\t")
                    mention_candidate = json.loads(candidate_str)

                    if "unknown" in mention_candidate:
                        mention_candidate["google_search"] = mention_google_dict[mention]

                    combine_file.write(mention + "\t" + json.dumps(mention_candidate, ensure_ascii=False) + "\n")

    def crawl_other_candidate_pv(self, candidate_path):
        """

        :param candidate_path:
        :return:
        """
        crawler = wiki_crawler.WikiEntityCrawler("", "")

        crawl_pv_path = candidate_path + "_crawl_pv"
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            with open(crawl_pv_path, "w", encoding="utf-8") as crawl_pv_file:
                for item in candidate_file:
                    item = item.strip()

                    if len(item.split("\t")) == 1:
                        candidate_pv = 0

                        pv_dict = crawler.get_pv(item)
                        if "views_sum" in pv_dict:
                            candidate_pv = pv_dict["views_sum"]
                            candidate_pv = candidate_pv / 1000000.0

                        crawl_pv_file.write(item + "\t" + str(candidate_pv) + "\n")
                        crawl_pv_file.flush()

    def get_page_id(self, entity_name):
        """
        get entity info
        :param entity_name:
        :return:
        """
        page_id = 0

        def make_page_info_url(entity_name):
            return "https://en.wikipedia.org/w/index.php?title=" + entity_name + "&action=info"

        try:
            response = requests.get(url=make_page_info_url(entity_name))
            response.encoding = 'utf8'
            html = etree.HTML(response.text)
            page_id_str = html.xpath("//*[@id='mw-pageinfo-article-id']/td[2]/text()")[0]
            page_id = int(page_id_str)
        except:
            print("error in get_page_info")
            page_id = 0

        return page_id

    def start_page_id_run(self, entity_name_path, page_id_path):
        """

        :param entity_name_path:
        :return:
        """
        with open(entity_name_path, "r", encoding="utf-8") as entity_name_file:
            with open(page_id_path, "w", encoding="utf-8") as page_id_file:
                for item in entity_name_file:
                    item = item.strip()

                    page_id = self.get_page_id(item)
                    page_id_file.write(item + "\t" + str(page_id) + "\n")
                    page_id_file.flush()

if __name__ == "__main__":

    test = Test()
    data_type = "wiki_clueweb"
    entity_name_path = "/root/fangzheng/data/page_id/" + data_type + "_filter_candidate_name"
    page_id_path = "/root/fangzheng/data/page_id/" + data_type + "_filter_candidate_pageid"
    test.start_page_id_run(entity_name_path, page_id_path)

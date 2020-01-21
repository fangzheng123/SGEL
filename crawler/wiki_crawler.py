# encoding:utf-8

from lxml import etree
import urllib
import urllib.request
import requests
import json
import os
from urllib import parse

class WikiEntityCrawler:
    """
    Crawl Wikipedia Data
    """
    def __init__(self):
        """
        :param save_path:
        """
        pass

    def page_analyze(self, url):
        """
        get page info
        :param url: page url
        :return:
        """
        url = url + "?action=raw"

        try:
            response = requests.get(url=url)
            response.encoding = "utf-8"
            raw_text = response.text
        except:
            raw_text = ""

        # analyse raw text
        result = {}
        result["infobox"] = self.get_infobox(raw_text)
        result["summary"], result["content"] = self.get_content(raw_text)
        result["category"] = self.get_category(raw_text)

        return result

    def get_infobox(self, raw_text):
        """

        :param raw_text:
        :return:
        """
        infobox = {}
        try:
            line_list = raw_text.split("\n")
            start_index = 0
            end_index = 0
            start_flag = False
            for index, line in enumerate(line_list):
                if len(line) > 0 and line.__contains__("{{Infobox"):
                    start_index = index
                    start_flag = True

                if start_flag and line == "}}" and index+1 != len(line_list):
                    if len(line_list[index+1]) == 0 or line_list[index+1][0] != "|":
                        end_index = index

                if end_index != 0 or end_index > 200:
                    break

            infobox_list = []
            if start_index != 0 and end_index != 0:
                infobox_list = line_list[start_index:end_index+1]

            for index, info_item in enumerate(infobox_list):
                info_item = info_item.strip()
                if index == 0:
                    if info_item.__contains__(" ") and len(info_item) > info_item.index(" ")+1:
                        type = info_item[info_item.index(" ")+1:]
                        if len(type) > 0:
                            infobox["top_type"] = type
                    continue

                if len(info_item) > 0 and info_item[0] == "|" and info_item.__contains__("="):
                    split_index = info_item.index("=")
                    key = info_item[:split_index]
                    value = info_item[split_index+1:]
                    key = key.strip().replace("|", "").replace(" ", "")
                    value = value.strip()
                    if len(value) > 0:
                        infobox[key] = value

        except:
            print("error in get_infobox")
            infobox = {}

        return infobox

    def get_content(self, raw_text):
        """

        :param raw_text:
        :return:
        """
        summary, content = "", ""
        try:
            line_list = raw_text.split("\n")
            start_index = 0
            end_index = 0
            for index, line in enumerate(line_list):
                line = line.strip()
                if index == len(line_list) - 1:
                    continue

                if line == "}}":
                    if (len(line_list[index + 1]) > 0 and line_list[index + 1][0] != "|") \
                            or len(line_list[index + 1]) == 0:
                        start_index = index + 1
                elif len(line) > 1 and line[0] == "|" and "".join(line[-2:]) == "}}":
                    if (len(line_list[index + 1]) > 0 and line_list[index + 1][0] != "|") \
                            or len(line_list[index + 1]) == 0:
                        start_index = index + 1

                if len(line) > 0 and line[0] == "=" and line[-1] == "=":
                    end_index = index - 1

                if end_index != 0:
                    break

            summary_list = []
            if start_index < end_index:
                summary_list = line_list[start_index:end_index]
            content_list = line_list[end_index+1:]

            if len(summary_list) > 0:
                summary = "\n".join(summary_list)
            if len(content_list) > 0:
                content = "\n".join(content_list)
        except:
            print("error in get_content")
            summary, content = "", ""

        return summary, content

    def get_category(self, raw_text):
        """
        get category
        :param html:
        :return:
        """
        category_list = []

        try:
            line_list = raw_text.split("\n")

            for line in reversed(line_list):
                if len(line) == 0:
                    break

                if line.__contains__("Category"):
                    line = line.replace("[", "").replace("]", "")
                    category_item = line[line.index(":")+1:]
                    category_list.append(category_item)

        except:
            print("error in get_category")
            category_list = []

        category_list.reverse()
        return category_list

    def get_pv(self, entity_name):
        """
        get entity pageview
        :param entity_name:
        :return:
        """
        res = {}

        def make_pv_url(entity_name):
            return "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/" + \
                   entity_name + \
                   "/monthly/1015070100/2019080900"
        try:
            response = requests.get(url=make_pv_url(entity_name))
            response.encoding = 'utf8'
            view_num = 0
            data = json.loads(response.text)["items"]
            if len(data) > 0:
                for item in data:
                    view_num += int(item["views"])
                res["views_sum"] = view_num
                res["views_recent"] = data[-1]["views"]
            else:
                res["views_sum"] = -1
                res["views_recent"] = -1
        except:
            print("error in get_pv")
            res = {}

        return res

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


    def crawl_single_page(self, url):
        """
        crawl singe entity page
        :param url:
        :return:
        """
        entity_dict = {}
        try:
            entity_dict = self.page_analyze(url)
            entity_dict["url"] = url
            entity_dict["name"] = url.split("/")[-1]
            entity_dict["popularity"] = self.get_pv(entity_dict["name"])
            entity_dict["page_id"] = self.get_page_id(entity_dict["name"])
        except:
            entity_dict = {}
            print("crawl error: " + url)

        return entity_dict

    def read_crawled_entity(self, wiki_path):
        """
        Traverse storage files to get crawled entities
        :return:
        """
        crawled_entity_set = set()

        if os.path.exists(wiki_path):
            with open(wiki_path, "r", encoding="utf-8") as wiki_file:
                for item in wiki_file:
                    item = item.strip()

                    if len(item.split("\t")) != 2:
                        continue

                    entity_name, entity_str = item.split("\t")
                    crawled_entity_set.add(entity_name)

        print("crawled entity num: {0}".format(len(crawled_entity_set)))
        return crawled_entity_set

    def read_redirect_name(self, candidate_redirect_path):
        """
        read entity redirect name
        :param candidate_redirect_path:
        :return:
        """
        redirect_dict = {}
        with open(candidate_redirect_path, "r", encoding="utf-8") as candidate_redirect_file:
            for item in candidate_redirect_file:
                item = item.strip()

                source_name, redirect_name = item.split("\t")
                redirect_dict[source_name] = redirect_name

        return redirect_dict

    def build_seed_urls(self, candidate_path, candidate_redirect_path):
        """

        :param candidate_path:
        :param candidate_redirect_path:
        :return:
        """
        # read redirect dict
        redirect_dict = self.read_redirect_name(candidate_redirect_path)

        url_set = set()
        with open(candidate_path, "r", encoding="utf-8") as candidate_file:
            for item in candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                target_redirect_url = mention_obj["target_redirect_url"]
                url_set.add(target_redirect_url)

                candidate_dict = mention_obj["candidate"]

                for candidate_type, candidate_list in candidate_dict.items():
                    if candidate_type == "mention_keyword_search":
                        tmp_list = []
                        for ele_list in candidate_list:
                            tmp_list.extend(ele_list)
                        candidate_list = tmp_list

                    for candidate_name in candidate_list:
                        if candidate_name == "" or candidate_name.__contains__("disambiguation"):
                            continue

                        if candidate_name in redirect_dict:
                            candidate_name = redirect_dict[candidate_name]

                        url = "https://en.wikipedia.org/wiki/" + candidate_name
                        url_set.add(url)

                # for candidate_obj in candidate_dict:
                #     candidate_name = candidate_obj["name"]
                #     if candidate_name == "" or candidate_name.__contains__("disambiguation"):
                #         continue
                #     if candidate_name in redirect_dict:
                #         candidate_name = redirect_dict[candidate_name]
                #
                #     url = "https://en.wikipedia.org/wiki/" + candidate_name
                #     url_set.add(url)


        print("url set len: {0}".format(len(url_set)))
        return url_set

    def start_run(self, candidate_path, candidate_redirect_path, wiki_path):
        """
        start crawl wiki page
        :param candidate_path:
        :param candidate_redirect_path:
        :param wiki_path:
        :return:
        """
        # read crawled entities
        crawled_entity_set = self.read_crawled_entity(wiki_path)

        # the url in the candidate file
        url_set = self.build_seed_urls(candidate_path, candidate_redirect_path)

        with open(wiki_path, "a+", encoding="utf-8") as wiki_file:
            for url in url_set:
                entity_name = url.split("/")[-1].split("#")[-1]

                if entity_name not in crawled_entity_set:
                    entity_dict = self.crawl_single_page(url)
                    if len(entity_dict) != 0:
                        wiki_file.write(entity_name + "\t" + json.dumps(entity_dict, ensure_ascii=False) + "\n")
                        wiki_file.flush()
                        crawled_entity_set.add(entity_name)

    def start_golden_run(self, golden_redirect_path, golden_wiki_path):
        """
        临时添加的方法，为了爬取golden实体的描述信息
        :param golden_redirect_path:
        :param golden_wiki_path:
        :return:
        """
        # read crawled entities
        crawled_entity_set = self.read_crawled_entity(golden_wiki_path)

        all_url_set = set()
        with open(golden_redirect_path, "r", encoding="utf-8") as golden_redirect_file:
            for item in golden_redirect_file:
                item = item.strip()

                mention_obj = json.loads(item)

                target_redirect_url = mention_obj["target_redirect_url"]

                if target_redirect_url == "" or target_redirect_url.__contains__("disambiguation"):
                    continue

                entity_name = target_redirect_url.split("/")[-1].split("#")[-1]
                if entity_name not in crawled_entity_set:
                    all_url_set.add(target_redirect_url)

        count = 0
        with open(golden_wiki_path, "a+", encoding="utf-8") as wiki_file:
            for url in all_url_set:
                entity_name = url.split("/")[-1].split("#")[-1]

                entity_dict = self.crawl_single_page(url)
                if len(entity_dict) != 0:
                    wiki_file.write(entity_name + "\t" + json.dumps(entity_dict, ensure_ascii=False) + "\n")
                    wiki_file.flush()

                count += 1

                if count % 50 == 0:
                    print(count)

    def start_name_run(self, name_path, candidate_redirect_path, wiki_path):
        """
        only has candidate name in name_path
        :param name_path:
        :param candidate_redirect_path:
        :param wiki_path:
        :return:
        """
        # read crawled entities
        crawled_entity_set = self.read_crawled_entity(wiki_path)

        # read redirect dict
        redirect_dict = self.read_redirect_name(candidate_redirect_path)

        url_set = set()
        with open(name_path, "r", encoding="utf-8") as name_file:
            for item in name_file:
                name = item.strip()

                if name == "":
                    continue

                if name in redirect_dict:
                    name = redirect_dict[name]

                url = "https://en.wikipedia.org/wiki/" + name
                url_set.add(url)

        with open(wiki_path, "a+", encoding="utf-8") as wiki_file:
            for url in url_set:
                entity_name = url.split("/")[-1].split("#")[-1]

                if entity_name not in crawled_entity_set:
                    entity_dict = self.crawl_single_page(url)
                    if len(entity_dict) != 0:
                        wiki_file.write(entity_name + "\t" + json.dumps(entity_dict, ensure_ascii=False) + "\n")
                        wiki_file.flush()
                        crawled_entity_set.add(entity_name)


if __name__ == "__main__":

    # Before crawl wiki page, we should get the redirect url of the source url,
    # the reason is that we crawl the raw text of entity, not the html page.

    # data_list = ["ace2004", "msnbc", "aquaint", "clueweb", "wiki", "aida_train", "aida_testA", "aida_testB"]

    data_type = "wiki_clueweb"
    filter_candidate_path = "/root/fangzheng/data/filter_candidate/" + data_type + "_filter_candidate"
    candidate_redirect_path = "/root/fangzheng/data/candidate_redirect/" + data_type + "_candidate_redirect"
    wiki_entity_path = "/root/fangzheng/data/wiki/" + data_type + "_wiki"

    crawler = WikiEntityCrawler()
    crawler.start_run(filter_candidate_path, candidate_redirect_path, wiki_entity_path)

    # name_path = "/root/fangzheng/data/filter_candidate/" + data_type + "_filter_candidate_no_wiki"
    # filter_candidate_redirect_path = "/root/fangzheng/data/candidate_redirect/" + data_type + "_candidate_redirect"
    # wiki_entity_path = "/root/fangzheng/data/wiki/" + data_type + "_wiki"
    # crawler.start_name_run(name_path, filter_candidate_redirect_path, wiki_entity_path)

    # golden_redirect_path = "/root/fangzheng/data/golden_redirect/" + data_type + "_golden_redirect"
    # golden_wiki_path = "/root/fangzheng/data/golden_wiki/" + data_type + "_golden_wiki"
    # crawler.test_golden_run(golden_redirect_path, golden_wiki_path)


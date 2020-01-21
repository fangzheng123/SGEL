# encoding: utf-8

from lxml import etree
import json
import os
import requests
from urllib import parse
import config_util
import data_util

class CandidateCrawler(object):
    """
    Crawl candidate entity for each mention
    """
    def __init__(self):
        """

        :param:
        """
        self.data_util = data_util.DataUtil()

    def get_complete_matching_page(self, mention_url):
        """
        get entity whose name exactly matches mention form
        :param mention_url:
        :return:
        """
        candidate_list = []
        response = requests.get(url=mention_url)
        response.encoding = "utf-8"
        try:
            html = etree.HTML(response.text)
            candidate_name = html.xpath("/html/head/title/text()")[0].replace(" - Wikipedia", "").replace(" ", "_")
            error_tips = html.xpath("//*[@id='noarticletext']/tbody/tr/td/b/text()")

            if len(error_tips) == 0:
                candidate_list.append(candidate_name)

        except:
            print("get mention page error: " + mention_url)

        return candidate_list

    def get_disambiguate_page(self, disam_url):
        """

        :param disam_url:
        :return:
        """
        candidate_list = []
        response = requests.get(url=disam_url)
        response.encoding = "utf-8"
        tmp_candidate_list = []
        try:
            html = etree.HTML(response.text)

            main_data = html.xpath("//*[@id='mw-content-text']/div/p/b/a/@href")
            other_data = html.xpath("/html/body/div[3]/div[3]/div[4]/div/ul/li/a/@href")

            tmp_candidate_list.extend(main_data)
            tmp_candidate_list.extend(other_data)

            for candidate in tmp_candidate_list:
                if not candidate.startswith('/wiki/'):
                    continue
                candidate = candidate[candidate.rindex("/") + 1:]
                if candidate.endswith("_(disambiguation)"):
                    continue
                elif candidate.find('#') > 0:
                    candidate = candidate[:candidate.find('#')]

                candidate_list.append(candidate)
        except:
            print("get disambiguate page error: " + disam_url)

        return candidate_list

    def get_partial_matching_page(self, partial_matching_url):
        """

        :param search_url:
        :return:
        """
        candidate_list = []
        response = requests.get(url=partial_matching_url)
        response.encoding = "utf-8"

        try:
            result_list = json.loads(response.text)
            candidate_url_list = result_list[-1]
            for candidate in candidate_url_list:
                candidate = candidate[candidate.rindex("/") + 1:]
                if candidate.endswith("_(disambiguation)"):
                    continue

                candidate_list.append(candidate)
        except:
            print("get partial matching error: " + partial_matching_url)

        return candidate_list

    def get_search_page(self, recommend_url):
        """

        :param recommend_url:
        :return:
        """
        candidate_list = []
        response = requests.get(url=recommend_url)
        response.encoding = "utf-8"
        try:
            html = etree.HTML(response.text)
            recommend_list = html.xpath("//*[@id='mw-content-text']/div/ul/li/div[1]/a/@href")

            for candidate in recommend_list:
                if not candidate.startswith('/wiki/'):
                    continue
                candidate = candidate[candidate.rindex("/") + 1:]
                if candidate.endswith("_(disambiguation)"):
                    continue
                elif candidate.find('#') > 0:
                    candidate = candidate[:candidate.find('#')]

                candidate_list.append(candidate)

        except:
            print("get search page error: " + recommend_url)

        return candidate_list

    def get_google_result(self, mention_name, Google_Custom_Search_Engine_ID, Google_Clould_Plarform_project_key):
        """
        Use google custom search engine API searching related wiki uris of unknow mentions

        :param mention_name: mention name to be searched.
        :param Google_Custom_Search_Engine_ID: ID of Google Custom Search engine, filter rules(eg. only wikipedia pages) for search results should be configured in the engine.
        :param Google_Clould_Plarform_project_key: An search engine must be bounded with a Google Cloud Platform project.
        :return: list of wiki uris searched out.
        """
        # restrict the search result to wikipedia pages
        search_content = mention_name + " wikipedia"
        # construct request url
        request_url = "https://www.googleapis.com/customsearch/v1?q=" + search_content + "&cx="+Google_Custom_Search_Engine_ID + "&key=" + Google_Clould_Plarform_project_key

        links = []
        try:
            rp = requests.get(request_url)
            rp_str = rp.text
            rp_json = json.loads(rp_str)
            js_items = rp_json.get("items")
            if js_items is None:
                pass
            else:
                for i in js_items:
                    links.append(i.get("link"))
        except:
            print("get google result error: " + mention_name)

        return links

    def make_mention_url(self, mention_name):
        """

        :param mention_name:
        :return:
        """
        mention_name = mention_name.replace(" ", "_")
        mention_name = parse.quote(mention_name)
        return "https://en.wikipedia.org/wiki/" + mention_name

    def make_disam_url(self, mention_name):
        """

        :param mention_name:
        :return:
        """
        mention_name = mention_name.replace(" ", "_")
        mention_name = parse.quote(mention_name)
        return "https://en.wikipedia.org/wiki/" + mention_name + "_(disambiguation)"

    def make_partial_matching_url(self, mention_name, result_num):
        """

        :param mention_name:
        :return:
        """
        mention_name = parse.quote(mention_name)
        return "https://en.wikipedia.org/w/api.php?action=opensearch&search="\
               + mention_name + "&limit=" + str(result_num) + "&namespace=0&format=json"

    def make_search_url(self, mention_name, result_num):
        """

        :param mention_name:
        :return:
        """
        mention_name = parse.quote(mention_name)

        return "https://en.wikipedia.org/w/index.php?search=" + mention_name \
               + "&limit=" + str(result_num) + "&title=Special%3ASearch&profile=advanced&fulltext=1&advancedSearch-current=%7B%7D&ns0=1"

    def read_crawled_candidate(self, common_candidate_path):
        """
        read crawled mention disambiguate
        :return:
        """
        crawled_mention_set = set()

        if os.path.exists(common_candidate_path):
            with open(common_candidate_path, "r", encoding="utf-8") as candidate_file:
                for item in candidate_file:
                    item = item.strip()

                    mention = item.split("\t")[0]
                    crawled_mention_set.add(mention)

        return crawled_mention_set

    def start_common_run(self, source_path, common_candidate_path, data_type, crawl_num):
        """
        crawl common candidate
        :param source_path:
        :param common_candidate_path:
        :param data_type:
        :param crawl_num:
        :return:
        """
        mention_set = self.read_crawled_candidate(common_candidate_path)
        print("mention num: {0}".format(len(mention_set)))

        with open(source_path, "r", encoding="utf-8") as source_file:
            with open(common_candidate_path, "a+", encoding="utf-8") as common_candidate_file:
                for item in source_file:
                    item = item.strip()
                    mention_json = json.loads(item)

                    mention_name = mention_json["mention_form"]

                    mention_synonym_list = []
                    if "synonym_mention" in mention_json:
                        mention_synonym_list = mention_json["synonym_mention"]

                    capitalize_mention_name = self.data_util.capitalize_mention(mention_name)

                    use_google_number = 0
                    google_key_index = 0
                    key_list = config_util.Google_Cloud_Platform_keys[::-1]
                    google_key = key_list[google_key_index]
                    mention_candidate_dict = {}
                    if capitalize_mention_name not in mention_set:
                        # 1. get entity whose name exactly matches mention form
                        mention_url = self.make_mention_url(capitalize_mention_name)
                        mention_candidate_dict["mention_page"] = self.get_complete_matching_page(mention_url)

                        # 2. get mention disambiguate page
                        mention_disam_url = self.make_disam_url(capitalize_mention_name)
                        mention_candidate_dict["mention_disam_page"] = self.get_disambiguate_page(mention_disam_url)

                        # 3. get partial matching entity
                        mention_partial_matching_url = self.make_partial_matching_url(capitalize_mention_name, crawl_num)
                        mention_candidate_dict["mention_partial_matching_page"] = self.get_partial_matching_page(mention_partial_matching_url)

                        # 4. get wikipedia search page
                        mention_search_url = self.make_search_url(capitalize_mention_name, crawl_num)
                        mention_candidate_dict["mention_search_page"] = self.get_search_page(mention_search_url)

                        # 5. if word number > 2, then use google search engine
                        if len(capitalize_mention_name.split(" ")) > 2 and (data_type in config_util.Use_Goole_Data_Names):
                            # get google search result
                            url_list = self.get_google_result(capitalize_mention_name, config_util.Google_Search_Engine_ID, google_key)
                            mention_candidate_dict["google_search_result"] = [url.split("/")[-1] for url in url_list]
                            use_google_number += 1

                        # 6. get wordnet search entity
                        word_candidate_list = []
                        for word in mention_synonym_list[:3]:
                            if word != mention_name:
                                mention_partial_matching_url = self.make_partial_matching_url(word, 10)
                                word_candidate_list.extend(self.get_partial_matching_page(
                                    mention_partial_matching_url))
                        mention_candidate_dict["mention_wordnet"] = word_candidate_list

                        # 7. if candidate number equals 0, then use google search engine
                        candidate_num = sum([len(candidate_list) for key, candidate_list in mention_candidate_dict.items()])
                        if candidate_num == 0 and "google_search_result" not in mention_candidate_dict:
                            url_list = self.get_google_result(capitalize_mention_name, config_util.Google_Search_Engine_ID, google_key)
                            mention_candidate_dict["google_search_result"] = [url.split("/")[-1] for url in url_list]

                            use_google_number += 1

                        if len(mention_set) % 100 == 0:
                            print("mention set num: " + str(len(mention_set)))

                        if use_google_number > 0 and use_google_number % 100 == 0:
                            google_key_index += 1
                            if google_key_index < len(config_util.Google_Cloud_Platform_keys):
                                google_key = config_util.Google_Cloud_Platform_keys[google_key_index]

                        mention_set.add(capitalize_mention_name)
                        common_candidate_file.write(
                            capitalize_mention_name + "\t" + json.dumps(mention_candidate_dict, ensure_ascii=False) + "\n")
                        common_candidate_file.flush()

                    if use_google_number % 10 == 1:
                        print(use_google_number)

    def start_keyword_run(self, source_path, keyword_candidate_path, crawl_num):
        """

        :param source_path:
        :param keyword_candidate_path:
        :param crawl_num:
        :return:
        """
        with open(source_path, "r", encoding="utf-8") as source_file:
            with open(keyword_candidate_path, "w", encoding="utf-8") as keyword_candidate_file:

                for item in source_file:
                    item = item.strip()
                    mention_obj = json.loads(item)

                    mention_name = mention_obj["mention_form"]
                    mention_keyword_list = mention_obj["context_keyword"]

                    capitalize_mention_name = self.data_util.capitalize_mention(mention_name)

                    # 1. mention + context keyword
                    keyword_search_result = []
                    if len(mention_keyword_list) > 0:
                        if len(mention_keyword_list) > 1 and len(capitalize_mention_name.strip().split(" ")) == 1:
                            capitalize_mention_name = capitalize_mention_name + " " + capitalize_mention_name
                        else:
                            capitalize_mention_name = capitalize_mention_name
                        for mention_keyword in mention_keyword_list:
                            if len(mention_keyword.split(" ")) > 1:
                                mention_keyword = ""

                            mention_search_url = self.make_search_url(capitalize_mention_name + " " + mention_keyword, crawl_num)
                            keyword_search_result.append(self.get_search_page(mention_search_url))

                    mention_obj["mention_keyword_search"] = keyword_search_result

                    keyword_candidate_file.write(json.dumps(mention_obj) + "\n")
                    keyword_candidate_file.flush()

    def combine_candidate(self, keyword_candidate_path, common_candidate_path, candidate_path):
        """
        combine keyword and common candidate
        :param keyword_candidate_path:
        :param common_candidate_path:
        :param candidate_path:
        :return:
        """
        mention_candidate_dict = {}
        with open(common_candidate_path, "r", encoding="utf-8") as common_candidate_file:
            for item in common_candidate_file:
                mention, candidate_str = item.strip().split("\t")
                candidate_obj = json.loads(candidate_str)
                mention_candidate_dict[mention] = candidate_obj

        mention_list = []
        with open(keyword_candidate_path, "r", encoding="utf-8") as keyword_candidate_file:
            for item in keyword_candidate_file:
                item = item.strip()

                mention_obj = json.loads(item)
                mention_form = mention_obj["mention_form"]
                capitalize_mention = self.data_util.capitalize_mention(mention_form)

                if capitalize_mention in mention_candidate_dict:
                    mention_obj["candidate"] = mention_candidate_dict[capitalize_mention]
                    # if "mention_keyword_search" in mention_obj:
                    #     mention_obj["candidate"]["mention_keyword_search"] = mention_obj["mention_keyword_search"]
                else:
                    print("no candidate: " + mention_form)

                mention_list.append(mention_obj)

        with open(candidate_path, "w", encoding="utf-8") as candidate_file:
            for mention_obj in mention_list:
                candidate_file.write(json.dumps(mention_obj) + "\n")

    def update_google_candidate(self, common_candidate_path):
        """
        update part mention's google candidate
        :return:
        """
        mention_candidate_dict = {}
        with open(common_candidate_path, "r", encoding="utf-8") as common_candidate_file:
            for item in common_candidate_file:
                item = item.strip()

                mention, candidate_str = item.split("\t")
                candidate_obj = json.loads(candidate_str)
                if (("google_search_result" not in candidate_obj)
                    or ("google_search_result" in candidate_obj and len(candidate_obj["google_search_result"]) == 0)) \
                        and len(mention.split(" ")) > 2:
                    # get google search result
                    url_list = self.get_google_result(mention, config_util.Google_Search_Engine_ID,
                                                      config_util.Google_Cloud_Platform_key3)
                    candidate_obj["google_search_result"] = [url.split("/")[-1] for url in url_list]

                mention_candidate_dict[mention] = candidate_obj

        with open(common_candidate_path, "w", encoding="utf-8") as common_candidate_file:
            for mention, candidate_obj in mention_candidate_dict.items():
                common_candidate_file.write(
                    mention + "\t" + json.dumps(candidate_obj, ensure_ascii=False) + "\n")
                common_candidate_file.flush()

    def start_run(self):
        data_type = "ace2004"
        source_path = "/root/fangzheng/data/source/" + data_type + "_in_wiki"
        common_candidate_path = "/root/fangzheng/data/candidate/" + data_type + "_common_candidate"
        keyword_candidate_path = "/root/fangzheng/data/candidate/" + data_type + "_keyword_candidate"
        candidate_path = "/root/fangzheng/data/candidate/" + data_type + "_candidate"

        # candidate_crawl.start_common_run(source_path, common_candidate_path, data_type, 20)
        candidate_crawl.start_keyword_run(source_path, keyword_candidate_path, 10)
        # candidate_crawl.combine_candidate(keyword_candidate_path, common_candidate_path, candidate_path)

        # candidate_crawl.update_google_candidate(common_candidate_path)
        # candidate_crawl.combine_candidate(keyword_candidate_path, common_candidate_path, candidate_path)

if __name__ == "__main__":
    candidate_crawl = CandidateCrawler()
    candidate_crawl.start_run()



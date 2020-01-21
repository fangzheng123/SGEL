import os
import json
import sys
sys.path.append("./")
from crawler.redirect_crawler import GoldenCrawler

class CheckSummary():
    def __init__(self):
        pass
    def check_summary_whether_empty(self, file_path):
        '''check whether the mention summaries in the file are empty'''
        empty_num = 0
        with open(file_path, "r") as file_to_check:
            for line in file_to_check:
                mention_dict = json.loads(line)
                if mention_dict["summary"] == "":
                    empty_num = empty_num + 1
        
        return empty_num

if __name__== "__main__":
    # # count the num of entities with empty summary
    # dir_path = "/home1/fangzheng/data/bert_el_data/source_data/total_parts"
    # file_list = os.listdir(dir_path)
    # ck = CheckSummary()
    # total_num = 0
    # for f in file_list:
    #     file_path = os.path.join(dir_path, f)
    #     num = ck.check_summary_whether_empty(file_path)
    #     print(f, ": ", num)
    #     total_num = total_num + num
    # print(total_num)


    f = open("/root/fangzheng/only_name", "r")
    gc = GoldenCrawler()
    redirect_pairs = []
    no_redirect_num = 0
    redirect_pairs_file = open("/root/fangzheng/redirect_dict", "w")

    for (i, line) in zip(range(60000), f):
        print('index:', i, line[:-1], end='')
        target_url = "https://en.wikipedia.org/wiki/" + line[:-1]
        url = gc.crawl_singe_page(target_url)
        redirect_name = url.split("/")[-1]
        # print("source: ", line[:-1], ", redirect: ", redirect_name, sep="")
        if line[:-1] == redirect_name or redirect_name == "":
            no_redirect_num = no_redirect_num + 1
            print(" NO_redirect")
        else:
            print(" "+redirect_name)
            pair = line[:-1] + "\t" + redirect_name
            redirect_pairs_file.write(pair+"\n")
            redirect_pairs_file.flush()

        if i % 500 == 0:
            redirect_pairs_file.close()
            redirect_pairs_file = open("/root/fangzheng/redirect_dict", "a")

    redirect_pairs_file.close()
    print(no_redirect_num)


 

    gc = GoldenCrawler()
    url = gc.crawl_singe_page(target_url)
    print(url)


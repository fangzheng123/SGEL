# encoding


import bz2file
import os
import pprint
import json

class AnalyseWikidata(object):
    """
    analyse 35GB .bz2 Wikidata(not wikipedia), download from https://dumps.wikimedia.org/wikidatawiki/entities/20190520/
    """

    def __init__(self, wikidata_path, wikiname_path):
        """

        :param wikidata_path:
        :param wikiname_path: all entity name in wikidata
        """
        self.wikidata_path = wikidata_path
        self.wikiname_path = wikiname_path


    def read_bz2(self):
        """
        direct read bz2 file, reference by https://akbaritabar.netlify.com/how_to_use_a_wikidata_dump
        :return:
        """
        with bz2file.open(self.wikidata_path, "r") as wikidata_file:
            with open(self.wikiname_path, "a+") as wikiname_file:
                # skip first two bytes: "{\n"
                wikidata_file.read(2)

                index = 0
                for line in wikidata_file:
                    try:
                        entity = json.loads(line.rstrip(',\n'))
                        if entity["type"] == "item":
                            if "sitelinks" in entity:
                                site_links = entity["sitelinks"]
                                if "enwiki" in site_links:
                                    wiki_name = site_links["enwiki"]["title"]
                                    wikiname_file.write(wiki_name.encode("utf-8") + "\n")
                                    wikiname_file.flush()

                    except json.decoder.JSONDecodeError:
                        continue

                    # if index > 10:
                    #     break

                    index += 1

                    if index % 100000 == 0:
                        print(index)




if __name__ == "__main__":
    analyse_wiki = AnalyseWikidata("/home/xmxie/caoyananGroup/fangzheng/wikidata/wikidata-20190520-all.json.bz2", "/home/xmxie/caoyananGroup/fangzheng/wikidata/entity_name")
    analyse_wiki.read_bz2()


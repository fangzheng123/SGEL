from data_util import DataUtil
import os
import json
import sys

class GatheringEntities(object):
    def __init__(self):
        pass

    def write_entities_into_one_file(self, directory, write_to_path):
        '''
        gather the entites files in one directory into one single file
        '''
        # get all the files name
        entity_file_list = os.listdir(directory)
        with open(write_to_path, "a+", encoding="utf-8") as write_to_file:
            for i, entity_file in enumerate(entity_file_list):           
                with open(os.path.join(directory, entity_file),  encoding="utf-8") as f:
                    entity_object = json.load(f)
                    entity_json_str = json.dumps(entity_object)
                    write_to_file.write(entity_json_str+"\n")
                if i % 100 == 0:
                    write_to_file.flush()
                    print(directory, ": ", i)
    
    def split_file(self, file_path, write_to_path, split_num):
        '''
        split the single file into several subfiles averagely. The subfiles will be named "part1/2/3".
        '''
        
        file_list = []
        for i in range(split_num):
            write_to_file = write_to_path + "/part" + str(i)
            file_list.append(open(write_to_file, "a", encoding="utf-8"))

        file_len = 0
        with open(file_path, "r", encoding="utf-8") as file_to_be_splitted:
            for line in file_to_be_splitted:
                file_len = file_len + 1

        with open(file_path, "r", encoding="utf-8") as file_to_be_splitted:
            len_num_per_file = int(file_len / split_num)
            for i, line in enumerate(file_to_be_splitted):
                file_index = int(i / len_num_per_file)
                if file_index < split_num:
                    file_to_write = file_list[file_index]
                else:
                    # beyond the index
                    file_to_write = file_list[-1]
                file_to_write.write(line)
                file_to_write.flush()
                if i % 100 == 0:
                    print(file_path, ": ", i)
        
        for file_to_write in file_list:
            file_to_write.close()

if __name__ == "__main__":

    # sc_path = "/home1/fangzheng/data/bert_el_data/source_data/source_part"
    # d_util = DataUtil()
    # dir_sets = d_util.get_source_part_dir(sc_path)
    # ge = GatheringEntities()

    # write_to_pathes = ["/home1/fangzheng/data/bert_el_data/source_data/all_source_data", "/home1/fangzheng/data/bert_el_data/source_data/all_source_data"]
    # for (dir_path, write_to_path) in zip(dir_sets.keys(), write_to_pathes):
    #     ge.write_entities_into_one_file(dir_path, write_to_path)
    
    # # count the memory requirement of all the data
    # dict_list = {}
    # with open("/home1/fangzheng/data/bert_el_data/source_data/all_source_data", "r") as f:
    #     for i, line in enumerate(f):
    #         json_dict = json.loads(line)
    #         dict_list[json_dict["name"]] = json_dict
    #         if i % 1000 == 0:
    #             print("index:", i, "memory usage(MB):", sys.getsizeof(dict_list)/1024)

    # # split the whole file into several subfiles
    # file_path = "/home1/fangzheng/data/bert_el_data/source_data/all_source_data"
    # write_to_path = "/home1/fangzheng/data/bert_el_data/source_data/total_parts2"

    # ge = GatheringEntities()
    # ge.split_file(file_path, write_to_path, 8)

    print(1)
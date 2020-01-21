# encoding: utf-8


class Test(object):

    def xgb_acc(self, bert_path):
        """

        :param bert_path:
        :return:
        """
        xgb_recall_count = 0
        group_set = set()
        with open(bert_path, "r", encoding="utf-8") as bert_file:
             for item in bert_file:
                 item = item.strip()

                 group_id_str, label_str, fea_str, mention_str, entity_str = item.split("\t")
                 group_id = int(group_id_str)
                 label = int(label_str)

                 if group_id not in group_set:
                     group_set.add(group_id)

                     if label == 1:
                         xgb_recall_count += 1

        print(xgb_recall_count, len(group_set), xgb_recall_count/len(group_set))


if __name__ == "__main__":

    test = Test()

    validate_name = "reuters128"
    source_dir = "/data/fangzheng/bert_el/"
    combine_val_path = source_dir + validate_name + "/bert/" + validate_name + "_cut_rank_format_bert_predict"
    test.xgb_acc(combine_val_path)





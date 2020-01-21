# encoding:utf-8


class DataUtil(object):

    def capitalize_mention(self, mention_form):
        """
        capitalize mention form
        :param mention_form:
        :return:
        """
        word_list = []
        for word in mention_form.strip().split(" "):
            word_list.append(word.capitalize())
        capitalize_mention_form = " ".join(word_list)

        return capitalize_mention_form
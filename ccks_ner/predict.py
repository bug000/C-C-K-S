import json
from typing import List

import tensorflow as tf

from kashgari.tasks.seq_labeling import BLSTMCRFModel
from tqdm import tqdm


class Predicter(object):

    @staticmethod
    def get_kb_dic():
        subject_dict = {}
        kb_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data"
        kr_reader = open(kb_path, 'r', encoding='utf-8')
        for line in kr_reader:
            # 小写
            knobj = json.loads(line.lower())

            """subject"""
            subject = knobj["subject"]
            if subject in subject_dict.keys():
                subject_dict[subject].append(knobj)
            else:
                subject_dict[subject] = [knobj]

            """alias"""
            for sub_alias in knobj["alias"]:
                if sub_alias in subject_dict.keys():
                    subject_dict[sub_alias].append(knobj)
                else:
                    subject_dict[sub_alias] = [knobj]
        return subject_dict

    def predict(self, json_lines: List) ->List:
        raise NotImplementedError

    def predict_devs(self, dev_path, result_path):
        result_writer = open(result_path, 'w', encoding="utf-8")
        json_lines = [json.loads(line) for line in open(dev_path, "r", encoding="utf-8").readlines()]

        pre_line_s = self.predict(json_lines)
        for pre_line in pre_line_s:
            result_writer.write(json.dumps(pre_line, ensure_ascii=False) + "\n")
            result_writer.flush()
        result_writer.close()


class Discriminater(Predicter):
    pass


class BiLSTMCRFPredicter(Predicter):

    def __init__(self, model_path):
        with tf.device('/gpu:0'):
            # mention -> entity_json_line
            self.subject_dic = super().get_kb_dic()

            self._model = BLSTMCRFModel.load_model(model_path)

    def token_stream(self, line: str, ngram: int):
        re_list = []
        line_list = list(line)
        # print line
        # print line_list
        for char_index in range(len(line_list) - ngram + 1):
            word = ''
            for cha_itea in range(ngram):
                word += line_list[char_index + cha_itea]
            re_list.append(word.strip())
        return re_list

    def extract_entity_text(self, entity_json_line: dict):
        """
        得到 entity 描述文本的 ngram list
        :param entity_json_line:
        :return:
        """
        all_tokens = []
        ngram = 2
        all_str = []
        all_str += entity_json_line["alias"]
        datas = entity_json_line["data"]
        for data in datas:
            all_str += data.values()
        for entity_str in all_str:
            tokens = self.token_stream(entity_str, ngram)
            all_tokens += tokens
        return list(filter(lambda token: token != "", all_tokens))

    def compute_zh_cos_sim(self, text_a: list, text_b: list):
        """
        两段未加权文本 cos
        :param text_a:
        :param text_b:
        :return:
        """
        text_a = set(text_a)
        text_b = set(text_b)
        text_a_len = len(text_a)
        text_b_len = len(text_b)
        intersection_len = len(text_a.intersection(text_b))

        cos = intersection_len/(text_a_len*text_b_len)
        return cos

    def get_subject_id(self, mention_text: str, pre_text: str):
        """
        先查找到全部的 mention_text 对应的实体
        每一个实体与 pre_text 比较 2 gram 相似度
        取相似度最高实体 返回 id
        :param mention_text:
        :param pre_text:
        :return:
        """
        entity_json_line_s = self.subject_dic.get(mention_text, [])

        cos_temp = -1.0
        subject_id_temp = "NIL"

        for entity_json_line in entity_json_line_s:
            entity_message_text = self.extract_entity_text(entity_json_line)
            cos_sim = self.compute_zh_cos_sim(entity_message_text, self.token_stream(pre_text, 2))
            subject_id = entity_json_line["subject_id"]
            if cos_sim > cos_temp:
                cos_temp = cos_sim
                subject_id_temp = subject_id
        return subject_id_temp

    def predict(self, json_lines):
        BEGIN = "B"
        MIDD = "M"
        END = "E"
        SINGLE = "S"
        NONE = "O"

        pre_lines = []

        pre_texts = [list(json_line["text"]) for json_line in json_lines]

        pre_seq_s = self.predict_text(pre_texts)

        for data_ind, pre_seq in enumerate(tqdm(pre_seq_s)):
            dev_line = json_lines[data_ind]
            dev_text = pre_texts[data_ind]

            mention_data = []
            for token_id, entity_mark in enumerate(pre_seq):
                # 序列开始标志
                if entity_mark.startswith(BEGIN) or entity_mark.startswith(SINGLE):

                    mention = {
                        "kb_id": "",
                        "mention": "",
                        "offset": str(token_id)
                    }

                    # 这种方法提取实体可能有重复现象 ?
                    for token_id_i in range(token_id, len(pre_seq)):
                        if pre_seq[token_id_i] is NONE:
                            subject_id_m = self.get_subject_id(mention["mention"], "".join(dev_text))
                            mention["kb_id"] = str(subject_id_m)
                            mention_data.append(mention)
                            break
                        mention["mention"] += dev_text[token_id_i]

                        # 已经等于最后一个索引了，说明直到最后一个 char 都是 entity str
                        if token_id_i == (len(pre_seq) - 1):
                            subject_id_m = self.get_subject_id(mention["mention"], "".join(dev_text))
                            mention["kb_id"] = str(subject_id_m)
                            mention_data.append(mention)

            dev_line["mention_data"] = mention_data
            pre_lines.append(dev_line)

        return pre_lines

    def predict_text(self, x_pre):
        return self._model.predict(x_pre, batch_size=1024)


def main():
    dev_path = "D:/data/biendata/ccks2019_el/ccks_train_data/validate.json"
    model_path = r"D:\data\biendata\ccks2019_el\ner_model"
    crfer = BiLSTMCRFPredicter(model_path)
    crfer.predict_devs(dev_path, dev_path+".pre.txt")


if __name__ == '__main__':
    main()



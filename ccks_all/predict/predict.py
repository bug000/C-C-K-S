import json
from typing import List

import tensorflow as tf

from kashgari.tasks.seq_labeling import BLSTMCRFModel
from tqdm import tqdm

from ccks_all.static import subject_id_dict, id2entity


class Predicter(object):

    def predict(self, json_lines: List) -> List:
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
    def predict(self, json_lines: List) -> List:
        raise NotImplementedError


class BiLSTMCRFPredicter(Predicter):

    def __init__(self, model_path, type_filter=False):
        # mention -> entity_json_line
        self.subject_id_dict = subject_id_dict
        self._model = BLSTMCRFModel.load_model(model_path)
        self.type_filter = type_filter

    def get_subject_ids(self, mention_text: str, type_predict: str):
        entis_ids = self.subject_id_dict.get(mention_text.strip(), [])
        for entit_id in entis_ids:
            entit = id2entity[entit_id]
            etype = str(entit["type"][0]).lower()
            if self.type_filter:
                if etype == type_predict:
                    yield entit["subject_id"]
            else:
                yield entit["subject_id"]

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
            if "mention_data" in dev_line.keys():
                key_mention_ids = [m["kb_id"] for m in dev_line["mention_data"]]
            else:
                key_mention_ids = []
            mention_data = []
            for token_id, entity_mark in enumerate(pre_seq):
                # 序列开始标志
                if entity_mark.startswith(BEGIN) or entity_mark.startswith(SINGLE):

                    type_predict = entity_mark.lower().split("-")[1]

                    mention = {
                        "kb_id": "",
                        "mention": "",
                        "offset": str(token_id),
                        "label": "-1"
                    }

                    # 这种方法提取实体可能有重复现象 ?
                    for token_id_i in range(token_id, len(pre_seq)):
                        if pre_seq[token_id_i] is NONE \
                                or (pre_seq[token_id_i].startswith(BEGIN) and token_id_i != token_id)\
                                or (pre_seq[token_id_i].startswith(SINGLE) and token_id_i != token_id):
                            subject_id_s = self.get_subject_ids(mention["mention"], type_predict)
                            for subject_id in subject_id_s:
                                mention["kb_id"] = str(subject_id)

                                if subject_id in key_mention_ids:
                                    mention["label"] = "1"
                                else:
                                    mention["label"] = "0"
                                mention_data.append(mention.copy())

                            break
                        mention["mention"] += dev_text[token_id_i]

                        # 已经等于最后一个索引了，说明直到最后一个 char 都是 entity str
                        if token_id_i == (len(pre_seq) - 1):
                            subject_id_s = self.get_subject_ids(mention["mention"], type_predict)
                            for subject_id in subject_id_s:
                                mention["kb_id"] = str(subject_id)
                                if subject_id in key_mention_ids:
                                    mention["label"] = "1"
                                else:
                                    mention["label"] = "0"
                                mention_data.append(mention.copy())

            dev_line["mention_data"] = mention_data
            pre_lines.append(dev_line)

        return pre_lines

    def predict_text(self, x_pre):
        return self._model.predict(x_pre, batch_size=1024)


class BiLSTMCRFntPredicter(Predicter):

    def __init__(self, model_path):
        # mention -> entity_json_line
        self.subject_id_dict = subject_id_dict
        self._model = BLSTMCRFModel.load_model(model_path)

    def get_subject_ids(self, mention_text: str):
        entisids = self.subject_id_dict.get(mention_text.strip(), [])
        for entit_id in entisids:
            entit = id2entity[entit_id]
            yield entit["subject_id"]

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
                    # 这种方法提取实体可能有重复现象 ?
                    mention = {
                        "kb_id": "",
                        "mention": "",
                        "offset": str(token_id),
                    }
                    for token_id_i in range(token_id, len(pre_seq)):
                        # 已经等于最后一个索引了，说明直到最后一个 char 都是 entity str
                        if pre_seq[token_id_i] is NONE \
                                or (pre_seq[token_id_i].startswith(BEGIN) and token_id_i != token_id) \
                                or (pre_seq[token_id_i].startswith(SINGLE) and token_id_i != token_id):
                            subject_id_s = self.get_subject_ids(mention["mention"])
                            for subject_id in subject_id_s:
                                mention["kb_id"] = str(subject_id)
                                mention_data.append(mention.copy())
                            break
                        mention["mention"] += dev_text[token_id_i]

                        if token_id_i == (len(pre_seq) - 1):
                            subject_id_s = self.get_subject_ids(mention["mention"])
                            for subject_id in subject_id_s:
                                mention["kb_id"] = str(subject_id)
                                mention_data.append(mention.copy())

                            # if token_id_i == (len(pre_seq) - 1):
                        #     subject_id_s = self.get_subject_ids(mention["mention"])
                        #     for subject_id in subject_id_s:
                        #         mention["kb_id"] = str(subject_id)
                        #         mention_data.append(mention)

            dev_line["mention_data"] = mention_data
            pre_lines.append(dev_line)

        return pre_lines

    def predict_text(self, x_pre):
        return self._model.predict(x_pre, batch_size=1024)


def main():
    dev_path = "D:/data/biendata/ccks2019_el/ccks_train_data/validate.json"
    model_path = r"D:\data\biendata\ccks2019_el\ner_model"
    # crfer = BiLSTMCRFPredicter(model_path)
    # crfer.predict_devs(dev_path, dev_path+".pre.txt")


if __name__ == '__main__':
    main()

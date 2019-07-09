import json
import re
from typing import List

import tensorflow as tf

from kashgari.tasks.seq_labeling import BLSTMCRFModel
from tqdm import tqdm

from ccks_all.esret import expand_sim_es
from ccks_all.static import subject_id_dict, id2entity


class Predicter(object):

    @staticmethod
    def explan_mention_text(fd_str_pair_s: List):
        for fd_str_pair in fd_str_pair_s:
            fd_str, ind = fd_str_pair
            # yield fd_str, ind, fd_str
            expand_strs = expand_sim_es(keywords=fd_str, jaccard_filt=0.6)
            for exp_str in expand_strs:
                yield exp_str, ind, fd_str

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

    def __init__(self, model_path, type_filter=False, save_label=False, batch=32, save_expand_subject=True):
        # mention -> entity_json_line
        self.subject_id_dict = subject_id_dict
        self._model = BLSTMCRFModel.load_model(model_path)
        # self._model = DDDDModel.load_model(model_path)
        self.type_filter = type_filter
        self.batch = batch
        self.save_label = save_label
        self.save_expand_subject = save_expand_subject

    def get_subject_ids(self, mention_text: str, type_predict: str = "None"):
        entis_ids = self.subject_id_dict.get(mention_text.strip(), [])
        for entit_id in entis_ids:
            entit = id2entity[entit_id]
            etype = str(entit["type"][0]).lower()
            if self.type_filter:
                if etype == type_predict:
                    yield entit["subject_id"]
            else:
                yield entit["subject_id"]

    @staticmethod
    def dsc_token(text: str):
        fds = re.findall(r"《([^《|》]*)》", text)
        for fd_str in fds:
            fd_str = fd_str.strip()
            ind = text.find(fd_str)
            yield fd_str, ind

    @staticmethod
    def eng_token(text: str):
        fds = re.findall(r"\b[a-z\d ]{1,100}\b", text)
        for fd_str in fds:
            fd_str = fd_str.strip()
            ind = text.find(fd_str)
            yield fd_str, ind

    @staticmethod
    def is_contain_this_subject_id(mention_data: List, subject_id: str):
        """判断是否已经找到这个实体"""
        mention_data_kb_id = [m["kb_id"] for m in mention_data]
        return subject_id in mention_data_kb_id

    def add_mention_data(self, mention_data: List, str_pairs: List, key_mention_ids=None):
        """增加一组实体字符串"""
        if key_mention_ids is None:
            key_mention_ids = []
        for fd_str_pair in str_pairs:
            exp_str, ind, fd_str = fd_str_pair

            subject_id_s = self.get_subject_ids(exp_str)
            for subject_id in subject_id_s:
                if not self.is_contain_this_subject_id(mention_data, str(subject_id)):
                    mention = {
                        "kb_id": str(subject_id),
                        "mention": fd_str,
                        "offset": str(ind)
                    }

                    if self.save_expand_subject:
                        mention["expand"] = exp_str

                    if self.save_label:
                        if subject_id in key_mention_ids:
                            mention["label"] = "1"
                        else:
                            mention["label"] = "0"

                    mention_data.append(mention.copy())

    """
    实体的文本和库中的文本不完全匹配:
    库中有这段文本，但不匹配
    库中有这段文本，label标注不对    
    
    
    //todo
    得到文本
    扩展文本
    得到id
    
    修改为先扩展 再匹配id
    
    
    """
    def gener_ner_pre_pairs(self, dev_text, pre_seq):
        BEGIN = "B"
        SINGLE = "S"
        NONE = "O"
        for token_id, entity_mark in enumerate(pre_seq):
            if entity_mark.startswith(BEGIN) or entity_mark.startswith(SINGLE):
                mention_text = ""
                for token_id_i in range(token_id, len(pre_seq)):
                    if pre_seq[token_id_i] is NONE \
                            or (pre_seq[token_id_i].startswith(BEGIN) and token_id_i != token_id) \
                            or (pre_seq[token_id_i].startswith(SINGLE) and token_id_i != token_id):
                        yield mention_text, token_id
                        break
                    # todo break 存疑  continue
                    mention_text += dev_text[token_id_i]
                    # 已经等于最后一个索引了，说明直到最后一个 char 都是 entity str
                    if token_id_i == (len(pre_seq) - 1):
                        yield mention_text, token_id

    def ana_line_pre(self, mention_data, dev_text, pre_seq, key_mention_ids):
        ner_pairs = list(self.gener_ner_pre_pairs(dev_text, pre_seq))
        expand_ner_pairs = list(self.explan_mention_text(fd_str_pair_s=ner_pairs.copy()))
        reg_pairs = list(self.dsc_token("".join(dev_text)))
        expand_reg_pairs = list(self.explan_mention_text(fd_str_pair_s=reg_pairs.copy()))

        self.add_mention_data(mention_data, str_pairs=expand_ner_pairs, key_mention_ids=key_mention_ids)
        self.add_mention_data(mention_data, str_pairs=expand_reg_pairs, key_mention_ids=key_mention_ids)
        return mention_data

    def predict(self, json_lines):
        pre_lines = []
        pre_texts = [list(json_line["text"]) for json_line in json_lines]

        pre_seq_s = self.predict_text(pre_texts)

        for data_ind, pre_seq in enumerate(tqdm(pre_seq_s)):
            mention_data = []
            dev_line = json_lines[data_ind]
            dev_text = pre_texts[data_ind]

            if "mention_data" in dev_line.keys():
                key_mention_ids = [m["kb_id"] for m in dev_line["mention_data"]]
            else:
                key_mention_ids = []
            self.ana_line_pre(mention_data, dev_text, pre_seq, key_mention_ids)
            dev_line["mention_data"] = mention_data
            pre_lines.append(dev_line)
        return pre_lines

    def predict2(self, json_lines):
        BEGIN = "B"
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
                    if "-" in entity_mark:
                        type_predict = entity_mark.lower().split("-")[1]
                    else:
                        type_predict = ""

                    mention = {
                        "kb_id": "",
                        "mention": "",
                        "offset": str(token_id),
                        "label": "-1"
                    }

                    # 这种方法提取实体可能有重复现象 ?
                    for token_id_i in range(token_id, len(pre_seq)):
                        if pre_seq[token_id_i] is NONE \
                                or (pre_seq[token_id_i].startswith(BEGIN) and token_id_i != token_id) \
                                or (pre_seq[token_id_i].startswith(SINGLE) and token_id_i != token_id):
                            subject_id_s = self.get_subject_ids(mention["mention"], type_predict)
                            for subject_id in subject_id_s:
                                mention["kb_id"] = str(subject_id)

                                if subject_id in key_mention_ids:
                                    mention["label"] = "1"
                                else:
                                    mention["label"] = "0"
                                if not self.save_label:
                                    mention.pop("label")
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
                                if not self.save_label:
                                    mention.pop("label")
                                mention_data.append(mention.copy())
            fd_str_pairs = list(self.dsc_token("".join(dev_text)))
            # expand_pair_s = list(self.explan_mention_text(fd_str_pair_s=fd_str_pairs.copy()))

            mention_data = self.add_mention_data(mention_data,
                                                 str_pairs=fd_str_pairs,
                                                 key_mention_ids=key_mention_ids)

            """expand all"""
            expand_pair_s = list(self.explan_mention_text(
                fd_str_pair_s=[(men_data["mention"], men_data["offset"]) for men_data in mention_data]))
            mention_data = self.add_mention_data(mention_data,
                                                 str_pairs=expand_pair_s,
                                                 key_mention_ids=key_mention_ids)

            # self.add_mention_data(mention_data,
            #                       fd_str_pairs=list(self.eng_token("".join(dev_text))),
            #                       key_mention_ids=key_mention_ids)

            dev_line["mention_data"] = mention_data
            pre_lines.append(dev_line)

        return pre_lines

    def predict_text(self, pre_texts):
        return self._model.predict(pre_texts, batch_size=self.batch)


def main():
    dev_path = "D:/data/biendata/ccks2019_el/ccks_train_data/validate.json"
    model_path = r"D:\data\biendata\ccks2019_el\ner_model"
    # crfer = BiLSTMCRFPredicter(model_path)
    # crfer.predict_devs(dev_path, dev_path+".pre.txt")


if __name__ == '__main__':
    main()

import json
import pandas as pd
import pickle
import re
from typing import List

import dill
import jieba
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm
import matchzoo as mz

from ccks_all.cut_text import kb_all_text_dic, all_text_dic
from ccks_all.eval import eval_file
from ccks_all.predict.predict import Discriminater, Predicter, BiLSTMCRFPredicter

from ccks_all.static import subject_dict, subject_index, id2entity


class NgramPredicter(Predicter):

    def load_entity_set(self):
        spic_dic_path = "D:/data/biendata/ccks2019_el/ccks_train_data/entity.cspace.fdic.txt"

        self.entity_set.update(
            open(spic_dic_path, "r", encoding="utf-8").readlines()
        )
        dic_path = "D:/data/biendata/ccks2019_el/ccks_train_data/entity.fdic.txt"
        self.entity_set.update(
            open(dic_path, "r", encoding="utf-8").readlines()
        )

    def __init__(self):
        """
            mention : entity_json_line
        """
        self.entity_set = set()
        self.subject_dic = subject_dict
        self.load_entity_set()

    def dsc_token(self, text: str):
        for i in range(len(text)):
            token = []
            for j in range(i, len(text)):
                token.append(text[j])
                yield "".join(token), i

    def _save_tk_result_tf(self, tokens, mention_data_set, key_mention_ids):
        for token in tokens:
            mention_text = token[0]
            mention_offset = str(token[1])
            if mention_text in self.subject_dic.keys():
                entie_s = self.subject_dic.get(mention_text, [])
                for entie_json in entie_s:
                    subject_id = entie_json["subject_id"]

                    if subject_id in key_mention_ids:
                        mention_data_set.append({
                            "kb_id": subject_id,
                            "mention": mention_text,
                            "offset": mention_offset,
                            "label": "1"
                        })
                    else:
                        mention_data_set.append({
                            "kb_id": subject_id,
                            "mention": mention_text,
                            "offset": mention_offset,
                            "label": "0"
                        })

    def pre_one(self, json_line):
        mention_data = []
        text = json_line["text"].lower()
        key_mention_ids = [m["kb_id"] for m in json_line["mention_data"]]
        reg_result = self.dsc_token(text)
        self._save_tk_result_tf(reg_result, mention_data, key_mention_ids)

        json_line["mention_data"] = list(mention_data)
        return json_line

    def predict(self, json_lines):
        return list(map(self.pre_one, json_lines))


class CutPredicter(Predicter):

    @classmethod
    def load_jieba_dic(cls):
        jieba.initialize()
        spic_dic_path = "D:/data/biendata/ccks2019_el/ccks_train_data/entity.cspace.fdic.txt"
        for word in open(spic_dic_path, "r", encoding="utf-8"):
            # jieba.add_word(word.strip(), freq=len(word.strip())*100, tag="subject")
            jieba.add_word(word.strip())

        dic_path = "D:/data/biendata/ccks2019_el/ccks_train_data/entity.fdic.txt"
        for word in open(dic_path, "r", encoding="utf-8"):
            # jieba.add_word(word.strip(), freq=len(word.strip())*100, tag="subject")
            jieba.add_word(word.strip())

        print("load dic.")

    def __init__(self):
        """
            mention : entity_json_line
        """
        self.subject_dic = subject_dict
        self.load_jieba_dic()

    def dsc_token(self, text: str):
        fds = re.findall(r"《([^《|》]*)》", text)
        for fd_str in fds:
            ind = text.find(fd_str)
            yield fd_str, ind

    def eng_token(self, text: str):
        fds = re.findall(r"\b[a-z\d ]{1,100}\b", text)
        for fd_str in fds:
            ind = text.find(fd_str)
            yield fd_str, ind

    # def sim_token(self, text: str):
    #     fds = re.findall(r"\b[a-z\d ]{1,100}\b", text)
    #     for fd_str in fds:
    #         ind = text.find(fd_str)
    #         yield fd_str, ind

    def jaccard(self, text1, text2):
        text1 = set(text1)  # 去重；如果不需要就改为list
        text2 = set(text2)
        i = text1 & text2  # 交集
        u = text1 | text2  # 并集
        jaccard_coefficient = float(len(i) / len(u))
        return jaccard_coefficient

    def _save_tk_result_tf(self, tokens, mention_data_set, key_mention_ids, contain_set):
        for token in tokens:
            mention_text = token[0]
            mention_offset = str(token[1])

            if mention_text in self.subject_dic.keys():
                entie_s = self.subject_dic.get(mention_text, [])
                for entie_json in entie_s:
                    subject_id = entie_json["subject_id"]

                    if subject_id not in contain_set:
                        contain_set.add(subject_id)
                        if subject_id in key_mention_ids:
                            mention_data_set.append({
                                "kb_id": subject_id,
                                "mention": mention_text,
                                "offset": mention_offset,
                                "label": "1"
                            })
                        else:
                            mention_data_set.append({
                                "kb_id": subject_id,
                                "mention": mention_text,
                                "offset": mention_offset,
                                "label": "0"
                            })
            # else:
            # if len(mention_text) > 3:
            #     subject_id = subject_index.retrivl(mention_text)[0]
            #     if subject_id in key_mention_ids:
            #         mention_data_set.append({
            #             "kb_id": subject_id,
            #             "mention": mention_text,
            #             "offset": mention_offset,
            #             "label": "1"
            #         })
            #     else:
            #         mention_data_set.append({
            #             "kb_id": subject_id,
            #             "mention": mention_text,
            #             "offset": mention_offset,
            #             "label": "0"
            #         })

    def pre_one(self, json_line):
        mention_data = []
        text = json_line["text"]
        key_mention_ids = [m["kb_id"] for m in json_line["mention_data"]]

        contain_set = set()

        # jieba_result = jieba.tokenize(text, mode="default", HMM=False)
        jieba_result = jieba.tokenize(text, mode="search", HMM=False)
        self._save_tk_result_tf(jieba_result, mention_data, key_mention_ids, contain_set)

        reg_result = self.dsc_token(text)
        self._save_tk_result_tf(reg_result, mention_data, key_mention_ids, contain_set)

        reg_result = self.eng_token(text)
        self._save_tk_result_tf(reg_result, mention_data, key_mention_ids, contain_set)

        json_line["mention_data"] = list(mention_data)
        return json_line

    def predict(self, json_lines):
        return list(map(self.pre_one, json_lines))


class ClfDiscriminater(Discriminater):

    def __init__(self, model_dir):
        entity_dict = {}
        kb_cut_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data.hanlp.txt"
        kg_loader = tqdm(open(kb_cut_path, "r", encoding="utf-8").readlines())
        kg_loader.set_description("ClfDiscriminater load kg")
        for line in kg_loader:
            js_line = json.loads(line)
            entity_dict[js_line["subject_id"]] = js_line

        toka_path = model_dir.format(r"\toka.bin")
        toka2_path = model_dir.format(r"\type_toka.bin")
        model_path = model_dir.format(r"best_model.hdf5")

        self.entity_dict = entity_dict
        self.tk = pickle.load(open(toka_path, 'rb'))
        self.type_tk = pickle.load(open(toka2_path, 'rb'))
        self.model = load_model(model_path)

    @staticmethod
    def extract_entity_text(entity_json_line: dict) -> str:
        """
        得到 entity 描述文本
        :param entity_json_line:
        :return:
        """
        all_str = ""
        all_str += "。".join(entity_json_line["alias"])
        datas = entity_json_line["data"]
        for data in datas:
            all_str += "。".join(data.values())
        all_str = all_str.replace("摘要", "。")
        return all_str

    def filt_json_line(self, tdata, batchsize):

        max_len_q = 50
        max_len_d = 800
        type_len = 2

        X_query = []
        X_doc = []
        X_type = []

        # tdata = json.loads(json_line)
        query_text = " ".join(jieba.cut(tdata["text"])).strip()

        mention_data = tdata["mention_data"]
        mention_data_dic = {mention["kb_id"]: mention for mention in mention_data}
        mention_ind_dic = {}
        for i, kb_id in enumerate(mention_data_dic.keys()):
            mention_ind_dic[i] = kb_id
            entity_data = self.entity_dict[kb_id]
            types = entity_data["type"]
            doc_text = self.extract_entity_text(entity_data)

            X_query.append(query_text)
            X_doc.append(doc_text)
            X_type.append(types)

        query_text_tokenized = self.tk.texts_to_sequences(X_query)
        X_query_text_pad = pad_sequences(query_text_tokenized, maxlen=max_len_q)
        doc_text_tokenized = self.tk.texts_to_sequences(X_doc)
        X_doc_text_pad = pad_sequences(doc_text_tokenized, maxlen=max_len_d)
        type_tokenized = self.type_tk.texts_to_sequences(X_type)
        X_type_pad = pad_sequences(type_tokenized, maxlen=type_len)

        pred = self.model.predict([X_query_text_pad, X_doc_text_pad, X_type_pad], batch_size=batchsize)

        predictions = np.round(np.argmax(pred, axis=1)).astype(int)

        mention_data_n = []
        for i, pre in enumerate(predictions):
            if pre == 1:
                mention_data_n.append(mention_data_dic[mention_ind_dic[i]])
        tdata["mention_data"] = mention_data_n
        return tdata

    def predict(self, json_lines):
        # return list(map(self.filt_json_line, json_lines))
        return [self.filt_json_line(line, 256) for line in tqdm(json_lines)]


class NoneTypeClfDiscriminater(Discriminater):

    def __init__(self, model_dir, batch=128):
        toka_path = model_dir.format(r"\toka.bin")
        model_path = model_dir.format(r"best_model.hdf5")

        self.tk = pickle.load(open(toka_path, 'rb'))
        self.model = load_model(model_path)
        self.batch = batch

    @staticmethod
    def extract_entity_text(entity_json_line: dict) -> str:
        """
        得到 entity 描述文本
        :param entity_json_line:
        :return:
        """
        all_str = ""
        all_str += "。".join(entity_json_line["alias"])
        datas = entity_json_line["data"]
        for data in datas:
            all_str += "。".join(data.values())
        all_str = all_str.replace("摘要", "。")
        return all_str

    def filt_json_line(self, tdata, batchsize):

        # X_query_text_pad, X_doc_text_pad, y_ohe = get_data_all_text("test", self.tk, 10000)

        max_len_q = 20
        max_len_d = 200
        X_query = []
        X_doc = []
        X_type = []

        # tdata = json.loads(json_line)
        query_text = " ".join(jieba.cut(tdata["text"])).strip()

        mention_data = tdata["mention_data"]
        if len(mention_data) >= 5:
            mention_data_dic = {mention["kb_id"]: mention for mention in mention_data}
            mention_ind_dic = {}
            for i, kb_id in enumerate(mention_data_dic.keys()):
                mention_ind_dic[i] = kb_id
                entity_data = id2entity[kb_id]
                types = entity_data["type"]
                # doc_text = self.extract_entity_text(entity_data)
                doc_text = kb_all_text_dic[kb_id]

                X_query.append(query_text)
                X_doc.append(doc_text)
                X_type.append(types)

            query_text_tokenized = self.tk.texts_to_sequences(X_query)
            X_query_text_pad = pad_sequences(query_text_tokenized, maxlen=max_len_q)
            doc_text_tokenized = self.tk.texts_to_sequences(X_doc)
            X_doc_pad = pad_sequences(doc_text_tokenized, maxlen=max_len_d)

            pred = self.model.predict([X_query_text_pad, X_doc_pad], batch_size=batchsize)

            # predictions = np.round(np.argmax(pred, axis=1)).astype(int)
            predictions = [1 if pre[0] < 0.6 else 0 for pre in pred]

            mention_data_n = []
            for i, pre in enumerate(predictions):
                if pre == 1:
                    mention_data_n.append(mention_data_dic[mention_ind_dic[i]])
            tdata["mention_data"] = mention_data_n

        return tdata

    def predict(self, json_lines):
        # return list(map(self.filt_json_line, json_lines))
        return [self.filt_json_line(line, self.batch) for line in tqdm(json_lines)]


class RankPredicter(Discriminater):

    def __init__(self, model_path, batch_size=8):

        preprocess_path = model_path + "preprocessor.dill"
        self.model = mz.load_model(model_path+"model")
        self.preprocessor = dill.load(open(preprocess_path, "rb"))
        self.batch_size = batch_size

    def filt_json_line(self, tdata):

        id_set = set()

        X_left = []
        X_left_id = []
        X_right = []
        X_right_id = []

        y = []

        text_id = tdata["text_id"]
        query_text = all_text_dic[text_id]
        mention_data = tdata["mention_data"]

        if len(mention_data) == 0:
            return tdata

        # id : mention
        mention_data_dic = {mention["kb_id"]: mention for mention in mention_data}
        for mention in mention_data:

            text_id_subject = text_id + mention["mention"]
            kb_id = mention["kb_id"]
            doc_text = kb_all_text_dic[kb_id]

            y_label = -1
            if "label" in mention.keys():
                y_label = int(mention["label"])

            pid = text_id + "_" + kb_id

            if pid not in id_set:
                id_set.add(pid)

                X_left.append(query_text)
                X_right.append(doc_text)
                X_left_id.append(text_id_subject)
                X_right_id.append(kb_id)
                y.append(y_label)

        df = pd.DataFrame({
            'text_left': X_left,
            'text_right': X_right,
            'id_left': X_left_id,
            'id_right': X_right_id,
            'label': y
        })
        pack_raw = mz.pack(df)
        pack_processed = self.preprocessor.transform(pack_raw, verbose=0)

        X, y = pack_processed.unpack()

        pred = self.model.predict(X, batch_size=self.batch_size)
        predictions = np.squeeze(pred, axis=1)

        id_left = X["id_left"]
        id_right = X["id_right"]

        mention_data_n = []

        id_left_index_dic = {}
        for i, id_left_ in enumerate(id_left):
            if id_left_ not in id_left_index_dic.keys():
                id_left_index_dic[id_left_] = [i]
            else:
                id_left_index_dic[id_left_].append(i)

        for id_left_indexs in id_left_index_dic.values():
            # id_right_s = [id_right[i] for i in id_left_indexs]
            # predictions_s = [predictions[i] for i in id_left_indexs]
            id_right_pre_dic = {}
            for i in id_left_indexs:
                id_right_ = id_right[i]
                predictions_ = predictions[i]
                id_right_pre_dic[id_right_] = predictions_
            sort_id_right_pre_dic = sorted(id_right_pre_dic.items(), key=lambda item: item[1], reverse=True)
            entity_id = sort_id_right_pre_dic[0][0]

            mention_data_n.append(mention_data_dic[entity_id])

        tdata["mention_data"] = mention_data_n

        return tdata

    def predict(self, json_lines: List) -> List:
        return [self.filt_json_line(line) for line in tqdm(json_lines)]


def step2():
    # key_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json"
    key_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json"

    dev_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json.ner.pre.json"
    # dev_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json.jieba.rank.filter.json"
    # dev_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json.ner.rank.filter.json"
    result_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json.ner.pre.rank.json"

    # result_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json.jieba.rank.pre.filter0.6.json"

    # model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m11\{}"

    # model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m18\{}"
    # cd = NoneTypeClfDiscriminater(model_dir, batch=256)
    model_dir = r"D:/data/biendata/ccks2019_el/entityrank/m2/"
    cd = RankPredicter(model_dir, batch_size=256)
    cd.predict_devs(dev_path, result_path)

    eval_file(key_path, result_path)


def step1():
    dev_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json"
    result_path = "D:/data/biendata/ccks2019_el/ccks_train_data/test.json.ner.pre2.json"

    """crf"""
    crf_model_path = r"D:\data\biendata\ccks2019_el\ner\m0.0"
    crfer = BiLSTMCRFPredicter(crf_model_path, type_filter=False)

    """jieba"""
    # crfer = CutPredicter()

    """ngram"""
    # crfer = NgramPredicter()

    crfer.predict_devs(dev_path, result_path)
    # eval_pre_text(dev_path, result_path)
    eval_file(dev_path, result_path)


def main():
    step1()
    # step2()


if __name__ == '__main__':
    main()

"""
ngram
f:0.040384977029712096
p:0.020934924269486652
r:0.9782572577628117

jieba
f:0.10971752110381326
p:0.06186784640974652
r:0.970634415584413



ner_tpf text
f:0.8268713155157437
p:0.9438671957671914
r:0.7967485222596271

ner_tpf id
f:0.6958460835460863
p:0.7975011463844766
r:0.6726364411308885


ner text
f:0.8464500719444991
p:0.937638095238091
r:0.8261534605312315

ner id
f:0.46031319051319
p:0.5167473544973543
r:0.453965886376993

"""

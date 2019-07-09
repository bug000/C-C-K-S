import json

from keras_bert import Tokenizer
import numpy as np

from ccks_all.cut_text import kb_all_text_dic_char


class PairTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class Preprocess(object):

    @staticmethod
    def seq_padding(X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])

    def process_line(self, **kwargs):
        raise NotImplementedError


class BertPreProcess(Preprocess):

    def __init__(self, tk: PairTokenizer):
        self.tk = tk

    def process_line(self, query_text, doc_text):
        return self.tk.encode(first=query_text, second=doc_text, max_len=512)

    def process_json(self, json_line):
        tdata = json.loads(json_line)
        query_text = tdata["text"]
        X1, X2 = [], []
        mention_data = tdata["mention_data"]
        for i, mention in enumerate(mention_data):
            kb_id = mention["kb_id"]
            doc_text = kb_all_text_dic_char[kb_id]
            x1, x2 = self.process_line(query_text, doc_text)
            X1.append(x1)
            X2.append(x2)

        X = [self.seq_padding(X1.copy()), self.seq_padding(X2.copy())]
        return X


class BertNerProcess(Preprocess):

    def __init__(self, tk: PairTokenizer):
        self.tk = tk

    def process_line(self, query_text):
        return self.tk.encode(first=query_text, max_len=40)

    def process_json(self, json_line):
        tdata = json.loads(json_line)
        query_text = tdata["text"]
        X1, X2 = [], []
        # mention_data = tdata["mention_data"]
        # for i, mention in enumerate(mention_data):
            # offset = mention["offset"]
            # mention = mention["mention"]
        x1, x2 = self.process_line(query_text)
        X1.append(x1)
        X2.append(x2)

        X = [self.seq_padding(X1.copy()), self.seq_padding(X2.copy())]
        return X


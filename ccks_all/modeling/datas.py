import json
import jieba
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from ccks_all.cut_text import train_text_dic, all_text_dic, kb_all_text_dic
from ccks_all.static import id2entity

root_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}"


def get_data_all_text(data_type: str, tk: Tokenizer, line_nub=-1):
    max_len_q = 20
    max_len_d = 200
    id_set = set()

    X_query = []
    X_doc = []

    y = []

    ohe = OneHotEncoder(sparse=False, categories='auto')
    ohe.fit(np.asarray([0, 1]).reshape(-1, 1))

    json_line_s = open(root_dir.format(data_type + ".json.jieba.pre.json"), "r", encoding="utf-8").readlines()

    query_data_loder = tqdm(json_line_s)
    query_data_loder.set_description("load query data lines %s" % data_type)
    for json_line in query_data_loder:
        tdata = json.loads(json_line)
        """
            {
                "text_id": "42094",
                "text": "如何评价乔·约翰逊?",
                "mention_data": [{
                    "kb_id": "30462",
                    "mention": "如何",
                    "offset": "0",
                    "label": "0"
                }, {
                    "kb_id": "30462",
                    "mention": "如何",
                    "offset": "0",
                    "label": "0"
                }
            }
        """
        text_id = tdata["text_id"]
        query_text = all_text_dic[text_id]

        mention_data = tdata["mention_data"]
        for mention in mention_data:
            kb_id = mention["kb_id"]
            # entity_data = id2entity[kb_id]
            """
                {
                    "alias": ["王超"],
                    "subject_id": "10006",
                    "subject": "王超",
                    "type": ["Human"],
                    "data": [{
                        "predicate": "义项 描述 ",
                        "object": "齐鲁书画研究院 书画家 "
                    }, {
                        "predicate": "标签 ",
                        "object": "人物 "
                    }, {
                        "predicate": "标签 ",
                        "object": "艺术家 "
                    }]
                }
            """
            # types = entity_data["type"]
            doc_text = kb_all_text_dic[kb_id]

            y_label = int(mention["label"])

            pid = text_id + "_" + kb_id

            if len(id_set) == line_nub:
                break

            if pid not in id_set:
                id_set.add(pid)

                X_query.append(query_text)
                X_doc.append(doc_text)
                y.append(y_label)
        else:
            continue
        break

    query_text_tokenized = tk.texts_to_sequences(X_query)
    doc_text_tokenized = tk.texts_to_sequences(X_doc)
    print("trans to token .")

    X_query_text_pad = pad_sequences(query_text_tokenized, maxlen=max_len_q)
    X_doc_text_pad = pad_sequences(doc_text_tokenized, maxlen=max_len_d)
    y_ohe = ohe.transform(np.asarray(y).reshape(-1, 1))

    print("load data .")
    return [X_query_text_pad, X_doc_text_pad], y_ohe


def get_data_all_char(data_type: str, tk: Tokenizer, line_nub=-1):
    max_len_q = 50
    max_len_d = 500

    entity_text_dict = load_cut_text("kb_data.all.jieba.text.tsv", col=1)

    id_set = set()

    X_query = []
    X_doc = []

    y = []

    ohe = OneHotEncoder(sparse=False, categories='auto')
    ohe.fit(np.asarray([0, 1]).reshape(-1, 1))

    json_line_s = open(root_dir.format(data_type + ".json.jieba.pre.json"), "r", encoding="utf-8").readlines()

    query_data_loder = tqdm(json_line_s)
    query_data_loder.set_description("load query data lines")
    for json_line in query_data_loder:
        tdata = json.loads(json_line)
        """
            {
                "text_id": "42094",
                "text": "如何评价乔·约翰逊?",
                "mention_data": [{
                    "kb_id": "30462",
                    "mention": "如何",
                    "offset": "0",
                    "label": "0"
                }, {
                    "kb_id": "30462",
                    "mention": "如何",
                    "offset": "0",
                    "label": "0"
                }
            }
        """
        text_id = tdata["text_id"]
        query_text = tdata["text"]
        query_text_sp = " ".join(list(query_text)).strip()

        mention_data = tdata["mention_data"]
        for mention in mention_data:
            kb_id = mention["kb_id"]
            """
                {
                    "alias": ["王超"],
                    "subject_id": "10006",
                    "subject": "王超",
                    "type": ["Human"],
                    "data": [{
                        "predicate": "义项 描述 ",
                        "object": "齐鲁书画研究院 书画家 "
                    }, {
                        "predicate": "标签 ",
                        "object": "人物 "
                    }, {
                        "predicate": "标签 ",
                        "object": "艺术家 "
                    }]
                }
            """
            # 加载未分词的 doc_text
            doc_text = entity_text_dict[kb_id]
            doc_text_sp = " ".join(list(doc_text)).strip()

            y_label = int(mention["label"])

            pid = text_id + "_" + kb_id

            if len(id_set) == line_nub:
                break

            if pid not in id_set:
                id_set.add(pid)

                X_query.append(query_text)
                X_doc.append(doc_text)
                y.append(y_label)
        else:
            continue
        break

    query_text_tokenized = tk.texts_to_sequences(X_query)
    doc_text_tokenized = tk.texts_to_sequences(X_doc)

    X_query_text_pad = pad_sequences(query_text_tokenized, maxlen=max_len_q)
    X_doc_text_pad = pad_sequences(doc_text_tokenized, maxlen=max_len_d)

    y_ohe = ohe.transform(np.asarray(y).reshape(-1, 1))

    print("load data .")
    return [X_query_text_pad, X_doc_text_pad], y_ohe


def get_data_multi_text(data_type: str, tk: Tokenizer, tktype: Tokenizer, line_nub=-1):
    max_len_q = 20
    max_len_subject = 5
    max_len_d = 200
    type_len = 2

    id_set = set()

    X_query = []
    X_doc = []
    X_subject = []
    X_type = []

    y = []

    ohe = OneHotEncoder(sparse=False, categories='auto')
    ohe.fit(np.asarray([0, 1]).reshape(-1, 1))

    json_line_s = open(root_dir.format(data_type + ".json.jieba.pre.json"), "r", encoding="utf-8").readlines()

    query_data_loder = tqdm(json_line_s)
    query_data_loder.set_description("load query data lines %s" % data_type)
    for json_line in query_data_loder:
        tdata = json.loads(json_line)
        """
            {
                "text_id": "42094",
                "text": "如何评价乔·约翰逊?",
                "mention_data": [{
                    "kb_id": "30462",
                    "mention": "如何",
                    "offset": "0",
                    "label": "0"
                }, {
                    "kb_id": "30462",
                    "mention": "如何",
                    "offset": "0",
                    "label": "0"
                }
            }
        """
        text_id = tdata["text_id"]
        query_text = all_text_dic[text_id]

        mention_data = tdata["mention_data"]
        for mention in mention_data:
            kb_id = mention["kb_id"]
            entity_data = id2entity[kb_id]
            """
                {
                    "alias": ["王超"],
                    "subject_id": "10006",
                    "subject": "王超",
                    "type": ["Human"],
                    "data": [{
                        "predicate": "义项 描述 ",
                        "object": "齐鲁书画研究院 书画家 "
                    }, {
                        "predicate": "标签 ",
                        "object": "人物 "
                    }, {
                        "predicate": "标签 ",
                        "object": "艺术家 "
                    }]
                }
            """
            types = entity_data["type"]
            doc_text = kb_all_text_dic[kb_id]
            subject_text = " ".join(jieba.cut(mention["mention"])).strip()

            y_label = int(mention["label"])

            pid = text_id + "_" + kb_id

            if len(id_set) == line_nub:
                break

            if pid not in id_set:
                id_set.add(pid)

                X_query.append(query_text)
                X_doc.append(doc_text)
                X_subject.append(subject_text)
                X_type.append(types)

                y.append(y_label)
        else:
            continue
        break

    query_text_tokenized = tk.texts_to_sequences(X_query)
    doc_text_tokenized = tk.texts_to_sequences(X_doc)
    subject_text_tokenized = tk.texts_to_sequences(X_subject)
    type_tokenized = tktype.texts_to_sequences(X_type)
    print("trans to token .")

    X_query_text_pad = pad_sequences(query_text_tokenized, maxlen=max_len_q)
    X_doc_text_pad = pad_sequences(doc_text_tokenized, maxlen=max_len_d)
    X_subject_text_pad = pad_sequences(subject_text_tokenized, maxlen=max_len_subject)
    X_type_pad = pad_sequences(type_tokenized, maxlen=type_len)

    y_ohe = ohe.transform(np.asarray(y).reshape(-1, 1))

    print("load data .")
    return [X_query_text_pad, X_doc_text_pad, X_subject_text_pad, X_type_pad], y_ohe



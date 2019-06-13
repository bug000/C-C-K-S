import json
import pickle
import random

import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, GRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

import jieba
from tqdm import tqdm

jieba.initialize()
# jieba.enable_parallel(10)

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

kb_cut_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data.hanlp.txt"

# embedding_path = "D:/data/word2vec/zh/test.txt"
embedding_path = "D:/data/word2vec/zh/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"

# train_dir = r"C:\python3workspace\kera_ner_demo\ccks_ner\modeling\pair_model\dt\m3\{}"
model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m11\{}"
log_filepath = model_dir.format(r"log")
toka_path = model_dir.format(r"\toka.bin")
toka2_path = model_dir.format(r"\type_toka.bin")
model_path = model_dir.format(r"bilstm_model.hdf5")

max_len_q = 50
max_len_d = 800
type_len = 2

embed_size = 300
max_features = 20000

tk = Tokenizer(lower=True, filters='')
type_tk = Tokenizer(lower=True, filters='')
root_path = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}.json.jieba.pre.json"


def get_layer(inp_a, inp_object_s, inp_type, ):
    def load_embedding(toka):
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf-8"))

        word_index = toka.word_index
        nb_words = min(max_features, len(word_index))
        embedding_matrix_ = np.zeros((nb_words + 1, embed_size))
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None: embedding_matrix_[i] = embedding_vector
        return embedding_matrix_, nb_words

    embedding_matrix, nb_words = load_embedding(tk)
    embed_layer_a = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)
    embed_layer_b = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)

    x_a = embed_layer_a(inp_a)
    x_b = embed_layer_b(inp_object_s)

    x_a = SpatialDropout1D(0.3)(x_a)
    x_b = SpatialDropout1D(0.3)(x_b)

    xc_a = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x_a)
    xc_b = Bidirectional(CuDNNLSTM(512, return_sequences=True))(x_b)

    xc_a_3 = Conv1D(16, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(xc_a)
    xc_a_2 = Conv1D(16, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(xc_a)

    xc_b_3 = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(xc_b)
    xc_b_2 = Conv1D(64, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(xc_b)

    avg_pool_a3 = GlobalAveragePooling1D()(xc_a_3)
    max_pool_a3 = GlobalMaxPooling1D()(xc_a_3)
    avg_pool_a2 = GlobalAveragePooling1D()(xc_a_2)
    max_pool_a2 = GlobalMaxPooling1D()(xc_a_2)

    avg_pool_b3 = GlobalAveragePooling1D()(xc_b_3)
    max_pool_b3 = GlobalMaxPooling1D()(xc_b_3)
    avg_pool_b2 = GlobalAveragePooling1D()(xc_b_2)
    max_pool_b2 = GlobalMaxPooling1D()(xc_b_2)

    x_a = concatenate([avg_pool_a3, max_pool_a3, avg_pool_a2, max_pool_a2])
    x_a = BatchNormalization()(x_a)
    x_a = Dropout(0.3)(Dense(32, activation='relu')(x_a))

    x_b = concatenate([avg_pool_b3, max_pool_b3, avg_pool_b2, max_pool_b2])
    x_b = BatchNormalization()(x_b)
    x_b = Dropout(0.1)(Dense(64, activation='relu')(x_b))

    x = concatenate([x_a, x_b, inp_type])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(64, activation='relu')(x))

    out = Dense(2, activation="sigmoid")(x)
    return out


def load_vocab():
    global tk
    global type_tk
    tk = pickle.load(open(toka_path, 'rb'))
    type_tk = pickle.load(open(toka2_path, 'rb'))
    print("load toka")


def build_vocab():
    full_text = []
    full_types = []

    """ kb vocab """
    kb_lines = open(kb_cut_path, "r", encoding="utf-8").readlines()
    load_kb = tqdm(kb_lines)
    load_kb.set_description("load kg")
    for line in load_kb:
        kb_data_s = json.loads(line)

        data = kb_data_s["data"]
        for kb_data in data:
            full_text.extend(kb_data.values())

        types = kb_data_s["type"]
        full_types.extend(types)

    """ query vocab """
    train_json_lines = open(root_path.format("train"), "r", encoding="utf-8").readlines()
    test_json_lines = open(root_path.format("test"), "r", encoding="utf-8").readlines()
    val_json_lines = open(root_path.format("validate"), "r", encoding="utf-8").readlines()

    def cut(json_line):
        text = json.loads(json_line)["text"].lower()
        return " ".join(jieba.cut(text)).strip()

    train_texts = [cut(json_line) for json_line in tqdm(train_json_lines)]
    test_texts = [cut(json_line) for json_line in tqdm(test_json_lines)]
    val_texts = [cut(json_line) for json_line in tqdm(val_json_lines)]

    full_text.extend(train_texts)
    full_text.extend(test_texts)
    full_text.extend(val_texts)
    full_text = list(set(full_text))
    full_text = list(filter(lambda text: len(text) > 10, full_text))
    tk.fit_on_texts(full_text)

    full_types = list(set(full_types))
    type_tk.fit_on_texts(full_types)

    pickle.dump(tk, open(toka_path, 'wb'))
    pickle.dump(type_tk, open(toka2_path, 'wb'))
    print("build_vocab")


def get_data_generator(data_type: str):
    ohe = OneHotEncoder(sparse=False)

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

    entity_dict = {}
    kg_loader = tqdm(open(kb_cut_path, "r", encoding="utf-8").readlines())
    kg_loader.set_description("load kg")
    for line in kg_loader:
        js_line = json.loads(line)
        entity_dict[js_line["subject_id"]] = js_line

    json_line_s = open(root_path.format(data_type), "r", encoding="utf-8").readlines()

    for json_line in json_line_s:
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
        query_text = " ".join(jieba.cut(tdata["text"])).strip()
        query_text_tokenized = tk.texts_to_sequences([query_text])
        X_query_text_pad = pad_sequences(query_text_tokenized, maxlen=max_len_q)

        mention_data = tdata["mention_data"]
        for mention in mention_data:
            kb_id = mention["kb_id"]
            entity_data = entity_dict[kb_id]
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
            type_tokenized = type_tk.texts_to_sequences(types)
            X_type_pad = pad_sequences(type_tokenized, maxlen=2)

            doc_text = extract_entity_text(entity_data)
            doc_text = " ".join(jieba.cut(doc_text)).strip()
            doc_text_tokenized = tk.texts_to_sequences([doc_text])
            X_doc_text_pad = pad_sequences(doc_text_tokenized, maxlen=max_len_d)

            y_label = mention["label"]
            y_label = ohe.fit_transform(y_label)

            yield X_query_text_pad, X_doc_text_pad, X_type_pad, y_label


def get_data_all(data_type: str, line_nub=-1):
    id_set = set()

    X_query = []
    X_doc = []
    X_type = []
    y = []

    ohe = OneHotEncoder(sparse=False)

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

    entity_dict = {}
    kg_loader = tqdm(open(kb_cut_path, "r", encoding="utf-8").readlines())
    kg_loader.set_description("load kg")
    for line in kg_loader:
        js_line = json.loads(line)
        entity_dict[js_line["subject_id"]] = js_line

    json_line_s = open(root_path.format(data_type), "r", encoding="utf-8").readlines()

    data_loder = tqdm(json_line_s)
    data_loder.set_description("load data lines")
    for json_line in data_loder:
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
        query_text = " ".join(jieba.cut(tdata["text"])).strip()
        text_id = tdata["text_id"]

        mention_data = tdata["mention_data"]
        for mention in mention_data:
            kb_id = mention["kb_id"]
            entity_data = entity_dict[kb_id]
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

            doc_text = extract_entity_text(entity_data)
            doc_text = " ".join(jieba.cut(doc_text)).strip()

            y_label = int(mention["label"])
            # y_label = ohe.fit_transform(y_label)

            pid = text_id + "_" + kb_id

            if len(id_set) == line_nub:
                break

            if pid not in id_set:
                id_set.add(pid)

                X_query.append(query_text)
                X_doc.append(doc_text)
                X_type.append(types)
                y.append(y_label)
        else:
            continue
        break

    query_text_tokenized = tk.texts_to_sequences(X_query)
    X_query_text_pad = pad_sequences(query_text_tokenized, maxlen=max_len_q)
    doc_text_tokenized = tk.texts_to_sequences(X_doc)
    X_doc_text_pad = pad_sequences(doc_text_tokenized, maxlen=max_len_d)
    type_tokenized = type_tk.texts_to_sequences(X_type)
    X_type_pad = pad_sequences(type_tokenized, maxlen=type_len)
    y_ohe = ohe.fit_transform(np.asarray(y).reshape(-1, 1))

    return X_query_text_pad, X_doc_text_pad, X_type_pad, y_ohe


def build_model(lr=0.0, lr_d=0.0):
    """data"""
    # build_vocab()
    load_vocab()

    # train_data_generator = get_data_generator("train")
    X_query_text_pad, X_doc_text_pad, X_type_pad, y_ohe = get_data_all("train", 100000)

    """layers"""
    inp_a = Input(shape=(max_len_q,))
    inp_object_s = Input(shape=(max_len_d,))
    inp_type = Input(shape=(type_len,))
    out = get_layer(inp_a, inp_object_s, inp_type)

    """call back"""
    check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    # early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    tb_cb = TensorBoard(log_dir=log_filepath)

    """fine-tune"""
    model = Model(inputs=[inp_a, inp_object_s, inp_type], outputs=out)
    # model.trainable = True
    # for layer in model.layers[:1]:
    #     layer.trainable = False
    # model.summary()

    """train"""
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model.fit(x=[X_query_text_pad, X_doc_text_pad, X_type_pad],
              y=y_ohe,
              batch_size=24,
              epochs=20,
              validation_split=0.5,
              verbose=1,
              class_weight="auto",
              callbacks=[check_point, early_stop, tb_cb])
    # model.fit_generator(train_data_generator,
    #                     batch_size=24,
    #                     epochs=20,
    #                     validation_split=0.3,
    #                     verbose=1,
    #                     class_weight={0: 1, 1: 10},
    #                     callbacks=[check_point, early_stop, tb_cb])
    K.clear_session()
    tf.reset_default_graph()

    model = load_model(model_path)
    return model


def get_model():
    model = load_model(model_path)
    return model


def predict(model):
    X_query_text_pad, X_doc_text_pad, X_type_pad, y_ohe = get_data_all("test", -1)
    pred = model.predict([X_query_text_pad, X_doc_text_pad, X_type_pad], batch_size=1024)

    predictions = np.round(np.argmax(pred, axis=1)).astype(int)
    y_test = np.round(np.argmax(y_ohe, axis=1)).astype(int)

    report = classification_report(y_test, predictions)

    print(report)


def main():
    td_model = build_model(lr=1e-4, lr_d=0)
    # td_model = get_model()
    predict(td_model)


if __name__ == '__main__':
    main()

# pred = td_model.predict([X_test_a, X_test_b], batch_size=1024)
# predictions = np.round(np.argmax(pred, axis=1)).astype(int)
#
# report = classification_report(y_test, predictions)
#
# print(report)

"""

no embedding m4

              precision    recall  f1-score   support

           0       0.97      0.48      0.65    179485
           1       0.27      0.94      0.42     36257

    accuracy                           0.56    215742
   macro avg       0.62      0.71      0.53    215742
weighted avg       0.86      0.56      0.61    215742


embedding m5
              precision    recall  f1-score   support

           0       0.99      0.78      0.87    179485
           1       0.47      0.94      0.63     36257

    accuracy                           0.81    215742
   macro avg       0.73      0.86      0.75    215742
weighted avg       0.90      0.81      0.83    215742



m6
              precision    recall  f1-score   support

           0       0.98      0.83      0.90    179485
           1       0.52      0.93      0.67     36257

    accuracy                           0.84    215742
   macro avg       0.75      0.88      0.78    215742
weighted avg       0.91      0.84      0.86    215742



"""

import json
import pickle
import random

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, GRU, CuDNNLSTM, BatchNormalization, Multiply, \
    CuDNNGRU, Dot
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score

from tqdm import tqdm

from ccks_all.cut_text import train_text_dic, kb_text_dic, all_text_dic, kb_predicate_dic, kb_tag_dic, kb_all_text_dic
from ccks_all.modeling.utils import Metrics, f1
from ccks_all.static import id2entity

from keras import backend as K


seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

root_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}"
# embedding_path = "D:/data/word2vec/zh/test.txt"
embedding_path = "D:/data/word2vec/zh/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"
predicate_embedding_path = "D:/data/biendata/ccks2019_el/ccks_train_data/kb_data.predicate.vec.txt"

# train_dir = r"C:\python3workspace\kera_ner_demo\ccks_ner\modeling\pair_model\dt\m3\{}"
model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m16\{}"
log_filepath = model_dir.format(r"log")
toka_path = model_dir.format(r"\toka.bin")
toka_type_path = model_dir.format(r"\type_toka.bin")
toka_predicate_path = model_dir.format(r"\predicate_toka.bin")
model_path = model_dir.format(r"best_model.hdf5")

max_len_q = 20
max_len_d = 200
max_predicate_nub = 10
max_tag_text_nub = 10
type_len = 2

embed_size = 300

tk = Tokenizer(lower=True, filters='')
tk_type = Tokenizer(lower=True, filters='')
tk_predicate = Tokenizer(lower=True, filters='')


def get_layer(inp_a, inp_object_s, inp_type, inp_predicate, inp_tag):

    def load_embedding(toka, max_features):

        def get_coefs(token, *arr):
            return token, np.asarray(arr, dtype='float32')

        embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding="utf-8"))

        word_index = toka.word_index
        nub_words = min(max_features, len(word_index))
        embedding_matrix_ = np.zeros((nub_words + 1, embed_size))
        for word, i in word_index.items():
            if i >= max_features:
                continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_[i] = embedding_vector
        return embedding_matrix_, nub_words

    def get_pooling(x):
        avg_pool_x = GlobalAveragePooling1D()(x)
        max_pool_x = GlobalMaxPooling1D()(x)
        return avg_pool_x, max_pool_x

    embedding_matrix, nb_words = load_embedding(tk, 10_0000)
    predicate_embedding_matrix, predicate_nb_words = load_embedding(tk_predicate, 41810)

    embed_layer_a = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)
    embed_layer_b = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)
    embed_layer_c = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)
    embed_predicate_layer = Embedding(predicate_nb_words + 1, embed_size, weights=[predicate_embedding_matrix], trainable=False)

    x_a = embed_layer_a(inp_a)
    x_b = embed_layer_b(inp_object_s)
    x_tag = embed_layer_c(inp_tag)
    x_predicate = embed_predicate_layer(inp_predicate)

    x_a = SpatialDropout1D(0.3)(x_a)
    x_b = SpatialDropout1D(0.3)(x_b)
    x_tag = SpatialDropout1D(0.3)(x_tag)
    x_predicate = SpatialDropout1D(0.3)(x_predicate)

    xc_a = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x_a)
    xc_b = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x_b)
    x_predicate = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x_predicate)
    x_tag = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x_tag)

    xc_a_3 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform', activation="relu")(xc_a)
    xc_a_2 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform', activation="relu")(xc_a)

    # x_entity_3 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform', activation="relu")(x_entity)
    # x_entity_2 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform', activation="relu")(x_entity)

    xc_tag_3 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform', activation="relu")(x_tag)
    xc_tag_2 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform', activation="relu")(x_tag)

    xc_b_3 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform', activation="relu")(xc_b)
    xc_b_2 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform', activation="relu")(xc_b)

    # avg_pool_entity3, max_pool_entity3 = get_pooling(x_entity_3)
    # avg_pool_entity2, max_pool_entity2 = get_pooling(x_entity_2)

    avg_pool_a3, max_pool_a3 = get_pooling(xc_a_3)
    avg_pool_a2, max_pool_a2 = get_pooling(xc_a_2)

    avg_pool_predicate, max_pool_predicate = get_pooling(x_predicate)

    avg_pool_b3, max_pool_b3 = get_pooling(xc_b_3)
    avg_pool_b2, max_pool_b2 = get_pooling(xc_b_2)
    avg_pool_tag3, max_pool_tag3 = get_pooling(xc_tag_3)
    avg_pool_tag2, max_pool_tag2 = get_pooling(xc_tag_2)

    x_tag = concatenate([avg_pool_tag3, max_pool_tag3, avg_pool_tag2, max_pool_tag2])
    x_tag = BatchNormalization()(x_tag)
    x_tag = Dropout(0.3)(Dense(32, activation='relu')(x_tag))

    x_predicate = concatenate([avg_pool_predicate, max_pool_predicate])
    x_predicate = BatchNormalization()(x_predicate)
    x_predicate = Dropout(0.3)(Dense(32, activation='relu')(x_predicate))

    x_a = concatenate([avg_pool_a3, max_pool_a3, avg_pool_a2, max_pool_a2])
    x_a = BatchNormalization()(x_a)
    x_a = Dropout(0.3)(Dense(32, activation='relu')(x_a))

    x_b = concatenate([avg_pool_b3, max_pool_b3, avg_pool_b2, max_pool_b2])
    # x_b = BatchNormalization()(x_b)
    # x_b = Dropout(0.3)(Dense(128, activation='relu')(x_b))
    x_b = BatchNormalization()(x_b)
    x_b = Dropout(0.3)(Dense(32, activation='relu')(x_b))

    x_e = concatenate([x_b, x_predicate, x_tag])
    x_e = BatchNormalization()(x_e)
    x_e = Dropout(0.1)(Dense(32, activation='relu')(x_e))

    # xm_b = Multiply()([x_a, x_b])
    # xm_b = BatchNormalization()(xm_b)
    # xm_b = Dropout(0.2)(Dense(32, activation='relu')(xm_b))
    #
    # xm_tag = Multiply()([x_a, x_tag])
    # xm_tag = BatchNormalization()(xm_tag)
    # xm_tag = Dropout(0.2)(Dense(32, activation='relu')(xm_tag))

    xm = Multiply()([x_a, x_e])
    xm = BatchNormalization()(xm)
    xm = Dropout(0.3)(Dense(32, activation='relu')(xm))
    # d1 = Dot(1)([x_a, x_e])
    # d2 = Dot(1)([x_a, x_e2])

    # xm = concatenate([xm_b, xm_e, xm_tag])
    # xm = BatchNormalization()(xm)
    # xm = Dropout(0.2)(Dense(128, activation='relu')(xm))
    # x = concatenate([x_tag, x_predicate, xm_tag, xm_b, x_a, x_b, inp_type])
    # x = concatenate([x_tag, x_predicate, x_b, inp_type])
    # x = BatchNormalization()(x)
    # x = Dropout(0.2)(Dense(128, activation='relu')(x))

    x = concatenate([x_a, x_e, xm, inp_type])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(128, activation='tanh')(x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(32, activation='tanh')(x))

    out = Dense(2, activation="sigmoid")(x)
    return out


def load_vocab():
    global tk
    global tk_type
    global tk_predicate
    tk = pickle.load(open(toka_path, 'rb'))
    tk_type = pickle.load(open(toka_type_path, 'rb'))
    tk_predicate = pickle.load(open(toka_predicate_path, 'rb'))
    print("load toka")


def build_vocab():
    full_types = []
    full_predicates = []
    for kb_data_s in id2entity.values():
        types = kb_data_s["type"]
        full_types.extend(types)
        full_predicates.extend([data["predicate"] for data in kb_data_s["data"]])
    tk_type.fit_on_texts(full_types)

    tk_predicate.fit_on_sequences(full_predicates)

    train_text = train_text_dic.values()
    kb_data_text = kb_text_dic.values()
    train_text = list(set(train_text))
    kb_data_text = list(set(kb_data_text))
    full_texts = train_text + kb_data_text
    tk.fit_on_texts(full_texts)

    pickle.dump(tk, open(toka_path, 'wb'))
    pickle.dump(tk_type, open(toka_type_path, 'wb'))
    pickle.dump(tk_predicate, open(toka_predicate_path, 'wb'))
    print("build_vocab")


def get_data_all(data_type: str, line_nub=-1):
    id_set = set()

    X_query = []
    X_predicate = []
    X_doc = []
    X_tag_text = []
    X_type = []
    y = []

    ohe = OneHotEncoder(sparse=False, categories='auto')
    ohe.fit(np.asarray([0, 1]).reshape(-1, 1))

    json_line_s = open(root_dir.format(data_type + ".json.jieba.pre.json"), "r", encoding="utf-8").readlines()

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
            # doc_text = kb_text_dic[kb_id]
            doc_text = kb_all_text_dic[kb_id]
            predicate_text_line = kb_predicate_dic[kb_id]
            tag_text_line = kb_tag_dic[kb_id]

            y_label = int(mention["label"])
            # y_label = ohe.fit_transform(y_label)

            pid = text_id + "_" + kb_id

            if len(id_set) == line_nub:
                break

            if pid not in id_set:
                id_set.add(pid)

                X_tag_text.append(tag_text_line)
                X_predicate.append(predicate_text_line)
                X_query.append(query_text)
                X_doc.append(doc_text)
                X_type.append(types)
                y.append(y_label)
        else:
            continue
        break

    query_text_tokenized = tk.texts_to_sequences(X_query)
    doc_text_tokenized = tk.texts_to_sequences(X_doc)
    tag_tokenized = tk.texts_to_sequences(X_tag_text)

    type_tokenized = tk_type.texts_to_sequences(X_type)

    predicate_tokenized = tk_predicate.texts_to_sequences(X_predicate)

    X_tag_pad = pad_sequences(tag_tokenized, maxlen=max_predicate_nub)
    X_predicate_pad = pad_sequences(predicate_tokenized, maxlen=max_predicate_nub)
    X_query_text_pad = pad_sequences(query_text_tokenized, maxlen=max_len_q)
    X_doc_text_pad = pad_sequences(doc_text_tokenized, maxlen=max_len_d)
    X_type_pad = pad_sequences(type_tokenized, maxlen=type_len)
    y_ohe = ohe.transform(np.asarray(y).reshape(-1, 1))

    print("load data .")
    return X_query_text_pad, X_doc_text_pad, X_type_pad, X_predicate_pad, X_tag_pad, y_ohe


def build_model(lr=0.0, lr_d=0.0):
    """
    data
    query 摘要 type 边 tag
    """
    # train_data_generator = get_data_generator("train")
    X_query_text_pad, X_doc_text_pad, X_type_pad, X_predicate_pad, X_tag_pad, y_ohe = get_data_all("train", -1)
    X_query_text_pad_val, X_doc_text_pad_val, X_type_pad_val, X_predicate_pad_val, X_tag_pad_val, y_ohe_val = get_data_all("validate", 100000)

    """layers"""
    inp_a = Input(shape=(max_len_q,))
    inp_object_s = Input(shape=(max_len_d,))
    inp_type = Input(shape=(type_len,))
    inp_predicate = Input(shape=(max_predicate_nub,))
    inp_tag = Input(shape=(max_tag_text_nub,))
    out = get_layer(inp_a, inp_object_s, inp_type, inp_predicate, inp_tag)

    """call back"""
    check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    tb_cb = TensorBoard(log_dir=log_filepath)
    metrics = Metrics(5)

    """fine-tune"""
    model = Model(inputs=[inp_a, inp_object_s, inp_type, inp_predicate, inp_tag], outputs=out)
    # model.trainable = True
    # for layer in model.layers[:1]:
    #     layer.trainable = False
    model.summary()

    """train"""
    # model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy", f1])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model.fit(x=[X_query_text_pad, X_doc_text_pad, X_type_pad, X_predicate_pad, X_tag_pad],
              y=y_ohe,
              batch_size=256,
              epochs=20,
              # validation_split=0.3,
              validation_data=([X_query_text_pad_val, X_doc_text_pad_val, X_type_pad_val, X_predicate_pad_val, X_tag_pad_val], y_ohe_val),
              verbose=1,
              class_weight="auto",
              # class_weight={0: 1, 1: 28},
              callbacks=[check_point, early_stop, tb_cb, metrics])

    K.clear_session()
    tf.reset_default_graph()

    model = load_model(model_path, custom_objects={'f1': f1})
    return model


def get_model():
    model = load_model(model_path)
    return model


def predict(model):
    X_query_text_pad, X_doc_text_pad, X_type_pad, X_predicate_pad, X_tag_pad, y_ohe = get_data_all("test", 10000)
    pred = model.predict([X_query_text_pad, X_doc_text_pad, X_type_pad, X_predicate_pad, X_tag_pad], batch_size=1024, verbose=1)

    predictions = np.round(np.argmax(pred, axis=1)).astype(int)
    y_test = np.round(np.argmax(y_ohe, axis=1)).astype(int)

    report = classification_report(y_test, predictions)

    print(report)


def main():
    # build_vocab()
    load_vocab()
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

embedding m15
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      9630
           1       0.49      0.32      0.39       370

    accuracy                           0.96     10000
   macro avg       0.73      0.65      0.68     10000
weighted avg       0.96      0.96      0.96     10000



              precision    recall  f1-score   support

           0       0.99      0.80      0.89      9630
           1       0.15      0.88      0.25       370

    accuracy                           0.81     10000
   macro avg       0.57      0.84      0.57     10000
weighted avg       0.96      0.81      0.86     10000



"""

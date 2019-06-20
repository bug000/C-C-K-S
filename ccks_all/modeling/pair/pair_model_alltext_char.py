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


from ccks_all.cut_text import train_text_dic, all_text_dic, kb_all_text_dic, load_cut_text
from ccks_all.modeling.datas import get_data_all_char
from ccks_all.modeling.utils import Metrics, f1

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
model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m17\{}"
log_filepath = model_dir.format(r"log")
toka_path = model_dir.format(r"toka.bin")
model_path = model_dir.format(r"best_model.hdf5")

max_len_q = 50
max_len_d = 500

embed_size = 300


def get_layer(inp_a, inp_b, tk:Tokenizer):
    def get_pooling(x):
        avg_pool_x = GlobalAveragePooling1D()(x)
        max_pool_x = GlobalMaxPooling1D()(x)
        return avg_pool_x, max_pool_x

    embed_layer_a = Embedding(len(tk.word_index) + 2, embed_size)
    embed_layer_b = Embedding(len(tk.word_index) + 2, embed_size)

    x_a = embed_layer_a(inp_a)
    x_b = embed_layer_b(inp_b)

    x_a = SpatialDropout1D(0.3)(x_a)
    x_b = SpatialDropout1D(0.3)(x_b)

    xc_a = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x_a)
    xc_b = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x_b)

    x_a_c_3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(xc_a)
    x_b_c_3 = Conv1D(64, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(xc_b)

    avg_pool_a3, max_pool_a3 = get_pooling(x_a_c_3)
    avg_pool_b3, max_pool_b3 = get_pooling(x_b_c_3)

    x_a = concatenate([avg_pool_a3, max_pool_a3])
    x_a = BatchNormalization()(x_a)
    x_a = Dropout(0.3)(Dense(32, activation='relu')(x_a))

    x_b = concatenate([avg_pool_b3, max_pool_b3])
    x_b = BatchNormalization()(x_b)
    x_b = Dropout(0.3)(Dense(32, activation='relu')(x_b))

    # xm = Multiply()([x_a, x_b])
    # xm = BatchNormalization()(xm)
    # xm = Dropout(0.3)(Dense(32, activation='relu')(xm))

    # d1 = Dot(1)([x_a, x_e])
    # d2 = Dot(1)([x_a, x_e2])

    x = concatenate([x_a, x_b])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(Dense(32, activation='relu')(x))

    out = Dense(2, activation="sigmoid")(x)
    return out


def load_vocab(toka_path: str)->Tokenizer:
    # global tk
    tk = pickle.load(open(toka_path, 'rb'))
    print("load toka")
    return tk


def build_vocab(toka_path: str):
    tk = Tokenizer(lower=True, filters='', char_level=True)
    train_text = train_text_dic.values()
    kb_data_text = kb_all_text_dic.values()
    train_text = list(set(train_text))
    kb_data_text = list(set(kb_data_text))
    full_texts = train_text + kb_data_text
    tk.fit_on_texts(full_texts)
    pickle.dump(tk, open(toka_path, 'wb'))
    print("build_vocab")
    return tk


def build_model(lr: float, lr_d: float, tk: Tokenizer):
    """data"""
    X_query_text_pad, X_doc_text_pad, y_ohe = get_data_all_char("train", tk, -1)
    X_query_text_pad_val, X_doc_text_pad_val, y_ohe_val = get_data_all_char("validate", tk, 100000)

    """layers"""
    inp_a = Input(shape=(max_len_q,))
    inp_b = Input(shape=(max_len_d,))
    out = get_layer(inp_a, inp_b, tk)

    """call back"""
    check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    tb_cb = TensorBoard(log_dir=log_filepath)
    metrics = Metrics()

    """fine-tune"""
    model = Model(inputs=[inp_a, inp_b], outputs=out)
    # model.trainable = True
    # for layer in model.layers[:1]:
    #     layer.trainable = False
    model.summary()

    """train"""
    # model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy", f1])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model.fit(x=[X_query_text_pad, X_doc_text_pad],
              y=y_ohe,
              batch_size=256,
              epochs=20,
              # validation_split=0.3,
              verbose=1,
              validation_data=([X_query_text_pad_val, X_doc_text_pad_val], y_ohe_val),
              # class_weight="auto",
              class_weight={0: 1, 1: 28},
              callbacks=[check_point, early_stop, tb_cb, metrics])
    # model.save(model_path)
    K.clear_session()
    tf.reset_default_graph()

    model = load_model(model_path, custom_objects={'f1': f1})
    return model


def get_model():
    model = load_model(model_path)
    return model


def predict(model, tk):
    X_query_text_pad, X_doc_text_pad, y_ohe = get_data_all_char("test", tk, 10000)
    pred = model.predict([X_query_text_pad, X_doc_text_pad], batch_size=1024, verbose=1)

    predictions = np.round(np.argmax(pred, axis=1)).astype(int)
    y_test = np.round(np.argmax(y_ohe, axis=1)).astype(int)

    report = classification_report(y_test, predictions)

    print(report)


def main():
    # build_vocab(toka_path)
    tk = load_vocab(toka_path)
    # td_model = build_model(lr=1e-4, lr_d=0)
    td_model = build_model(lr=1e-3, lr_d=0, tk=tk)
    # td_model = get_model()
    predict(td_model, tk=tk)


if __name__ == '__main__':
    main()

# pred = td_model.predict([X_test_a, X_test_b], batch_size=1024)
# predictions = np.round(np.argmax(pred, axis=1)).astype(int)
#
# report = classification_report(y_test, predictions)
#
# print(report)

"""


m17
              precision    recall  f1-score   support

           0       0.98      0.99      0.98      9630
           1       0.50      0.38      0.43       370

    accuracy                           0.96     10000
   macro avg       0.74      0.68      0.71     10000
weighted avg       0.96      0.96      0.96     10000




embedding m15
              precision    recall  f1-score   support

           0       0.97      0.99      0.98      9630
           1       0.49      0.32      0.39       370

    accuracy                           0.96     10000
   macro avg       0.73      0.65      0.68     10000
weighted avg       0.96      0.96      0.96     10000





"""

import json
import pickle
import random

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Conv1D, GRU, CuDNNLSTM, BatchNormalization, Multiply, \
    CuDNNGRU, Dot
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from sklearn.metrics import classification_report

from ccks_all.cut_text import train_text_dic, all_text_dic, kb_all_text_dic
from ccks_all.modeling.datas import get_data_multi_text
from ccks_all.modeling.utils import Metrics, f1
from ccks_all.static import id2entity

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

root_dir = r"D:\data\biendata\ccks2019_el\ccks_train_data\{}"
# embedding_path = "D:/data/word2vec/zh/test.txt"
embedding_path = "D:/data/word2vec/zh/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.utf8.txt"

max_len_q = 20
max_len_subject = 5
max_len_d = 200
type_len = 2

embed_size = 300


def get_layer(inp_a, inp_b, inp_subject, inp_type, tk):
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

    embed_layer_a = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)
    embed_layer_b = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)
    embed_layer_subject = Embedding(nb_words + 1, embed_size, weights=[embedding_matrix], trainable=False)

    x_a = embed_layer_a(inp_a)
    x_b = embed_layer_b(inp_b)
    x_subject = embed_layer_subject(inp_subject)

    x_a = SpatialDropout1D(0.3)(x_a)
    x_b = SpatialDropout1D(0.3)(x_b)
    x_subject = SpatialDropout1D(0.3)(x_subject)

    x_r_a = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x_a)
    x_r_b = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x_b)
    x_r_subject = Bidirectional(CuDNNLSTM(8, return_sequences=True))(x_subject)

    x_a_r_3 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_r_a)
    x_b_r_3 = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_r_b)
    x_subject_r_3 = Conv1D(8, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_r_subject)

    avg_pool_a3, max_pool_a3 = get_pooling(x_a_r_3)
    avg_pool_b3, max_pool_b3 = get_pooling(x_b_r_3)
    avg_pool_subject3, max_pool_subject3 = get_pooling(x_subject_r_3)

    x_a = concatenate([avg_pool_a3, max_pool_a3])
    x_a = BatchNormalization()(x_a)
    x_a = Dropout(0.3)(Dense(32, activation='relu')(x_a))

    x_b = concatenate([avg_pool_b3, max_pool_b3])
    x_b = BatchNormalization()(x_b)
    x_b = Dropout(0.3)(Dense(128, activation='relu')(x_b))
    x_b = BatchNormalization()(x_b)
    x_b = Dropout(0.3)(Dense(32, activation='relu')(x_b))

    x_subject = concatenate([avg_pool_subject3, max_pool_subject3])
    x_subject = BatchNormalization()(x_subject)
    x_subject = Dropout(0.3)(Dense(4, activation='relu')(x_subject))

    # xm = Multiply()([x_a, x_b])
    # xm = BatchNormalization()(xm)
    # xm = Dropout(0.3)(Dense(16, activation='relu')(xm))

    d1 = Dot(1, trainable=False)([x_a, x_b])
    # d2 = Dot(1, trainable=False)([x_subject, xm])

    # x = concatenate([inp_type, x_a, x_b, xm, x_subject])
    x = concatenate([inp_type, d1, x_subject])
    x = BatchNormalization()(x)
    x = Dropout(0.3)(Dense(32, activation='relu')(x))

    out = Dense(2, activation="sigmoid")(x)
    return out


def load_vocab(toka_path_, toka_type_path_):
    tk = pickle.load(open(toka_path_, 'rb'))
    tk2 = pickle.load(open(toka_type_path_, 'rb'))
    print("load toka")
    return tk, tk2


def build_vocab(toka_path_, toka_type_path_):
    tk = Tokenizer(lower=True, filters='')
    tk_type = Tokenizer(lower=True, filters='')
    train_text = train_text_dic.values()
    kb_data_text = kb_all_text_dic.values()
    train_text = list(set(train_text))
    kb_data_text = list(set(kb_data_text))
    full_texts = train_text + kb_data_text
    full_types = [kb_data_s["type"] for kb_data_s in id2entity.values()]
    tk.fit_on_texts(full_texts)
    tk_type.fit_on_texts(full_types)

    pickle.dump(tk, open(toka_path_, 'wb'))
    pickle.dump(tk_type, open(toka_type_path_, 'wb'))
    print("build_vocab")
    return tk, tk_type


def build_model(lr, lr_d, tk, tk_type, model_dir):
    log_filepath = model_dir.format(r"log")
    model_path = model_dir.format(r"best_model.hdf5")

    """data"""
    X, y = get_data_multi_text("train", tk, tk_type, -1)
    X_val, y_val = get_data_multi_text("validate", tk, tk_type, 10_0000)

    """layers"""
    inp_a = Input(shape=(max_len_q,))
    inp_b = Input(shape=(max_len_d,))
    inp_subject = Input(shape=(max_len_subject,))
    inp_type = Input(shape=(type_len,))
    out = get_layer(inp_a, inp_b, inp_subject, inp_type, tk)

    """call back"""
    check_point = ModelCheckpoint(model_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=1)
    tb_cb = TensorBoard(log_dir=log_filepath)
    metrics = Metrics(4)

    """fine-tune"""
    model = Model(inputs=[inp_a, inp_b, inp_subject, inp_type], outputs=out)
    # model.trainable = True
    # for layer in model.layers[:1]:
    #     layer.trainable = False
    model.summary()

    """train"""
    # model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy", f1])
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    model.fit(x=X,
              y=y,
              batch_size=256,
              epochs=20,
              validation_data=(X_val, y_val),
              # validation_split=0.1,
              verbose=1,
              # class_weight="auto",
              class_weight={0: 1, 1: 30},
              callbacks=[check_point, early_stop, tb_cb, metrics])

    K.clear_session()
    tf.reset_default_graph()

    model = load_model(model_path, custom_objects={'f1': f1})
    return model


def predict(model, tk, tk_type):
    X, y = get_data_multi_text("test", tk, tk_type, 10000)
    pred = model.predict(X, batch_size=1024, verbose=1)

    predictions = np.round(np.argmax(pred, axis=1)).astype(int)
    y_test = np.round(np.argmax(y, axis=1)).astype(int)

    report = classification_report(y_test, predictions)

    print(report)


def main():
    model_dir = r"D:\data\biendata\ccks2019_el\entityclf\m20\{}"
    toka_path = model_dir.format(r"\toka.bin")
    toka2_path = model_dir.format(r"\toka_type.bin")

    # tk, tk_type = build_vocab(toka_path, toka2_path)
    tk, tk_type = load_vocab(toka_path, toka2_path)

    td_model = build_model(lr=1e-3, lr_d=0, tk=tk, tk_type=tk_type, model_dir=model_dir)
    # td_model = get_model()
    predict(td_model, tk=tk, tk_type=tk_type)


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


m17
              precision    recall  f1-score   support

           0       0.98      0.99      0.99   1383775
           1       0.68      0.55      0.61     49772

    accuracy                           0.98   1433547
   macro avg       0.83      0.77      0.80   1433547
weighted avg       0.97      0.98      0.97   1433547

"""
